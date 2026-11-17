"""Optional same-tile, self-discovered start states. Never replays PPO samples.

Disabled by default, including evaluation. Enable explicitly with
``env.v135_stage_starts.enabled=True`` for a diagnostic training run.
"""
from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv


def batch_snapshot(value, env_id, num_envs):
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == num_envs:
        return value[env_id].detach().cpu().clone()
    if isinstance(value, dict):
        return {k: saved for k, v in value.items()
                if (saved := batch_snapshot(v, env_id, num_envs)) is not None}
    return None


def restore_batch(value, saved, env_id):
    if isinstance(saved, torch.Tensor):
        value[env_id].copy_(saved.to(value.device))
    elif isinstance(saved, dict):
        for key, child in saved.items():
            restore_batch(value[key], child, env_id)


class V135TrainingEnv(ManagerBasedRLEnv):
    def _objects_for_snapshot(self):
        yield "env", {k: v for k, v in vars(self).items()
                      if k.startswith(("_v11_", "_v13_", "_v135_")) and k != "_v135_stage_pool"}
        yield "actions", vars(self.action_manager)
        for name, term in self.action_manager._terms.items():
            yield "action_" + name, vars(term)
        yield "command", vars(self.command_manager.get_term("base_velocity"))
        for name, actuator in self.scene["robot"].actuators.items():
            yield "actuator_" + name, vars(actuator)

    def _capture_stage(self, env_id):
        asset = self.scene["robot"]
        snapshot = {name: batch_snapshot(obj, env_id, self.num_envs)
                    for name, obj in self._objects_for_snapshot()}
        snapshot["physical"] = (
            asset.data.root_state_w[env_id].detach().cpu().clone(),
            asset.data.joint_pos[env_id].detach().cpu().clone(),
            asset.data.joint_vel[env_id].detach().cpu().clone())
        snapshot["tile"] = self.scene.env_origins[env_id].detach().cpu().clone()
        snapshot["level"] = int(self.scene.terrain.terrain_levels[env_id])
        snapshot["type"] = int(self.scene.terrain.terrain_types[env_id])
        snapshot["captured_step"] = int(self.common_step_counter)
        snapshot["ring_index"] = int(getattr(self, "_v11_ring_index", 0))
        snapshot["delays"] = {}
        for name, actuator in asset.actuators.items():
            for field in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
                delay = getattr(actuator, field, None)
                if delay is None:
                    continue
                ring = delay._circular_buffer
                if ring._buffer is not None:
                    history = torch.roll(ring._buffer[:, env_id], -ring._pointer - 1, 0)
                    snapshot["delays"][(name, field)] = (
                        history.detach().cpu().clone(), int(ring._num_pushes[env_id]),
                        int(delay._time_lags[env_id]))
        return snapshot

    @torch.inference_mode()
    def _restore_stage(self, env_id, snapshot):
        asset = self.scene["robot"]
        ids = torch.tensor([env_id], device=self.device)
        root, pos, vel = (t.to(self.device)[None] for t in snapshot["physical"])
        asset.write_root_state_to_sim(root, env_ids=ids)
        asset.write_joint_state_to_sim(pos, vel, env_ids=ids)
        for name, obj in self._objects_for_snapshot():
            if name in snapshot:
                restore_batch(obj, snapshot[name], env_id)
        # Preserve chronological perception history despite the shared write head.
        if hasattr(self, "_v11_feature_ring"):
            shift = int(self._v11_ring_index) - snapshot["ring_index"]
            self._v11_feature_ring[env_id] = self._v11_feature_ring[env_id].roll(shift, 0)
        for (name, field), (history, pushes, lag) in snapshot["delays"].items():
            delay = getattr(asset.actuators[name], field)
            ring = delay._circular_buffer
            ring._buffer[:, env_id] = history.to(self.device).roll(ring._pointer + 1, 0)
            ring._num_pushes[env_id] = pushes
            delay.set_time_lag(lag, batch_ids=ids)
        # Restored events cannot pay again; contact solver state must revalidate.
        def clear_events(obj):
            if not isinstance(obj, dict):
                return
            for key, value in obj.items():
                if isinstance(value, dict):
                    clear_events(value)
                elif isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.num_envs:
                    if (key.endswith(("_event", "_delta", "last_episode_length")) or key in
                        ("valid", "seed", "ordered", "wrong_side", "duplicate", "skip", "dual_skip",
                         "locked_support_steps", "support_steps", "stable_frames", "last_episode_length", "episode")):
                        # `valid` in cached geometry is a mask, not a completion pulse.
                        if key != "valid" or obj is self._v135_riser_identity:
                            value[env_id] = 0
        for name, obj in self._objects_for_snapshot():
            if name == "env":
                clear_events(obj)
        self._v135_stage_started[env_id] = True
        self._v135_riser_identity["episode_ordered_valid"][env_id] = False
        self._v135_stage_restore_count += 1

    def _reset_idx(self, env_ids):
        if hasattr(self, "_v135_behavior_metrics"):
            # Reward/ledger state still describes the terminal transition here.
            # After super() it describes the new reset state and would lose it.
            self._v135_behavior_metrics.record()
            self._v135_behavior_metrics.finish(env_ids)
        super()._reset_idx(env_ids)
        if not hasattr(self, "_v135_behavior_metrics"):
            from .v135_behavior_metrics import V135BehaviorMetrics
            self._v135_behavior_metrics = V135BehaviorMetrics(self)
        if not hasattr(self, "_v135_stage_started"):
            self._v135_stage_started = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self._v135_stage_pool = {}
            self._v135_stage_restore_count = 0
            self._v135_stage_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._v135_stage_captured_target = torch.zeros_like(self._v135_stage_target)
            self._v135_stage_stable = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._v135_stage_started[env_ids] = False
        self._v135_stage_stable[env_ids] = 0
        self._v135_stage_target[env_ids] = 0
        self._v135_stage_captured_target[env_ids] = 0
        cfg = self.cfg.v135_stage_starts
        if not cfg.enabled:
            return
        for env_id in env_ids.tolist():
            snapshot = self._v135_stage_pool.get(env_id)
            if snapshot is None or torch.rand((), device=self.device) >= cfg.probability:
                continue
            same_tile = (snapshot["level"] == int(self.scene.terrain.terrain_levels[env_id])
                         and snapshot["type"] == int(self.scene.terrain.terrain_types[env_id])
                         and torch.equal(snapshot["tile"], self.scene.env_origins[env_id].cpu()))
            fresh = self.common_step_counter - snapshot["captured_step"] <= cfg.max_age_steps
            if same_tile and fresh:
                self._restore_stage(env_id, snapshot)

    def step(self, action):
        result = super().step(action)
        rewards = result[1]
        if not torch.isfinite(rewards).all():
            terms = self.reward_manager._step_reward
            bad_terms = (~torch.isfinite(terms)).sum(dim=0).nonzero().flatten().tolist()
            names = self.reward_manager.active_terms
            bad_envs = (~torch.isfinite(rewards)).nonzero().flatten()[:16].tolist()
            root_finite = torch.isfinite(self.scene["robot"].data.root_state_w).all(dim=1)
            raise RuntimeError(
                "V135_NONFINITE_REWARD "
                f"terms={[(names[i], int((~torch.isfinite(terms[:, i])).sum())) for i in bad_terms]} "
                f"envs={bad_envs} root_nonfinite={int((~root_finite).sum())} "
                f"action_nonfinite={int((~torch.isfinite(action)).sum())}"
            )
        self._v135_behavior_metrics.record()
        self._v135_behavior_metrics.publish()
        cfg = self.cfg.v135_stage_starts
        if not cfg.enabled:
            return result
        strict = self._v135_riser_identity
        body = self._v135_whole_body
        eligible = (strict["episode_ordered_valid"] & strict["seeded"]
                    & ((strict["target_id"] == 2) | (strict["target_id"] == 3))
                    & ~self.reset_buf & ~self._v135_stage_started
                    & (self.episode_length_buf > cfg.stable_frames)
                    & (self.scene["robot"].data.projected_gravity_b[:, 2] < -0.9))
        force = self.scene.sensors["contact_forces"].data.net_forces_w[:, body["contacts"], 2]
        eligible &= (force > self.cfg.v135_descent_support_transfer.contact_force_threshold).sum(1) >= 3
        eligible &= torch.isfinite(self.scene["robot"].data.root_state_w).all(1)
        changed = self._v135_stage_target != strict["target_id"]
        self._v135_stage_stable[changed] = 0
        self._v135_stage_target.copy_(strict["target_id"])
        self._v135_stage_stable = torch.where(eligible, self._v135_stage_stable + 1,
                                             torch.zeros_like(self._v135_stage_stable))
        candidates = ((self._v135_stage_stable >= cfg.stable_frames)
                      & (self._v135_stage_captured_target != strict["target_id"])).nonzero().flatten()
        for env_id in candidates[:cfg.max_captures_per_step].tolist():
            if len(self._v135_stage_pool) < cfg.max_cached_envs or env_id in self._v135_stage_pool:
                self._v135_stage_pool[env_id] = self._capture_stage(env_id)
                self._v135_stage_captured_target[env_id] = strict["target_id"][env_id]
        self.extras.setdefault("log", {})["Curriculum/v135_stage_cache_count"] = len(self._v135_stage_pool)
        self.extras["log"]["Curriculum/v135_stage_start_fraction"] = self._v135_stage_started.float().mean()
        return result
