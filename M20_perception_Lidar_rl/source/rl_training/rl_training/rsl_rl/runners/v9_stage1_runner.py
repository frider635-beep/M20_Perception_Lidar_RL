"""RSL-RL runner for V9 Stage-1 transfer learning.

The inherited 57-D locomotion actor is loaded from the colleague checkpoint.
Only the newly appended elevation columns of the first actor layer receive
gradients; the critic is initialized from the same checkpoint and remains
trainable.  This runner is selected only by the V9 Stage-1 task.
"""

from __future__ import annotations

import json
import os

import torch

from rsl_rl.runners import OnPolicyRunner

from rl_training.tasks.manager_based.locomotion.velocity import mdp


class V9Stage1Runner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        cfg = dict(train_cfg)
        checkpoint = cfg.pop("v9_base_checkpoint", None)
        checkpoint = checkpoint or os.environ.get(
            "V9_BASE_CHECKPOINT",
            "/home/ubuntu/m20_perceptive_final/eval_outputs/v9_initialization_gate/colleague_model9999/model_9999.pt",
        )
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        self.v9_base_checkpoint = checkpoint
        self._load_v9_initialization(checkpoint)
        self._configure_v9_freeze()
        self._resolve_metric_ids()
        self._write_v9_audit()

    @staticmethod
    def _copy_linear(dst, src, *, elevation=False):
        with torch.no_grad():
            if elevation:
                dst.weight.zero_()
                dst.weight[:, :57].copy_(src["mlp.0.weight"].to(dst.weight.device))
            else:
                dst.weight.copy_(src["mlp.0.weight"].to(dst.weight.device))
            dst.bias.copy_(src["mlp.0.bias"].to(dst.bias.device))

    def _load_v9_initialization(self, checkpoint_path: str):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"V9 base checkpoint not found: {checkpoint_path}")
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor_src = loaded.get("actor_state_dict", {})
        critic_src = loaded.get("critic_state_dict", {})
        policy = self.alg.policy
        if policy.actor[0].in_features != 372 or policy.actor[-1].out_features != 16:
            raise RuntimeError("V9 actor shape mismatch; expected 372D -> 16D")
        if policy.critic[0].in_features != 247:
            raise RuntimeError("V9 critic shape mismatch; expected 247D critic")
        with torch.no_grad():
            policy.actor[0].weight.zero_()
            policy.actor[0].weight[:, :57].copy_(actor_src["mlp.0.weight"].to(policy.actor[0].weight.device))
            policy.actor[0].bias.copy_(actor_src["mlp.0.bias"].to(policy.actor[0].bias.device))
            for idx in (2, 4, 6):
                policy.actor[idx].weight.copy_(actor_src[f"mlp.{idx}.weight"].to(policy.actor[idx].weight.device))
                policy.actor[idx].bias.copy_(actor_src[f"mlp.{idx}.bias"].to(policy.actor[idx].bias.device))
            for idx in (0, 2, 4, 6):
                policy.critic[idx].weight.copy_(critic_src[f"mlp.{idx}.weight"].to(policy.critic[idx].weight.device))
                policy.critic[idx].bias.copy_(critic_src[f"mlp.{idx}.bias"].to(policy.critic[idx].bias.device))
            if hasattr(policy, "log_std") and "distribution.log_std_param" in actor_src:
                policy.log_std.copy_(actor_src["distribution.log_std_param"].to(policy.log_std.device))
        self.v9_initial_iter = int(loaded.get("iter", 9999))

    def _configure_v9_freeze(self):
        policy = self.alg.policy
        for parameter in policy.actor.parameters():
            parameter.requires_grad_(False)
        for parameter in policy.critic.parameters():
            parameter.requires_grad_(True)
        for name in ("log_std", "std", "distribution", "log_std_param"):
            value = getattr(policy, name, None)
            if isinstance(value, torch.nn.Parameter):
                value.requires_grad_(False)
        elevation_weight = policy.actor[0].weight
        elevation_weight.requires_grad_(True)
        mask = torch.zeros_like(elevation_weight)
        mask[:, 57:] = 1.0
        elevation_weight.register_hook(lambda grad: grad * mask)
        self.v9_trainable_actor = ["actor.0.weight[:,57:372]"]
        self.v9_frozen_actor = [
            "actor.0.weight[:,:57]", "actor.0.bias", "actor.2.*", "actor.4.*", "actor.6.*", "log_std"
        ]

    def _resolve_metric_ids(self):
        raw = self.env.unwrapped
        robot = raw.scene["robot"]
        ids = []
        for name in mdp.WHEEL_NAMES:
            found, names = robot.find_bodies(name)
            if len(found) != 1 or names != [name]:
                raise RuntimeError(f"V9 wheel order mismatch for {name}: {names}")
            ids.append(found[0])
        self._v9_wheel_body_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        contact = raw.scene.sensors["contact_forces"]
        contact_ids = []
        for name in mdp.WHEEL_NAMES:
            found, names = contact.find_bodies(name)
            if len(found) != 1 or names != [name]:
                raise RuntimeError(f"V9 contact wheel order mismatch for {name}: {names}")
            contact_ids.append(found[0])
        self._v9_contact_body_ids = torch.tensor(contact_ids, device=self.device, dtype=torch.long)
        self._v9_base_actor = self._make_base_actor()

    def _make_base_actor(self):
        import copy
        actor = copy.deepcopy(self.alg.policy.actor).to(self.device).eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        return actor

    def _v9_metrics(self, obs):
        raw = self.env.unwrapped
        policy_obs = self.alg.policy.get_actor_obs(obs)
        with torch.inference_mode():
            action = self.alg.policy.actor(policy_obs)
            base_action = self._v9_base_actor(policy_obs)
            delta = action - base_action
        state = mdp.perceptive_spatial_state(raw)
        robot = raw.scene["robot"]
        gravity = robot.data.projected_gravity_b
        command = raw.command_manager.get_command("base_velocity")[:, 0]
        wheel_forces = raw.scene.sensors["contact_forces"].data.net_forces_w[:, self._v9_contact_body_ids]
        vertical_contact = mdp.classify_vertical_riser_contact(
            state["wheel_yaw"][..., :2], state["z_actual_world"], wheel_forces,
            state["riser_x"], state["z_lower"], state["z_upper"],
            wheel_radius=raw.cfg.spatial_guidance.wheel_collision_radius,
        ) & state["valid"]
        metrics = {
            "V9/action_delta/all": delta.norm(dim=1).mean(),
            "V9/action_delta/legs": delta[:, :12].norm(dim=1).mean(),
            "V9/action_delta/wheels": delta[:, 12:].norm(dim=1).mean(),
            "V9/elevation_weight_norm": self.alg.policy.actor[0].weight[:, 57:].norm(),
            "V9/action_delta_norm": delta.norm(dim=1).mean(),
            "V9/wheel_action_max_abs": action[:, 12:].abs().max(),
            "first_riser/riser_valid": state["valid"].float().mean(),
            "first_riser/guide_active": state["active"].float().mean(),
            "first_riser/riser_distance": torch.where(
                state["active"], state["riser_distance"], torch.zeros_like(state["riser_distance"])
            ).sum() / state["active"].sum().clamp_min(1),
            "first_riser/precontact_clearance_pass": (
                (state["active"] & (state["riser_distance"] >= 0) & (state["progress"] >= 0.999)).sum()
                / (state["active"] & (state["riser_distance"] >= 0)).sum().clamp_min(1)
            ),
            "first_riser/crossing": (
                robot.data.root_pos_w[:, 0] - raw.scene.env_origins[:, 0] > 1.10 + 0.31
            ).float().mean(),
            "first_riser/vertical_contact": vertical_contact.any(dim=1).float().mean(),
            "body/base_vx": robot.data.root_lin_vel_b[:, 0].mean(),
            "body/track_lin_vel_abs_error": (robot.data.root_lin_vel_b[:, 0] - command).abs().mean(),
            "body/base_roll": torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6)).mean(),
            "body/base_pitch": torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6)).mean(),
            "body/fall": raw.termination_manager.terminated.float().mean(),
        }
        for wi, wheel in enumerate(("FL", "FR", "HL", "HR")):
            metrics[f"V9/action_delta/{wheel}"] = delta[:, wi * 3 : wi * 3 + 3].norm(dim=1).mean()
            metrics[f"first_riser/{wheel}_actual_lift"] = state["actual_lift"][:, wi].mean()
        return metrics

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        for name, value in self._v9_metrics(locs["obs"]).items():
            self.writer.add_scalar(name, float(value.detach().cpu()), locs["it"])

    def _write_v9_audit(self):
        audit = {
            "base_checkpoint": os.path.abspath(self.v9_base_checkpoint),
            "base_iter": self.v9_initial_iter,
            "actor_shape": [372, 512, 256, 128, 16],
            "critic_shape": [247, 512, 256, 128, 1],
            "trainable_actor_scope": self.v9_trainable_actor,
            "frozen_actor_scope": self.v9_frozen_actor,
            "critic_trainable": True,
            "official_reward_unchanged": True,
            "metrics_only": True,
        }
        print("V9_TRAINABLE_ACTOR_PARAMS=actor.0.weight[:,57:372]", flush=True)
        print("V9_FROZEN_OFFICIAL_ACTOR_PARAMS=" + ",".join(self.v9_frozen_actor), flush=True)
        print("V9_CRITIC_TRAINABLE=TRUE", flush=True)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "v9_parameter_audit.json"), "w") as handle:
                json.dump(audit, handle, indent=2)
