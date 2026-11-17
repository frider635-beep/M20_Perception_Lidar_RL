"""RSL-RL runner for V9 Stage-1 without optional legacy AMP imports."""

from __future__ import annotations

import copy
import json
import os

import torch

from rsl_rl.runners import OnPolicyRunner

from rl_training.tasks.manager_based.locomotion.velocity import mdp


class V9Stage1Runner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        cfg = dict(train_cfg)
        checkpoint = cfg.pop("v9_base_checkpoint", None) or os.environ.get(
            "V9_BASE_CHECKPOINT",
            "/home/ubuntu/m20_perceptive_final/eval_outputs/v9_initialization_gate/colleague_model9999/model_9999.pt",
        )
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        self.v9_base_checkpoint = checkpoint
        self.v9_diagnostic_logging = os.environ.get("V9_DIAGNOSTIC_LOGGING", "1") != "0"
        self.v9_regression_snapshot = os.environ.get("V9_DIAGNOSTIC_REGRESSION_SNAPSHOT")
        self._load_initialization(checkpoint)
        self._freeze_actor()
        self._resolve_metric_ids()
        self._write_audit()

    def _load_initialization(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"V9 base checkpoint not found: {path}")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        actor_src, critic_src = loaded["actor_state_dict"], loaded["critic_state_dict"]
        policy = self.alg.policy
        if policy.actor[0].in_features != 372 or policy.actor[-1].out_features != 16:
            raise RuntimeError("V9 actor must be 372D -> 16D")
        if policy.critic[0].in_features != 247:
            raise RuntimeError("V9 critic must be 247D")
        with torch.no_grad():
            policy.actor[0].weight.zero_()
            policy.actor[0].weight[:, :57].copy_(actor_src["mlp.0.weight"].to(self.device))
            policy.actor[0].bias.copy_(actor_src["mlp.0.bias"].to(self.device))
            for index in (2, 4, 6):
                policy.actor[index].weight.copy_(actor_src[f"mlp.{index}.weight"].to(self.device))
                policy.actor[index].bias.copy_(actor_src[f"mlp.{index}.bias"].to(self.device))
            for index in (0, 2, 4, 6):
                policy.critic[index].weight.copy_(critic_src[f"mlp.{index}.weight"].to(self.device))
                policy.critic[index].bias.copy_(critic_src[f"mlp.{index}.bias"].to(self.device))
            if hasattr(policy, "log_std"):
                policy.log_std.copy_(actor_src["distribution.log_std_param"].to(self.device))
        self.v9_initial_iter = int(loaded.get("iter", 9999))
        self._v9_base_actor = copy.deepcopy(policy.actor).eval()
        for parameter in self._v9_base_actor.parameters():
            parameter.requires_grad_(False)

    def _freeze_actor(self):
        policy = self.alg.policy
        for parameter in policy.actor.parameters():
            parameter.requires_grad_(False)
        for parameter in policy.critic.parameters():
            parameter.requires_grad_(True)
        if hasattr(policy, "log_std"):
            policy.log_std.requires_grad_(False)
        weight = policy.actor[0].weight
        weight.requires_grad_(True)
        mask = torch.zeros_like(weight)
        mask[:, 57:] = 1.0
        weight.register_hook(lambda gradient: gradient * mask)

    def _resolve_metric_ids(self):
        raw, body_ids, contact_ids, wheel_joint_ids, leg_joint_ids = self.env.unwrapped, [], [], [], []
        robot, contact = raw.scene["robot"], raw.scene.sensors["contact_forces"]
        for name, joint_name in zip(
            mdp.WHEEL_NAMES, ("fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint")
        ):
            ids, names = robot.find_bodies(name)
            cids, cnames = contact.find_bodies(name)
            jids, jnames = robot.find_joints(joint_name, preserve_order=True)
            if (
                len(ids) != 1 or names != [name] or len(cids) != 1 or cnames != [name]
                or len(jids) != 1 or jnames != [joint_name]
            ):
                raise RuntimeError(f"V9 wheel order mismatch for {name}")
            body_ids.append(ids[0])
            contact_ids.append(cids[0])
            wheel_joint_ids.append(jids[0])
        for names in (
            ("fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint"),
            ("fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint"),
            ("hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint"),
            ("hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint"),
        ):
            ids, resolved = robot.find_joints(list(names), preserve_order=True)
            if len(ids) != 3 or resolved != list(names):
                raise RuntimeError(f"V9 leg joint order mismatch: expected {names}, got {resolved}")
            leg_joint_ids.append(ids)
        self._v9_wheel_body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        self._v9_contact_body_ids = torch.tensor(contact_ids, device=self.device, dtype=torch.long)
        self._v9_wheel_joint_ids = torch.tensor(wheel_joint_ids, device=self.device, dtype=torch.long)
        self._v9_leg_joint_ids = torch.tensor(leg_joint_ids, device=self.device, dtype=torch.long)
        self._v9_spin_streak = torch.zeros((raw.num_envs, 4), device=self.device, dtype=torch.int32)

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)

    @staticmethod
    def _masked_quantile(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
        selected = value[mask]
        if selected.numel() == 0:
            return torch.zeros((), device=value.device, dtype=value.dtype)
        return torch.quantile(selected.float(), q)

    def _metrics(self, obs):
        raw, policy = self.env.unwrapped, self.alg.policy
        policy_obs = policy.get_actor_obs(obs)
        with torch.inference_mode():
            action = policy.actor(policy_obs)
            base_action = self._v9_base_actor(policy_obs)
            delta = action - base_action
        state = mdp.perceptive_spatial_state(raw)
        robot = raw.scene["robot"]
        gravity = robot.data.projected_gravity_b
        command = raw.command_manager.get_command("base_velocity")[:, 0]
        forces = raw.scene.sensors["contact_forces"].data.net_forces_w[:, self._v9_contact_body_ids]
        force_norm = forces.norm(dim=-1)
        contact = force_norm >= 1.0
        wheel_actual_velocity = robot.data.joint_vel[:, self._v9_wheel_joint_ids]
        wheel_raw_v9 = action[:, 12:16]
        wheel_raw_base = base_action[:, 12:16]
        wheel_delta = wheel_raw_v9 - wheel_raw_base
        # The policy-mean target is useful for comparing V9 against BASE.  The
        # unqualified target is the exact latest target held by the real action
        # term after wrapper clipping, official scale and offset.
        wheel_target_policy_mean = wheel_raw_v9 * 5.0
        wheel_action_term = raw.action_manager.get_term("joint_vel")
        wheel_raw_executed = wheel_action_term.raw_actions
        wheel_target = wheel_action_term.processed_actions
        vertical = mdp.classify_vertical_riser_contact(
            state["wheel_yaw"][..., :2], state["z_actual_world"], forces,
            state["riser_x"], state["z_lower"], state["z_upper"],
            wheel_radius=raw.cfg.spatial_guidance.wheel_collision_radius,
        ) & state["valid"]
        old_precontact = state["valid"] & state["active"] & (state["riser_distance"] > 0) & ~vertical
        true_precontact = state["true_precontact_clearance"]
        leg_joint_pos = robot.data.joint_pos[:, self._v9_leg_joint_ids]
        leg_joint_default = robot.data.default_joint_pos[:, self._v9_leg_joint_ids]
        joint_delta = (leg_joint_pos - leg_joint_default).norm(dim=2)
        leg_delta = torch.stack(
            [delta[:, index * 3 : index * 3 + 3].norm(dim=1) for index in range(4)], dim=1
        )
        metrics = {
            "V9/action_delta/all": delta.norm(dim=1).mean(),
            "V9/action_delta/legs": delta[:, :12].norm(dim=1).mean(),
            "V9/action_delta/wheels": delta[:, 12:].norm(dim=1).mean(),
            "V9/elevation_weight_norm": policy.actor[0].weight[:, 57:].norm(),
            "V9/action_delta_norm": delta.norm(dim=1).mean(),
            "V9/wheel_action_max_abs": action[:, 12:].abs().max(),
            "first_riser/old_precontact_clearance_pass": (
                (old_precontact & (state["old_progress"] >= 0.999)).sum()
                / old_precontact.sum().clamp_min(1)
            ),
            # Preserve the historical tag for a clean before/after overlay;
            # its semantics remain explicitly OLD and it is not a success KPI.
            "first_riser/precontact_clearance_pass": (
                (old_precontact & (state["old_progress"] >= 0.999)).sum()
                / old_precontact.sum().clamp_min(1)
            ),
            "first_riser/true_precontact_clearance": true_precontact.float().sum()
            / state["valid"].sum().clamp_min(1),
            "first_riser/true_precontact_front": true_precontact[:, :2].any(dim=1).float().mean(),
            "first_riser/riser_plane_crossing": state["riser_plane_crossing_latched"].float().mean(),
            "first_riser/upper_tread_support": state["upper_tread_support_latched"].float().mean(),
            "first_riser/vertical_contact_latched": state["vertical_contact_latched"].float().mean(),
            "first_riser/absolute_clearance_progress": state["absolute_progress"].mean(),
            "first_riser/final_required_lift": state["final_required_lift"].mean(),
            "first_riser/vertical_contact": vertical.any(dim=1).float().mean(),
            "first_riser/crossing": (
                robot.data.root_pos_w[:, 0] - raw.scene.env_origins[:, 0] > 1.41
            ).float().mean(),
            "first_riser/riser_distance": torch.where(
                state["active"], state["riser_distance"], torch.zeros_like(state["riser_distance"])
            ).sum() / state["active"].sum().clamp_min(1),
            "body/base_vx": robot.data.root_lin_vel_b[:, 0].mean(),
            "body/track_lin_vel_abs_error": (robot.data.root_lin_vel_b[:, 0] - command).abs().mean(),
            "body/base_roll": torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6)).mean(),
            "body/base_pitch": torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6)).mean(),
            "body/fall": raw.termination_manager.terminated.float().mean(),
        }
        true_any = state["true_precontact_clearance_latched"].any(dim=1)
        crossing_any = state["riser_plane_crossing_latched"].any(dim=1)
        support_any = state["upper_tread_support_latched"].any(dim=1)
        first_riser_success = state["upper_tread_support_latched"].all(dim=1)
        metrics["first_riser/P_crossing_given_true_clearance"] = (
            (crossing_any & true_any).sum() / true_any.sum().clamp_min(1)
        )
        metrics["first_riser/P_tread_support_given_crossing"] = (
            (support_any & crossing_any).sum() / crossing_any.sum().clamp_min(1)
        )
        metrics["first_riser/success_given_true_clearance"] = (
            (first_riser_success & true_any).sum() / true_any.sum().clamp_min(1)
        )
        for index, wheel in enumerate(("FL", "FR", "HL", "HR")):
            metrics[f"V9/action_delta/{wheel}"] = delta[:, index * 3 : index * 3 + 3].norm(dim=1).mean()
            metrics[f"first_riser/{wheel}_actual_lift"] = state["actual_lift"][:, index].mean()
            metrics[f"first_riser/{wheel}_actual_geometric_lift"] = state["actual_geometric_lift"][:, index].mean()
            metrics[f"first_riser/{wheel}_absolute_progress"] = state["absolute_progress"][:, index].mean()
            metrics[f"first_riser/TRUE_PRECONTACT_CLEARANCE/{wheel}"] = (
                state["true_precontact_clearance_latched"][:, index].float().mean()
            )
            metrics[f"first_riser/RISER_PLANE_CROSSING/{wheel}"] = (
                state["riser_plane_crossing_latched"][:, index].float().mean()
            )
            metrics[f"first_riser/UPPER_TREAD_SUPPORT/{wheel}"] = (
                state["upper_tread_support_latched"][:, index].float().mean()
            )
            abs_wheel_delta = wheel_delta[:, index].abs()
            metrics[f"V9/wheel_delta/{wheel}"] = wheel_delta[:, index].mean()
            metrics[f"V9/wheel_delta/{wheel}_mean_abs"] = abs_wheel_delta.mean()
            metrics[f"V9/wheel_delta/{wheel}_p95_abs"] = torch.quantile(abs_wheel_delta, 0.95)
            metrics[f"V9/wheel_delta/{wheel}_max_abs"] = abs_wheel_delta.max()
            metrics[f"V9/wheel_raw_v9/{wheel}"] = wheel_raw_v9[:, index].mean()
            metrics[f"V9/wheel_raw_v9/{wheel}_max_abs"] = wheel_raw_v9[:, index].abs().max()
            metrics[f"V9/wheel_raw_base/{wheel}"] = wheel_raw_base[:, index].mean()
            metrics[f"V9/wheel_raw_executed/{wheel}"] = wheel_raw_executed[:, index].mean()
            metrics[f"V9/wheel_target/{wheel}"] = wheel_target[:, index].mean()
            metrics[f"V9/wheel_target/{wheel}_max_abs"] = wheel_target[:, index].abs().max()
            metrics[f"V9/wheel_target_policy_mean/{wheel}"] = wheel_target_policy_mean[:, index].mean()
            metrics[f"V9/wheel_velocity_actual/{wheel}"] = wheel_actual_velocity[:, index].mean()
            metrics[f"V9/wheel_velocity_actual/{wheel}_max_abs"] = wheel_actual_velocity[:, index].abs().max()
            metrics[f"V9/wheel_contact/{wheel}"] = contact[:, index].float().mean()

            mask = old_precontact[:, index]
            metrics[f"V9/precontact_sample_fraction/{wheel}"] = mask.float().mean()
            metrics[f"V9/precontact_lift/{wheel}"] = self._masked_mean(state["actual_lift"][:, index], mask)
            metrics[f"V9/precontact_leg_action_delta/{wheel}"] = self._masked_mean(leg_delta[:, index], mask)
            metrics[f"V9/precontact_joint_delta/{wheel}"] = self._masked_mean(joint_delta[:, index], mask)
            metrics[f"V9/precontact_lift_ref/{wheel}"] = self._masked_mean(state["lift_ref"][:, index], mask)
            metrics[f"V9/precontact_clearance_progress/{wheel}"] = self._masked_mean(
                state["progress"][:, index], mask
            )
            metrics[f"V9/precontact_base_vx/{wheel}"] = self._masked_mean(
                robot.data.root_lin_vel_b[:, 0], mask
            )

            active_leg_prelift = mask & (state["actual_lift"][:, index] > 0.005) & (
                leg_delta[:, index] > 0.05
            ) & (joint_delta[:, index] > 0.03)
            wheel_only = mask & (abs_wheel_delta > 1.0) & ~active_leg_prelift
            # Wheel joint signs are negative for normal +X travel on this M20.
            # Use target/actual agreement instead of assuming a positive sign.
            forward_propulsion = mask & (wheel_target[:, index].abs() > 1.0) & (
                wheel_target[:, index] * wheel_actual_velocity[:, index] > 0.0
            ) & (robot.data.root_lin_vel_b[:, 0] > 0.1)
            coordination = active_leg_prelift & forward_propulsion
            airborne_high_spin = (wheel_target[:, index].abs() > 50.0) & (
                wheel_actual_velocity[:, index].abs() > 40.0
            ) & ~contact[:, index]
            self._v9_spin_streak[:, index] = torch.where(
                airborne_high_spin,
                self._v9_spin_streak[:, index] + 1,
                torch.zeros_like(self._v9_spin_streak[:, index]),
            )
            abnormal_spin = self._v9_spin_streak[:, index] >= 2
            metrics[f"V9/diagnosis/ACTIVE_LEG_PRELIFT/{wheel}"] = self._masked_mean(
                active_leg_prelift.float(), mask
            )
            metrics[f"V9/diagnosis/FORWARD_WHEEL_PROPULSION/{wheel}"] = self._masked_mean(
                forward_propulsion.float(), mask
            )
            metrics[f"V9/diagnosis/LEG_WHEEL_COORDINATION/{wheel}"] = self._masked_mean(
                coordination.float(), mask
            )
            metrics[f"V9/diagnosis/WHEEL_ONLY_SHORTCUT/{wheel}"] = self._masked_mean(
                wheel_only.float(), mask
            )
            metrics[f"V9/diagnosis/AIRBORNE_HIGH_SPIN_INSTANT/{wheel}"] = airborne_high_spin.float().mean()
            metrics[f"V9/diagnosis/ABNORMAL_SINGLE_WHEEL_SPIN/{wheel}"] = abnormal_spin.float().mean()

            distance = state["riser_distance"][:, index]
            for label, lower, upper in (
                ("30_20cm", 0.20, 0.30), ("20_10cm", 0.10, 0.20),
                ("10_05cm", 0.05, 0.10), ("05_00cm", 0.00, 0.05),
            ):
                bin_mask = mask & (distance > lower) & (distance <= upper)
                prefix = f"V9/precontact_distance_{label}/{wheel}"
                metrics[f"{prefix}/sample_fraction"] = bin_mask.float().mean()
                metrics[f"{prefix}/actual_lift"] = self._masked_mean(state["actual_lift"][:, index], bin_mask)
                metrics[f"{prefix}/leg_action_delta"] = self._masked_mean(leg_delta[:, index], bin_mask)
                metrics[f"{prefix}/joint_position_delta"] = self._masked_mean(joint_delta[:, index], bin_mask)
                metrics[f"{prefix}/wheel_action_delta"] = self._masked_mean(wheel_delta[:, index], bin_mask)
                metrics[f"{prefix}/wheel_target"] = self._masked_mean(wheel_target[:, index], bin_mask)
                metrics[f"{prefix}/wheel_actual_velocity"] = self._masked_mean(
                    wheel_actual_velocity[:, index], bin_mask
                )
                metrics[f"{prefix}/contact"] = self._masked_mean(contact[:, index].float(), bin_mask)
                metrics[f"{prefix}/base_vx"] = self._masked_mean(robot.data.root_lin_vel_b[:, 0], bin_mask)
        return metrics

    def _write_regression_snapshot(self, obs, iteration: int):
        """Persist read-only trajectory state for logging ON/OFF regression."""
        if not self.v9_regression_snapshot:
            return
        raw, policy = self.env.unwrapped, self.alg.policy
        with torch.inference_mode():
            policy_obs = policy.get_actor_obs(obs)
            raw_action = policy.actor(policy_obs)
            wheel_term = raw.action_manager.get_term("joint_vel")
            leg_term = raw.action_manager.get_term("joint_pos")
            robot = raw.scene["robot"]
            snapshot = {
                "iteration": int(iteration),
                "raw_action": raw_action.detach().cpu(),
                "action_manager_raw": raw.action_manager.action.detach().cpu(),
                "processed_leg_action": leg_term.processed_actions.detach().cpu(),
                "processed_wheel_action": wheel_term.processed_actions.detach().cpu(),
                "root_pos": robot.data.root_pos_w.detach().cpu(),
                "root_quat": robot.data.root_quat_w.detach().cpu(),
                "root_lin_vel": robot.data.root_lin_vel_w.detach().cpu(),
                "root_ang_vel": robot.data.root_ang_vel_w.detach().cpu(),
                "joint_pos": robot.data.joint_pos.detach().cpu(),
                "joint_vel": robot.data.joint_vel.detach().cpu(),
            }
        path = os.path.abspath(self.v9_regression_snapshot)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(snapshot, path)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        self._write_regression_snapshot(locs["obs"], locs["it"])
        if not self.v9_diagnostic_logging:
            return
        for name, value in self._metrics(locs["obs"]).items():
            self.writer.add_scalar(name, float(value.detach().cpu()), locs["it"])

    def _write_audit(self):
        audit = {
            "base_checkpoint": os.path.abspath(self.v9_base_checkpoint),
            "base_iter": self.v9_initial_iter,
            "actor_shape": [372, 512, 256, 128, 16],
            "critic_shape": [247, 512, 256, 128, 1],
            "trainable_actor_scope": ["actor.0.weight[:,57:372]"],
            "frozen_official_actor": True,
            "critic_trainable": True,
        }
        print("V9_TRAINABLE_ACTOR_PARAMS=actor.0.weight[:,57:372]", flush=True)
        print("V9_FROZEN_OFFICIAL_PARAMS=TRUE", flush=True)
        print("V9_CRITIC_TRAINABLE=TRUE", flush=True)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "v9_parameter_audit.json"), "w") as handle:
                json.dump(audit, handle, indent=2)


__all__ = ["V9Stage1Runner"]
