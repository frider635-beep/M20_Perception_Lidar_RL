"""RSL-RL runner and TensorBoard metrics for M20 V10."""

from __future__ import annotations

import os

import torch
import rsl_rl.runners.on_policy_runner as on_policy_runner_module
from rsl_rl.runners import OnPolicyRunner

from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.v10_unified_policy import V10UnifiedActorCritic


class V10UnifiedRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        # OnPolicyRunner resolves the configured class with eval() in its module.
        on_policy_runner_module.V10UnifiedActorCritic = V10UnifiedActorCritic
        super().__init__(env, dict(train_cfg), log_dir=log_dir, device=device)
        policy = self.alg.policy
        if (policy.actor_obs_dim, policy.action_dim) != (60, 16):
            raise RuntimeError("V10 runner dimension gate failed")
        self._last_fr_action = torch.zeros((env.num_envs, 3), device=device)
        self._v10_interval_prev = {}
        self._write_manifests()
        print(f"ACTOR_OBS_DIM={policy.actor_obs_dim}")
        print(f"CRITIC_OBS_DIM={policy.critic_obs_dim}")
        print(f"ACTION_DIM={policy.action_dim}")

    def _write_manifests(self):
        if self.log_dir is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        actor = (
            "M20 V10 actor observation (60D)\n"
            "0:3 base_ang_vel; 3:6 projected_gravity; 6:9 command; "
            "9:25 joint_pos; 25:41 joint_vel; 41:57 previous_action; "
            "57 b_actor; 58 H_actor; 59 D_actor\n"
            "Simulator-only values and the 315D PGTT map are forbidden.\n"
        )
        critic = (
            "M20 V10 critic observation (official 247D + PGTT 315D + geometry 3D + teacher 4D)\n"
            "Privileged additions: clean PGTT, b/H/D truth, FR Z/contact/vertical-contact/progress.\n"
        )
        for name, content in (("actor_observation_manifest.txt", actor), ("critic_observation_manifest.txt", critic)):
            with open(os.path.join(self.log_dir, name), "w", encoding="ascii") as stream:
                stream.write(content)

    @staticmethod
    def _masked_mean(value, mask):
        return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)

    def _v10_metrics(self, obs):
        raw = self.env.unwrapped
        state = mdp.v10_state(raw)
        robot = raw.scene["robot"]
        gravity = robot.data.projected_gravity_b
        actor_obs = self.alg.policy.get_actor_obs(obs)
        with torch.inference_mode():
            action = self.alg.policy.actor(self.alg.policy.actor_obs_normalizer(actor_obs))
        fr_action = action[:, 3:6]
        fr_rate = (fr_action - self._last_fr_action).abs().mean(dim=1)
        self._last_fr_action.copy_(fr_action)
        stair = state["valid"]
        b = state["b_raw"]
        totals = raw._v10_metrics_total
        def total(name):
            return totals[name].sum().float()
        def ratio(numerator, denominator):
            return numerator / denominator.clamp_min(1.0)
        def interval_total(name):
            current = total(name)
            previous = self._v10_interval_prev.get(name, torch.zeros_like(current))
            self._v10_interval_prev[name] = current.clone()
            return current - previous
        terrain = raw.scene.terrain
        topology = state["topology"]
        blind_shortcut_count = getattr(
            raw, "_v10_blind_shortcut_count", torch.zeros((), device=self.device)
        ).float()
        stair_official_up_count = getattr(
            raw, "_v10_stair_official_up_count", torch.zeros((), device=self.device)
        ).float()
        stair_curriculum_count = getattr(
            raw, "_v10_stair_curriculum_count", torch.zeros((), device=self.device)
        ).float()
        stair_promotion_count = getattr(
            raw, "_v10_stair_promotion_count", torch.zeros((), device=self.device)
        ).float()
        stair_gated_official_up_count = getattr(
            raw, "_v10_stair_gated_official_up_count", torch.zeros((), device=self.device)
        ).float()
        stair_would_fail_first_count = getattr(
            raw, "_v10_stair_would_fail_first_count", torch.zeros((), device=self.device)
        ).float()
        gated_outcome_counts = getattr(
            raw,
            "_v10_gated_outcome_counts",
            torch.zeros(5, device=self.device),
        ).float()
        gated_unclassified_count = getattr(
            raw, "_v10_gated_unclassified_count", torch.zeros((), device=self.device)
        ).float()
        velocity_cfg = raw.reward_manager.get_term_cfg("track_lin_vel_xy_exp")
        velocity_reward = velocity_cfg.func(raw, **velocity_cfg.params) * velocity_cfg.weight * raw.step_dt
        clearance_cfg = raw.reward_manager.get_term_cfg("fr_clearance_progress")
        clearance_reward = (
            clearance_cfg.func(raw, **clearance_cfg.params) * clearance_cfg.weight * raw.step_dt
        )
        post_steps = interval_total("post_first_full_steps")
        post_motion_steps = interval_total("post_first_motion_steps")
        post_reverse_steps = interval_total("post_first_reverse_steps")
        post_strong_reverse_steps = interval_total("post_first_strong_reverse_steps")
        post_v_along_sum = interval_total("post_first_v_along_command_sum")
        post_command_speed_sum = interval_total("post_first_command_speed_sum")
        post_velocity_error_sum = interval_total("post_first_velocity_error_sum")
        post_xy_reward_sum = interval_total("post_first_xy_reward_sum")
        post_first_events = interval_total("post_first_event_count")
        active_clearance_encounter = (
            (topology == 3) & state["target_valid"] & state["first_riser_locked"]
        )
        metrics = {
            "V10_Perception/StairSampleFraction": stair.float().mean(),
            "V10_Perception/BActiveFraction": b.float().mean(),
            "V10_Perception/BRaw": b.float().mean(),
            "V10_Perception/BActor": state["actor_feature"][:, 0].mean(),
            "V10_Perception/HMean": self._masked_mean(state["h_raw"], stair),
            "V10_Perception/DMean": self._masked_mean(state["d_raw"], b),
            "V10_Perception/HRaw": state["h_raw"].mean(),
            "V10_Perception/HActor": state["actor_feature"][:, 1].mean(),
            "V10_Perception/DRaw": state["d_raw"].mean(),
            "V10_Perception/DActor": state["actor_feature"][:, 2].mean(),
            "V10_Perception/FirstRiserLockedFraction": state["first_riser_locked"].float().mean(),
            "V10_Perception/FirstRiserConsumedFraction": state["first_riser_consumed"].float().mean(),
            "V10_Perception/PrecontactVerticalContactFraction": state["vertical_contact"].float().mean(),
            "Reward/fr_clearance_progress": clearance_reward.mean(),
            "V10_Perception/MaxClearanceProgress": self._masked_mean(
                state["max_clearance_progress"], active_clearance_encounter
            ),
            "V10_EventDensity/FirstStep": state["first_step_event"].float().mean(),
            "V10_EventDensity/Miss": state["perception_miss_event"].float().mean(),
            "V10_EventDensity/SingleStepComplete": state["single_step_complete_event"].float().mean(),
            "V10_EventDensity/FullStairSuccess": state["full_stair_event"].float().mean(),
            "V10_EventDensity/Joint": state["joint_event"].float().mean(),
            "V10_EventDensity/BlindOnly": state["blind_only_event"].float().mean(),
            "V10_Diagnostics/S1/BaseVx": self._masked_mean(state["base_vx"], b),
            "V10_Diagnostics/S1/CommandVx": self._masked_mean(raw.command_manager.get_command("base_velocity")[:, 0], b),
            "V10_Diagnostics/S1/VelocityTrackingReward": self._masked_mean(velocity_reward, b),
            "V10_Diagnostics/S1/Roll": self._masked_mean(torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6)), b),
            "V10_Diagnostics/S1/Pitch": self._masked_mean(torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6)), b),
            "V10_Diagnostics/S1/FRActionAbs": self._masked_mean(fr_action.abs().mean(dim=1), b),
            "V10_Diagnostics/S1/FRActionRate": self._masked_mean(fr_rate, b),
            "V10_Diagnostics/PGTT/DelaySteps": torch.tensor(float(raw.cfg.v10_perception.perception_delay_steps), device=self.device),
            "V10_Diagnostics/PGTT/HRawVsActorError": (state["h_raw"] - state["actor_feature"][:, 1]).abs().mean(),
            "V10_Diagnostics/PGTT/DRawVsActorError": (state["d_raw"] - state["actor_feature"][:, 2]).abs().mean(),
            "V10_Curriculum/HeightLevelMean": terrain.terrain_levels.float().mean(),
            "V10_Curriculum/FlatFraction": (topology == 0).float().mean(),
            "V10_Curriculum/RoughFraction": (topology == 1).float().mean(),
            "V10_Curriculum/SingleStepFraction": (topology == 2).float().mean(),
            "V10_Curriculum/FullStairFraction": (topology == 3).float().mean(),
            "V10_Curriculum/BlindShortcutRate": ratio(
                blind_shortcut_count, stair_official_up_count
            ),
            "V10_Curriculum/NBlindShortcut": blind_shortcut_count,
            "V10_Curriculum/NOfficialUpStair": stair_official_up_count,
            "Curriculum/StairTerrainLevelMean": self._masked_mean(
                terrain.terrain_levels.float(), topology == 3
            ),
            "Curriculum/NonStairTerrainLevelMean": self._masked_mean(
                terrain.terrain_levels.float(), topology != 3
            ),
            "Curriculum/StairOfficialUpRate": ratio(
                stair_official_up_count, stair_curriculum_count
            ),
            "Curriculum/StairPromotionRate": ratio(
                stair_promotion_count, stair_curriculum_count
            ),
            "Curriculum/StairPromotionBlockedByFirstRate": ratio(
                torch.zeros_like(stair_gated_official_up_count), stair_gated_official_up_count
            ),
            "V10_Curriculum/WouldFailFirstAtOfficialUpRate": ratio(
                stair_would_fail_first_count, stair_gated_official_up_count
            ),
            "V10_Core/PostFirst/ActiveSamplesPerRollout": post_steps,
            "V10_Core/PostFirst/CommandedMotionSamplesPerRollout": post_motion_steps,
            "V10_Core/PostFirst/FirstEventsPerRollout": post_first_events,
            "V10_Core/PostFirst/ReverseAnyRate": ratio(post_reverse_steps, post_motion_steps),
            "V10_Core/PostFirst/ReverseBelowMinus0p2Rate": ratio(
                post_strong_reverse_steps, post_motion_steps
            ),
            "V10_Core/PostFirst/BaseVelocityAlongCommand": ratio(
                post_v_along_sum, post_motion_steps
            ),
            "V10_Core/PostFirst/CommandSpeed": ratio(
                post_command_speed_sum, post_motion_steps
            ),
            "V10_Core/PostFirst/VelocityErrorNorm": ratio(
                post_velocity_error_sum, post_motion_steps
            ),
            "V10_Core/PostFirst/ExtraXYRewardPerActiveStep": ratio(
                post_xy_reward_sum, post_steps
            ),
        }
        gated_labels = (
            "VerticalContactBeforeClearance",
            "ClearanceNotReached",
            "ClearanceReachedButNoCrossing",
            "CrossingReachedButNoUpperSupport",
            "FirstSuccess",
        )
        lifetime_prefix = "V10_Curriculum/GatedOpportunity/Lifetime"
        metrics[f"{lifetime_prefix}/N"] = stair_gated_official_up_count
        for index, label in enumerate(gated_labels):
            metrics[f"{lifetime_prefix}/N{label}"] = gated_outcome_counts[index]
            metrics[f"{lifetime_prefix}/{label}Rate"] = ratio(
                gated_outcome_counts[index], stair_gated_official_up_count
            )
        metrics[f"{lifetime_prefix}/NUnclassified"] = gated_unclassified_count
        metrics[f"{lifetime_prefix}/UnclassifiedRate"] = ratio(
            gated_unclassified_count, stair_gated_official_up_count
        )
        gated_recent = getattr(raw, "_v10_gated_recent", None)
        gated_recent_next = getattr(raw, "_v10_gated_recent_next", 0)
        if gated_recent is not None:
            metrics["V10_Perception/Recent100ClearanceProgressMean"] = (
                mdp.v10_recent_gated_clearance_mean(gated_recent, gated_recent_next, 100)
            )
            metrics["V10_Perception/Recent500ClearanceProgressMean"] = (
                mdp.v10_recent_gated_clearance_mean(gated_recent, gated_recent_next, 500)
            )
            for window in (100, 500):
                _, denominator = mdp.v10_recent_gated_outcome_counts(
                    gated_recent, gated_recent_next, window
                )
                prefix = f"V10_Curriculum/GatedOpportunity/Recent{window}"
                metrics[f"{prefix}/N"] = denominator.float()
                for index, label in enumerate(gated_labels):
                    numerator, field_denominator = mdp.v10_recent_gated_outcome_counts(
                        gated_recent, gated_recent_next, window, outcome=index
                    )
                    metrics[f"{prefix}/N{label}"] = numerator.float()
                    metrics[f"{prefix}/{label}Rate"] = ratio(
                        numerator.float(), field_denominator.float()
                    )
                unclassified, field_denominator = mdp.v10_recent_gated_outcome_counts(
                    gated_recent, gated_recent_next, window, outcome=-1
                )
                metrics[f"{prefix}/NUnclassified"] = unclassified.float()
                metrics[f"{prefix}/UnclassifiedRate"] = ratio(
                    unclassified.float(), field_denominator.float()
                )
        terrain_steps = totals["terrain_steps"].sum(dim=0).float()
        terrain_b_steps = totals["terrain_b_steps"].sum(dim=0).float()
        for index, label in enumerate(("Flat", "Rough", "Single", "Full")):
            metrics[f"V10_Perception/{label}/BActiveFraction"] = ratio(
                terrain_b_steps[index], terrain_steps[index]
            )
        false_events = totals["false_trigger_events"].sum(dim=0).float()
        metrics["V10_Perception/Flat/FalseTriggerRate"] = ratio(false_events[0], terrain_steps[0])
        metrics["V10_Perception/Rough/FalseTriggerRate"] = ratio(false_events[1], terrain_steps[1])

        stair_encounters = totals["stair_encounters"].sum(dim=0).float()
        event_totals = {
            name: totals[name].sum(dim=0).float()
            for name in (
                "first_success", "miss", "single_complete", "full_success",
                "joint_success", "blind_only_success",
            )
        }
        single_n, full_n = stair_encounters[0], stair_encounters[1]
        all_n = single_n + full_n

        def add_lifetime(scope, denominator, fields):
            for prefix in (f"V10_Task/{scope}", f"V10_Task/Lifetime/{scope}"):
                metrics[f"{prefix}/NEncounter"] = denominator
                for output, numerator in fields.items():
                    metrics[f"{prefix}/N{output.removesuffix('Rate')}"] = numerator
                    metrics[f"{prefix}/{output}"] = ratio(numerator, denominator)

        add_lifetime("AllStair", all_n, {
            "FirstStepSuccessRate": event_totals["first_success"].sum(),
            "PerceptionMissRate": event_totals["miss"].sum(),
        })
        add_lifetime("Single", single_n, {
            "FirstStepSuccessRate": event_totals["first_success"][0],
            "PerceptionMissRate": event_totals["miss"][0],
            "SingleStepCompleteRate": event_totals["single_complete"][0],
        })
        add_lifetime("Full", full_n, {
            "FirstStepSuccessRate": event_totals["first_success"][1],
            "PerceptionMissRate": event_totals["miss"][1],
            "FullStairSuccessRate": event_totals["full_success"][1],
            "JointSuccessRate": event_totals["joint_success"][1],
            "BlindOnlySuccessRate": event_totals["blind_only_success"][1],
        })

        metrics.update({
            "V10_Perception/D_first_valid": ratio(
                total("d_first_valid_sum"), total("d_first_valid_count")
            ),
            "V10_Perception/D_first_lock": ratio(
                total("d_first_lock_sum"), total("d_first_lock_count")
            ),
            "V10_Perception/D_b_on": ratio(
                total("d_first_b_on_sum"), total("d_first_b_on_count")
            ),
            "V10_Perception/BDurationPerCompletedEncounterSeconds": ratio(
                total("b_duration_steps") * raw.step_dt, total("b_duration_encounters")
            ),
            "V10_Perception/DAtBOn": ratio(total("d_at_b_on_sum"), total("d_at_b_on_count")),
            "V10_Perception/DAtBRawOff": ratio(total("d_at_b_raw_off_sum"), total("d_at_b_raw_off_count")),
            "V10_Perception/DAtBActorOff": ratio(total("d_at_b_actor_off_sum"), total("d_at_b_actor_off_count")),
            "V10_Perception/TeacherSuccessWhileBActiveFraction": ratio(
                total("teacher_success_while_b"), total("first_success")
            ),
            "V10_Perception/TeacherMissWhileBActiveFraction": ratio(
                total("teacher_miss_while_b"), total("miss")
            ),
        })
        blocker_steps = totals["pre_b_on_blocker_steps"].sum(dim=0).float()
        blocker_denominator = blocker_steps.sum()
        for index, label in enumerate((
            "no_valid_riser", "not_locked", "D_out_of_range", "command_not_forward"
        )):
            metrics[f"V10_Perception/PreBOnBlocker/{label}"] = ratio(
                blocker_steps[index], blocker_denominator
            )

        bucket_labels = ("05_08", "08_11", "11_14", "14_17", "17_20", "20_23")
        bucket_encounters = totals["bucket_encounters"].sum(dim=0).float()
        bucket_totals = {
            name: totals[name].sum(dim=0).float()
            for name in (
                "bucket_first_success", "bucket_miss", "bucket_single_complete",
                "bucket_full_success", "bucket_joint_success", "bucket_blind_only_success",
            )
        }
        for topology_index, topology_label in enumerate(("Single", "Full")):
            for bucket_index, bucket_label in enumerate(bucket_labels):
                denominator = bucket_encounters[topology_index, bucket_index]
                prefix = f"V10_Task/Lifetime/{topology_label}/{bucket_label}"
                metrics[f"{prefix}/NEncounter"] = denominator
                metrics[f"{prefix}/FirstStepSuccessRate"] = ratio(
                    bucket_totals["bucket_first_success"][topology_index, bucket_index], denominator
                )
                metrics[f"{prefix}/PerceptionMissRate"] = ratio(
                    bucket_totals["bucket_miss"][topology_index, bucket_index], denominator
                )
                if topology_label == "Single":
                    metrics[f"{prefix}/SingleStepCompleteRate"] = ratio(
                        bucket_totals["bucket_single_complete"][topology_index, bucket_index], denominator
                    )
                else:
                    for key, output in (
                        ("bucket_full_success", "FullStairSuccessRate"),
                        ("bucket_joint_success", "JointSuccessRate"),
                        ("bucket_blind_only_success", "BlindOnlySuccessRate"),
                    ):
                        metrics[f"{prefix}/{output}"] = ratio(
                            bucket_totals[key][topology_index, bucket_index], denominator
                        )

        def recent(key, window, field=None, bucket=None):
            return mdp.v10_recent_window_counts(
                raw._v10_recent_metrics[key], raw._v10_recent_next[key], window, field=field, bucket=bucket
            )

        def recent_mean(key, window, field, bucket=None):
            return mdp.v10_recent_window_mean(
                raw._v10_recent_metrics[key], raw._v10_recent_next[key], window,
                field=field, bucket=bucket,
            )

        for window in (100, 500):
            for key, label, fields in (
                ("all", "AllStair", ("first_success", "miss")),
                ("single", "Single", ("first_success", "miss", "single_complete")),
                ("full", "Full", ("first_success", "miss", "full_success", "joint_success", "blind_only_success")),
            ):
                _, denominator = recent(key, window)
                prefix = f"V10_Task/Recent{window}/{label}"
                metrics[f"{prefix}/NEncounter"] = denominator.float()
                output_names = {
                    "first_success": "FirstStepSuccessRate",
                    "miss": "PerceptionMissRate",
                    "single_complete": "SingleStepCompleteRate",
                    "full_success": "FullStairSuccessRate",
                    "joint_success": "JointSuccessRate",
                    "blind_only_success": "BlindOnlySuccessRate",
                }
                for field in fields:
                    numerator, field_denominator = recent(key, window, field=field)
                    metrics[f"{prefix}/{output_names[field]}"] = ratio(
                        numerator.float(), field_denominator.float()
                    )
                if key == "all":
                    continue
                for bucket_index, bucket_label in enumerate(bucket_labels):
                    _, bucket_denominator = recent(key, window, bucket=bucket_index)
                    bucket_prefix = f"{prefix}/{bucket_label}"
                    metrics[f"{bucket_prefix}/NEncounter"] = bucket_denominator.float()
                    for field in fields:
                        numerator, field_denominator = recent(
                            key, window, field=field, bucket=bucket_index
                        )
                        metrics[f"{bucket_prefix}/{output_names[field]}"] = ratio(
                            numerator.float(), field_denominator.float()
                        )

        # Encounter-conditioned joint diagnostics. These are simulation-only
        # metrics and never enter observations, rewards, actions, or curriculum.
        for bucket_index, bucket_label in enumerate(bucket_labels):
            _, denominator = recent("all", 500, bucket=bucket_index)
            prefix = f"V10_HeightJoint/Recent500/{bucket_label}"
            metrics[f"{prefix}/NEncounter"] = denominator.float()
            metrics[f"{prefix}/HMeanCm"] = 100.0 * recent_mean(
                "all", 500, "height", bucket=bucket_index
            )
            metrics[f"{prefix}/FRActionAmplitudeMean"] = recent_mean(
                "all", 500, "fr_action_amplitude_mean", bucket=bucket_index
            )
            metrics[f"{prefix}/FRActionAmplitudeMax"] = recent_mean(
                "all", 500, "fr_action_amplitude_max", bucket=bucket_index
            )
            for field, output in (
                ("fr_hipx_action", "FRHipXAction"),
                ("fr_hipy_action", "FRHipYAction"),
                ("fr_knee_action", "FRKneeAction"),
                ("abs_fr_hipx_action", "AbsFRHipXAction"),
                ("abs_fr_hipy_action", "AbsFRHipYAction"),
                ("abs_fr_knee_action", "AbsFRKneeAction"),
            ):
                metrics[f"{prefix}/{output}"] = recent_mean(
                    "all", 500, field, bucket=bucket_index
                )
            metrics[f"{prefix}/MaxActiveFRLift"] = recent_mean(
                "all", 500, "max_active_fr_lift", bucket=bucket_index
            )
            metrics[f"{prefix}/ActiveLiftRatio"] = recent_mean(
                "all", 500, "active_lift_ratio", bucket=bucket_index
            )
            metrics[f"{prefix}/FRAbsoluteMaxLiftCm"] = 100.0 * recent_mean(
                "all", 500, "absolute_fr_max_lift", bucket=bucket_index
            )
            metrics[f"{prefix}/ClearanceProgressMean"] = recent_mean(
                "all", 500, "max_clearance_progress", bucket=bucket_index
            )
            first, first_denominator = recent(
                "all", 500, field="first_success", bucket=bucket_index
            )
            metrics[f"{prefix}/FirstStepSuccessRate"] = ratio(
                first.float(), first_denominator.float()
            )
        # Keep TensorBoard focused on the current failure mode. The old output
        # duplicated lifetime/recent windows, emitted nonexistent Single-task
        # panels, and expanded FR joint diagnostics into dozens of low-value
        # tags. The underlying counters remain available for offline audits.
        keep_exact = {
            "Reward/fr_clearance_progress",
            "V10_Perception/BActiveFraction",
            "V10_Perception/FirstRiserLockedFraction",
            "V10_Perception/FirstRiserConsumedFraction",
            "V10_Perception/Flat/FalseTriggerRate",
            "V10_Perception/Rough/FalseTriggerRate",
            "V10_Curriculum/FullStairFraction",
            "Curriculum/StairTerrainLevelMean",
            "Curriculum/NonStairTerrainLevelMean",
            "Curriculum/StairOfficialUpRate",
            "Curriculum/StairPromotionRate",
        }
        keep_prefixes = ("V10_Core/", "V10_Task/Recent500/Full/")
        return {
            key: value for key, value in metrics.items()
            if key in keep_exact or key.startswith(keep_prefixes)
        }

    def log(self, locs: dict, width=80, pad=35):
        super().log(locs, width=width, pad=pad)
        if self.writer is not None:
            for key, value in self._v10_metrics(locs["obs"]).items():
                self.writer.add_scalar(key, value.item(), locs["it"])


__all__ = ["V10UnifiedRunner"]
