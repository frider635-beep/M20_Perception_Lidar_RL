"""Read-only rollout/episode metrics with explicit denominators and censoring."""
import torch


class V135BehaviorMetrics:
    PREFIX = "Diag/V135Behavior/"

    def __init__(self, env):
        self.env = env
        self.last_step = -1
        self.sums = {}
        self.b1 = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.b2 = torch.zeros_like(self.b1)
        self.b3 = torch.zeros_like(self.b1)
        self.exit_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.recovered = torch.zeros_like(self.b1)
        self.defect_run = torch.zeros(env.num_envs, device=env.device)
        self.terrain_levels = int(env.cfg.scene.terrain.terrain_generator.num_rows)
        self.command_bins = ((0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0))

    def add(self, name, value):
        value = value.float().sum().detach()
        if name not in self.sums:
            self.sums[name] = torch.zeros((), device=self.env.device)
        self.sums[name] += value

    def record(self):
        e = self.env
        if self.last_step == e.common_step_counter or not hasattr(e, "_v135_whole_body"):
            return
        self.last_step = int(e.common_step_counter)
        self.sums = {}
        s, whole, flat = e._v135_riser_identity, e._v135_whole_body, e._v135_post_stair
        terrain = e.scene.terrain.terrain_types
        ascent = torch.zeros_like(self.b1)
        for kind in e.cfg.v11_perception.ascending_stair_terrain_type_ids:
            ascent |= terrain == int(kind)
        partial = getattr(e, "_v135_stage_started", torch.zeros_like(self.b1))
        full = ascent & ~partial
        self.b1 |= full & s["first_lift_b1_event"] & s["episode_ordered_valid"]
        self.b2 |= self.b1 & s["ordered"] & (s["target_id"] == 3) & s["episode_ordered_valid"]
        self.b3 |= self.b2 & s["ordered"] & (s["target_id"] == 4) & s["episode_ordered_valid"]
        if "reward_snapshot" in s:
            snap = s["reward_snapshot"]
            for k in range(1, int(s["max_flight_risers"]) + 1):
                mask = full & (snap["target_id"] == k) & snap["target_valid"]
                prefix = f"AscentByRiser/B{k}/"
                for key, value in (("frames", mask),
                    ("height_missing_frames", mask & ~snap["target_height_valid"]),
                    ("valid_events", mask & s["valid"]),
                    ("ordered_events", mask & s["ordered"]),
                    ("error_events", mask & (s["wrong_side"] | s["skip"] | s["duplicate"] | s["dual_skip"]))):
                    self.add(prefix + key, value)
            if hasattr(e, "_v135_ascent_support"):
                body = e._v135_ascent_support
                active = full & snap["target_valid"] & body["measured"]
                for key in ("quality", "support", "slip", "roll_error", "pitch_error", "ground_valid"):
                    self.add("AscentSupport/" + key + "_sum", body[key] * active)
                self.add("AscentSupport/frames", active)
                for phase in ("preparation", "swing", "acceptance"):
                    mask = full & body[phase]
                    self.add(f"AscentEntry/{phase}_frames", mask)
                    self.add(f"AscentEntry/{phase}_roll_sum", body["roll_error"] * mask)
                self.add("AscentEntry/B1_frames", active & (snap["target_id"] == 1))
                self.add("AscentEntry/B1_roll_sum", body["roll_error"] * active * (snap["target_id"] == 1))
                self.add("AscentEntry/outside_support_frames", active & (body["support_margin"] < 0))
                self.add("AscentEntry/settlement_quality_sum", body["settlement_quality"] * full * s["valid"])
                self.add("AscentEntry/settlements", full & s["valid"])
                for evidence in ("settlement_evidence", "missing_preparation", "missing_swing", "missing_acceptance"):
                    self.add("AscentEntry/" + evidence, full & s["valid"] & body[evidence])
            for width in sorted(set(e.cfg.v135_riser_identity.ascent_tread_widths)):
                mask = full & (s["step_width"] - width).abs().lt(1e-5)
                prefix = f"AscentByTread/T{round(width * 100):02d}/"
                for key, value in (("frames", mask), ("valid_events", mask & s["valid"]),
                        ("ordered_events", mask & s["ordered"]),
                        ("skip_events", mask & (s["skip"] | s["dual_skip"]))):
                    self.add(prefix + key, value)
        self.add("Ascent/actionable_frames", ascent & s["target_valid"])
        self.add("Ascent/full_frames", full)
        self.add("Ascent/partial_frames", ascent & partial)
        for key in ("valid", "ordered", "wrong_side", "duplicate", "skip", "dual_skip"):
            self.add("Ascent/" + key + "_events", s[key] & full)
            self.add("Partial/" + key + "_events", s[key] & partial)
        for key in ("first_lift_event", "first_lift_empty_event", "first_lift_skip_event", "first_lift_b1_event",
                    "b1_identity_rejected", "native_unconfirmed_event"):
            self.add("Ascent/" + key, s[key] & full)
        velocity = e.scene["robot"].data.root_lin_vel_w[:, :2]
        heading = e.command_manager.get_term("base_velocity").v11_path_heading_w
        speed = (velocity * torch.stack((heading.cos(), heading.sin()), 1)).sum(1)
        self.add("Ascent/speed_sum", speed * full)
        commanded = e.command_manager.get_command("base_velocity")[:, 0]
        for level in range(self.terrain_levels):
            mask = full & (e.scene.terrain.terrain_levels == level)
            prefix = f"AscentByLevel/L{level}/"
            self.add(prefix + "frames", mask)
            self.add(prefix + "valid_events", mask & s["valid"])
            self.add(prefix + "ordered_events", mask & s["ordered"])
            self.add(prefix + "skip_events", mask & s["skip"])
            self.add(prefix + "speed_sum", speed * mask)
        for index, (lower, upper) in enumerate(self.command_bins):
            mask = full & (commanded >= lower) & (commanded < upper)
            prefix = f"AscentByCommand/C{index}/"
            self.add(prefix + "frames", mask)
            self.add(prefix + "valid_events", mask & s["valid"])
            self.add(prefix + "ordered_events", mask & s["ordered"])
            self.add(prefix + "skip_events", mask & s["skip"])
            self.add(prefix + "speed_sum", speed * mask)
        self.add("Ascent/strict_platform_events", e._v11_state["landing"]["platform_exit_event"]
                 & s["episode_ordered_valid"] & s["flight_complete"] & full)
        down = whole["active"]
        self.add("Descent/frames", down)
        self.add("Descent/speed_sum", speed * down)
        for name, value in (("slip", whole["slip"]), ("quality", whole["quality"]),
                            ("terrain_pitch_error", whole["pitch_error"]),
                            ("ground_evidence", whole["ground_valid"]),
                            ("support", whole["support"])):
            self.add("Descent/" + name + "_sum", value * down)
        extension = (whole["leg_length_ratio"][:, 2:] > e.cfg.v135_descent_support_transfer.persistent_extension_ratio).any(1)
        folded = (whole["leg_length_ratio"][:, 2:] < e.cfg.v135_descent_support_transfer.folded_length_ratio).any(1)
        reach = (whole["signed_rear_reach"] > e.cfg.v135_descent_support_transfer.workspace_free).any(1)
        for name, mask in (("extension", extension), ("folding", folded), ("backward_reach", reach)):
            self.add("Descent/" + name + "_frames", down & mask)
        defect = down & (extension | folded | reach)
        self.defect_run = torch.where(defect, self.defect_run + float(e.step_dt), torch.zeros_like(self.defect_run))
        self.add("Descent/prolonged_defect_frames", self.defect_run >= e.cfg.v135_descent_support_transfer.defect_duration)
        new_flight = ~flat["recovery_started"] & (self.exit_step >= 0)
        abandoned = new_flight & ~self.recovered
        self.add("Flat/censored_exits", abandoned)
        self.add("Flat/censored_seconds_sum", (e.common_step_counter - self.exit_step).float() * e.step_dt * abandoned)
        self.exit_step[new_flight] = -1
        self.recovered[new_flight] = False
        started = flat["recovery_started"] & (self.exit_step < 0)
        self.exit_step[started] = e.common_step_counter
        self.add("Flat/geometric_exits", started)
        recovered = flat["completion_event"] & (self.exit_step >= 0) & ~self.recovered
        self.add("Flat/recoveries", recovered)
        self.add("Flat/recovery_seconds_sum", (e.common_step_counter - self.exit_step).float() * float(e.step_dt) * recovered)
        self.recovered |= recovered
        self.add("Flat/pending_frames", (self.exit_step >= 0) & ~self.recovered)

    def finish(self, ids):
        self.record()
        e = self.env
        if not hasattr(e, "_v135_riser_identity"):
            return
        types = e.scene.terrain.terrain_types[ids]
        ascent = torch.zeros_like(types, dtype=torch.bool)
        for kind in e.cfg.v11_perception.ascending_stair_terrain_type_ids:
            ascent |= types == int(kind)
        partial = getattr(e, "_v135_stage_started", torch.zeros_like(self.b1))[ids]
        full = ascent & ~partial & (e.episode_length_buf[ids] > 0)
        self.add("Ascent/full_episodes", full)
        ledger = e._v135_riser_identity
        self.add("Ascent/full_chain_episodes", full & ledger["flight_complete"][ids] & ledger["episode_ordered_valid"][ids])
        self.add("Ascent/strict_prefix_sum", ledger["strict_prefix_depth"][ids] * full)
        self.add("Ascent/flight_length_sum", full.float() * ledger["flight_riser_count"][ids])
        for width in sorted(set(e.cfg.v135_riser_identity.ascent_tread_widths)):
            mask = full & (ledger["step_width"][ids] - width).abs().lt(1e-5)
            prefix = f"AscentByTread/T{round(width * 100):02d}/"
            self.add(prefix + "episodes", mask)
            self.add(prefix + "full_chain_episodes", mask & ledger["flight_complete"][ids] & ledger["episode_ordered_valid"][ids])
        for k in range(1, int(ledger["max_flight_risers"]) + 1):
            self.add(f"AscentByRiser/B{k}/prefix_episodes", full & (ledger["strict_prefix_depth"][ids] >= k))

        for name, mask in (("B1", self.b1), ("B2", self.b2), ("B3", self.b3)):
            self.add("Ascent/" + name + "_episodes", mask[ids] & full)
        censored = (self.exit_step[ids] >= 0) & ~self.recovered[ids]
        self.add("Flat/censored_exits", censored)
        self.add("Flat/censored_seconds_sum", (e.common_step_counter - self.exit_step[ids]).float() * e.step_dt * censored)
        for value in (self.b1, self.b2, self.b3, self.recovered):
            value[ids] = False
        self.exit_step[ids] = -1
        self.defect_run[ids] = 0

    def publish(self):
        # Installed RSL Logger averages each extra across rollout steps and only
        # visits keys in the FIRST step. Emit every key every step; step_count*T
        # therefore becomes the exact rollout total after its mean reduction.
        log = dict(self.env.extras.get("log", {}))
        self.env.extras["log"] = log
        for key in list(log):
            if key.startswith(self.PREFIX):
                del log[key]
        for key in (
            "Ascent/first_lift_event", "Ascent/first_lift_empty_event", "Ascent/first_lift_skip_event",
            "Ascent/first_lift_b1_event", "Ascent/b1_identity_rejected", "Ascent/native_unconfirmed_event",
            "Ascent/full_chain_episodes", "Ascent/strict_prefix_sum", "Ascent/flight_length_sum",
            "Ascent/full_episodes", "Ascent/B1_episodes", "Ascent/B2_episodes", "Ascent/B3_episodes",
            "Ascent/actionable_frames", "Ascent/full_frames", "Ascent/partial_frames", "Ascent/speed_sum",
            "Ascent/strict_platform_events", "Ascent/valid_events", "Ascent/ordered_events",
            "Ascent/wrong_side_events", "Ascent/duplicate_events", "Ascent/skip_events",
            "Ascent/dual_skip_events", "Partial/valid_events", "Partial/ordered_events",
            "Partial/wrong_side_events", "Partial/duplicate_events", "Partial/skip_events",
            "Partial/dual_skip_events", "Descent/frames", "Descent/speed_sum", "Descent/slip_sum",
            "Descent/quality_sum", "Descent/terrain_pitch_error_sum", "Descent/ground_evidence_sum",
            "Descent/support_sum", "Descent/extension_frames", "Descent/folding_frames",
            "Descent/backward_reach_frames", "Descent/prolonged_defect_frames", "Flat/geometric_exits",
            "Flat/recoveries", "Flat/recovery_seconds_sum", "Flat/censored_exits",
            "Flat/censored_seconds_sum", "Flat/pending_frames",
        ):
            self.sums.setdefault(key, torch.zeros((), device=self.env.device))
        ledger = getattr(self.env, "_v135_riser_identity", {})
        for k in range(1, int(ledger.get("max_flight_risers", 0)) + 1):
            for suffix in ("frames", "height_missing_frames", "valid_events", "ordered_events", "error_events", "prefix_episodes"):
                self.sums.setdefault(f"AscentByRiser/B{k}/{suffix}", torch.zeros((), device=self.env.device))
        for key in ("frames", "quality_sum", "support_sum", "slip_sum", "roll_error_sum", "pitch_error_sum", "ground_valid_sum"):
            self.sums.setdefault("AscentSupport/" + key, torch.zeros((), device=self.env.device))
        for width in sorted(set(self.env.cfg.v135_riser_identity.ascent_tread_widths)):
            for suffix in ("frames", "valid_events", "ordered_events", "skip_events", "episodes", "full_chain_episodes"):
                self.sums.setdefault(f"AscentByTread/T{round(width * 100):02d}/" + suffix, torch.zeros((), device=self.env.device))
        for suffix in ("settlement_evidence", "missing_preparation", "missing_swing", "missing_acceptance", "B1_frames", "B1_roll_sum", "outside_support_frames", "settlement_quality_sum", "settlements",
                       "preparation_frames", "preparation_roll_sum", "swing_frames", "swing_roll_sum", "acceptance_frames", "acceptance_roll_sum"):
            self.sums.setdefault("AscentEntry/" + suffix, torch.zeros((), device=self.env.device))
        horizon = float(self.env.cfg.v135_longtrain.rollout_steps_per_update)
        for prefix in [f"AscentByLevel/L{level}/" for level in range(self.terrain_levels)] + [
            f"AscentByCommand/C{index}/" for index in range(len(self.command_bins))]:
            for suffix in ("frames", "valid_events", "ordered_events", "skip_events", "speed_sum"):
                self.sums.setdefault(prefix + suffix, torch.zeros((), device=self.env.device))
        for key, value in self.sums.items():
            log[self.PREFIX + key] = value.clone() * horizon
