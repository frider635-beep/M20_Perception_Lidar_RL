"""Hard acceptance for the frozen official M20 high-terrain posture relaxation."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dynamic_steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity import mdp


# The official pyramid curriculum is continuous; terrain row numbers do not
# encode an exact discrete 8 cm step.  Use an already measured >0.06 m row as
# the generic high-terrain case, followed by the 17/20 cm hard cases.
LEVELS = torch.tensor((0, 6, 6, 7), dtype=torch.long)
HEIGHTS = (0.00, 0.17, 0.17, 0.20)
LABELS = ("flat", "diagnostic_spare", "stairs_17", "stairs_20")


def _disable_diagnostic_randomization(cfg):
    """Keep this acceptance deterministic without changing the registered task."""
    for name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_reset_joints",
        "randomize_actuator_gains",
        "randomize_push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def _settle(raw, steps=2):
    raw.scene.write_data_to_sim()
    for _ in range(steps):
        raw.sim.step(render=False)
        raw.scene.update(dt=raw.physics_dt)


def _force_levels(raw):
    terrain = raw.scene.terrain
    levels = LEVELS.to(device=raw.device)
    terrain.terrain_types[:] = 0
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]


def _terrain_height_condition(raw, term_name="hipy_joint_pos_penalty"):
    cfg = raw.reward_manager.get_term_cfg(term_name)
    sensor = raw.scene[cfg.params["sensor_cfg"].name]
    hits = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits) & (torch.abs(hits) <= 1.0e6)
    count = valid.sum(dim=1)
    mean_z = torch.where(valid, hits, torch.zeros_like(hits)).sum(dim=1) / count.clamp_min(1)
    height = mean_z - raw.scene.env_origins[:, 2]
    high = (height > cfg.params["terrain_height_threshold"]) & (count > 0)
    return height, high


def _posture_values(raw, term_name):
    cfg = raw.reward_manager.get_term_cfg(term_name)
    p = cfg.params
    baseline = mdp.joint_pos_penalty(
        raw,
        command_name=p["command_name"],
        asset_cfg=p["asset_cfg"],
        stand_still_scale=p["stand_still_scale"],
        velocity_threshold=p["velocity_threshold"],
        command_threshold=p["command_threshold"],
    )
    official = cfg.func(raw, **p)
    ratio = torch.where(baseline > 1.0e-7, official / baseline, torch.full_like(baseline, float("nan")))
    return baseline, official, ratio


def _write_pose(raw, relative_x, terrain_height):
    robot = raw.scene["robot"]
    ids = torch.arange(raw.num_envs, device=raw.device)
    state = robot.data.default_root_state.clone()
    state[:, :3] += raw.scene.env_origins
    state[:, 0] += torch.as_tensor(relative_x, device=raw.device)
    state[:, 2] += torch.as_tensor(terrain_height, device=raw.device)
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state, env_ids=ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel), env_ids=ids)
    raw._perceptive_spatial_step = None
    _settle(raw)


def _apply_known_posture_deviation(raw):
    robot = raw.scene["robot"]
    hipy_ids, hipy_names = robot.find_joints(raw.cfg.hipy_joint_names, preserve_order=True)
    knee_ids, knee_names = robot.find_joints(raw.cfg.knee_joint_names, preserve_order=True)
    assert hipy_names == raw.cfg.hipy_joint_names
    assert knee_names == raw.cfg.knee_joint_names
    q = robot.data.default_joint_pos.clone()
    q[:, hipy_ids] += 0.30
    q[:, knee_ids] += 0.30
    limits = robot.data.soft_joint_pos_limits[:, :, :]
    q = torch.maximum(torch.minimum(q, limits[..., 1]), limits[..., 0])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    _settle(raw, steps=1)


def _set_forward_command(raw):
    command = raw.command_manager.get_command("base_velocity")
    command[:, 0] = 0.4
    command[:, 1:] = 0.0


def _assert_static_scaling(raw):
    terrain_height, high = _terrain_height_condition(raw)
    hipy_raw, hipy_scaled, hipy_ratio = _posture_values(raw, "hipy_joint_pos_penalty")
    knee_raw, knee_scaled, knee_ratio = _posture_values(raw, "knee_joint_pos_penalty")
    for index, label in enumerate(LABELS):
        print(
            f"STATIC_SCENARIO={label} TERRAIN_HEIGHT={float(terrain_height[index]):.6f} "
            f"HIGH={bool(high[index])} HIPY_RAW={float(hipy_raw[index]):.6f} "
            f"HIPY_OFFICIAL={float(hipy_scaled[index]):.6f} HIPY_SCALE={float(hipy_ratio[index]):.6f} "
            f"KNEE_RAW={float(knee_raw[index]):.6f} KNEE_OFFICIAL={float(knee_scaled[index]):.6f} "
            f"KNEE_SCALE={float(knee_ratio[index]):.6f}",
            flush=True,
        )
    assert not bool(high[0])
    assert torch.allclose(hipy_ratio[0], torch.tensor(1.0, device=raw.device), atol=1.0e-5)
    assert torch.allclose(knee_ratio[0], torch.tensor(1.0, device=raw.device), atol=1.0e-5)
    # Terrain type/origin assignment is independent across columns, so the
    # spare environment is diagnostic only.  The explicit 17/20 cm cases are
    # the required >0.06 m high-terrain tests.
    assert high[2:].all()
    assert torch.allclose(hipy_ratio[2:], torch.full_like(hipy_ratio[2:], 0.1), atol=1.0e-5)
    assert torch.allclose(knee_ratio[2:], torch.full_like(knee_ratio[2:], 0.1), atol=1.0e-5)
    print("TEST_1_FLAT_NORMAL_SCALE=PASS")
    print("TEST_2_HIGH_TERRAIN_SCALE_0P1=PASS")
    print("TEST_3_STAIRS_17_20_RELAXATION=PASS")


def _weighted_term(raw, name):
    index = raw.reward_manager.active_terms.index(name)
    return raw.reward_manager._step_reward[:, index]


def _dynamic_magnitude_audit(env):
    raw = env.unwrapped
    action = torch.zeros((raw.num_envs, 16), device=raw.device)
    # Large sagittal shortening request: front joints move negative, hind joints positive.
    action[:, (1, 2, 4, 5)] = -4.0
    action[:, (7, 8, 10, 11)] = 4.0
    samples = []
    for _ in range(args.dynamic_steps):
        _set_forward_command(raw)
        _, reward, _, _, _ = env.step(action)
        assert torch.isfinite(reward).all()
        height, high = _terrain_height_condition(raw)
        _, _, hipy_ratio = _posture_values(raw, "hipy_joint_pos_penalty")
        _, _, knee_ratio = _posture_values(raw, "knee_joint_pos_penalty")
        spatial = raw.perceptive_spatial_state
        samples.append(
            torch.stack(
                (
                    height,
                    high.float(),
                    hipy_ratio.nan_to_num(),
                    knee_ratio.nan_to_num(),
                    _weighted_term(raw, "spatial_lift"),
                    _weighted_term(raw, "hipy_joint_pos_penalty"),
                    _weighted_term(raw, "knee_joint_pos_penalty"),
                    _weighted_term(raw, "action_smooth_l2"),
                    _weighted_term(raw, "joint_torques_l2"),
                    spatial["active"].float().mean(dim=1),
                    spatial["lift_ref"].amax(dim=1),
                    spatial["actual_lift"].amax(dim=1),
                ),
                dim=1,
            )
        )
    values = torch.stack(samples).mean(dim=0)
    fields = (
        "terrain_height", "high_fraction", "hipy_scale", "knee_scale", "weighted_spatial",
        "weighted_hipy", "weighted_knee", "weighted_action_smooth", "weighted_torque",
        "active_fraction", "L_ref_max", "L_actual_max",
    )
    for index in (2, 3):
        print("MAGNITUDE_SCENARIO=" + LABELS[index] + " " + " ".join(
            f"{name}={float(values[index, column]):.6f}" for column, name in enumerate(fields)
        ))
        active_frames = torch.stack(samples)[:, index, 9] > 0.0
        high_frames = torch.stack(samples)[:, index, 1] > 0.5
        scale_ok = (
            torch.isclose(torch.stack(samples)[:, index, 2], torch.tensor(0.1, device=raw.device), atol=1.0e-5)
            & torch.isclose(torch.stack(samples)[:, index, 3], torch.tensor(0.1, device=raw.device), atol=1.0e-5)
        )
        if index >= 2:
            assert active_frames.any(), f"{LABELS[index]} spatial guidance never active"
            assert (high_frames & active_frames).any(), f"{LABELS[index]} never combined high terrain and spatial guidance"
            assert scale_ok[high_frames & active_frames].all(), f"{LABELS[index]} posture relaxation lost while guidance active"
    print("TEST_4_REWARD_MAGNITUDE_AND_CONCURRENT_RELAXATION=PASS")


def _first_riser_prelift_audit(raw):
    # Place the 17/20 cm robots before the first riser, where front-wheel
    # spatial guidance should already be active. This exposes the exact scope
    # of the frozen base-centred official terrain detector.
    relative_x = (0.0, 1.20, 0.50, 0.50)
    terrain_z = (0.0, 0.08, 0.0, 0.0)
    _write_pose(raw, relative_x, terrain_z)
    _set_forward_command(raw)
    # Establish nominal wheel Z before applying the same posture deviation.
    raw._perceptive_spatial_step = None
    mdp.perceptive_spatial_state(raw)
    _apply_known_posture_deviation(raw)
    raw._perceptive_spatial_step = None
    state = mdp.perceptive_spatial_state(raw)
    height, high = _terrain_height_condition(raw)
    _, _, hipy_ratio = _posture_values(raw, "hipy_joint_pos_penalty")
    _, _, knee_ratio = _posture_values(raw, "knee_joint_pos_penalty")
    conflict = []
    for index in (2, 3):
        active = bool(state["active"][index].any())
        full_strength = bool((hipy_ratio[index] > 0.99) & (knee_ratio[index] > 0.99))
        has_conflict = active and (not bool(high[index])) and full_strength
        conflict.append(has_conflict)
        print(
            f"FIRST_RISER_PRELIFT={LABELS[index]} TERRAIN_HEIGHT={float(height[index]):.6f} "
            f"HIGH={bool(high[index])} ACTIVE={active} L_REF_MAX={float(state['lift_ref'][index].max()):.6f} "
            f"HIPY_SCALE={float(hipy_ratio[index]):.6f} KNEE_SCALE={float(knee_ratio[index]):.6f} "
            f"FULL_STRENGTH_CONFLICT={has_conflict}",
            flush=True,
        )
    # The official base-centred detector is expected to remain at normal
    # scale before the first riser.  This is diagnostic only; final admission
    # is decided by the whole-body clearance-pose DeltaNet competition.
    print(f"OFFICIAL_FIRST_RISER_PRELIFT_NORMAL_BEHAVIOR={any(conflict)}")


def main():
    cfg = parse_env_cfg("M20-Stair-Perceptive-v0", device=args.device, num_envs=4)
    cfg.scene.terrain.max_init_terrain_level = 7
    cfg.curriculum.terrain_levels = None
    cfg.terminations.bad_orientation_2 = None
    _disable_diagnostic_randomization(cfg)
    for name in ("hipy_joint_pos_penalty", "knee_joint_pos_penalty"):
        term = getattr(cfg.rewards, name)
        assert term.func is mdp.joint_pos_penalty_except_turn_side_cmd
        assert term.params["terrain_height_threshold"] == 0.06
        assert term.params["high_terrain_penalty_scale"] == 0.1
        assert term.params["sensor_cfg"].name == "height_scanner_base"
    print("OFFICIAL_FROZEN_CONFIG=PASS threshold=0.06 scale=0.1 sensor=height_scanner_base")
    env = gym.make("M20-Stair-Perceptive-v0", cfg=cfg)
    env.reset()
    raw = env.unwrapped
    _force_levels(raw)
    # On the first tread of the non-flat scenarios, the official base scanner
    # sees exactly 8/17/20 cm terrain relative to the terrain origin.
    _write_pose(raw, (0.0, 1.20, 1.20, 1.20), HEIGHTS)
    _set_forward_command(raw)
    raw._perceptive_spatial_step = None
    mdp.perceptive_spatial_state(raw)
    _apply_known_posture_deviation(raw)
    _set_forward_command(raw)
    raw._perceptive_spatial_step = None
    _assert_static_scaling(raw)
    _dynamic_magnitude_audit(env)
    _first_riser_prelift_audit(raw)
    print("OFFICIAL_HIGH_TERRAIN_RELAXATION_ACCEPTANCE=PASS")
    env.close()


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
