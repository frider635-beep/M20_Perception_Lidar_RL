"""Isaac runtime checks for Level-1 guidance and stair curriculum decisions."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils
from rl_training.tasks.manager_based.locomotion.velocity import mdp


def _disable_randomization(cfg):
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


def _settle(raw, steps=12):
    raw.scene.write_data_to_sim()
    for _ in range(steps):
        raw.sim.step(render=False)
        raw.scene.update(dt=raw.physics_dt)


def _force_levels(raw, levels):
    terrain = raw.scene.terrain
    levels = torch.as_tensor(levels, device=raw.device, dtype=torch.long)
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]


def _write_default_pose(raw, relative_x, relative_z=None, roll=None):
    robot = raw.scene["robot"]
    ids = torch.arange(raw.num_envs, device=raw.device)
    state = robot.data.default_root_state.clone()
    state[:, :3] += raw.scene.env_origins
    state[:, 0] += torch.as_tensor(relative_x, device=raw.device)
    if relative_z is not None:
        state[:, 2] += torch.as_tensor(relative_z, device=raw.device)
    if roll is not None:
        roll = torch.as_tensor(roll, device=raw.device)
        zeros = torch.zeros_like(roll)
        state[:, 3:7] = math_utils.quat_from_euler_xyz(roll, zeros, zeros)
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state, env_ids=ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel), env_ids=ids)
    raw._perceptive_spatial_step = None
    if hasattr(raw, "_perceptive_z_nom"):
        del raw._perceptive_z_nom
    _settle(raw)


def level_one_guidance_test(raw):
    # Keep one low-level reference and place three positive-stair environments
    # near their first riser so the continuous guide is definitely active.
    _force_levels(raw, (0, 1, 1, 1))
    _write_default_pose(raw, (0.00, 0.80, 0.80, 0.80))
    raw._perceptive_spatial_step = None
    state = mdp.perceptive_spatial_state(raw)
    raw_spatial = mdp.perceptive_spatial_lift(raw)
    weighted_spatial = raw_spatial * raw.cfg.rewards.spatial_lift.weight
    for index in range(raw.num_envs):
        print(
            f"LEVEL1_SCENARIO=env{index} LEVEL={int(raw.scene.terrain.terrain_levels[index])} "
            f"VALID={int(state['valid'][index].sum())} ACTIVE={int(state['active'][index].sum())} "
            f"RISER_HEIGHT_MAX={float(state['riser_height'][index].max()):.6f} "
            f"DISTANCE_MIN={float(state['riser_distance'][index].amin()):.6f} "
            f"GUIDE_GAIN={float(state['guide_gain'][index]):.6f} "
            f"SPATIAL_RAW={float(state['spatial_raw'][index]):.6f} "
            f"SPATIAL_WEIGHTED={float(weighted_spatial[index]):.6f}",
            flush=True,
        )
    assert state["valid"][1:].any(dim=1).all()
    assert state["active"][1:].any(dim=1).all()
    assert torch.allclose(state["guide_gain"][1:], torch.full_like(state["guide_gain"][1:], 0.30))
    assert (raw_spatial[1:] > 0.0).all()
    assert (weighted_spatial[1:] > 0.0).all()
    print("LEVEL0_FLAT_MATH_ZERO_VERIFIED_SEPARATELY=PASS")
    print("LEVEL1_POSITIVE_STAIR_SPATIAL_GUIDANCE_ACTIVE=PASS")


def curriculum_decision_test(raw):
    # All start at Level 3. Scenarios: success, low progress, partial hold, fall.
    _force_levels(raw, (3, 3, 3, 3))
    relative_x = (3.55, 0.00, 1.20, 1.20)
    relative_z = (0.48, 0.00, 0.00, 0.00)
    roll = (0.00, 0.00, 0.00, 1.30)
    _write_default_pose(raw, relative_x, relative_z, roll)
    before = raw.scene.terrain.terrain_levels.clone()
    ids = torch.arange(raw.num_envs, device=raw.device)
    mdp.perceptive_stair_levels(raw, ids)
    after = raw.scene.terrain.terrain_levels.clone()
    success = raw.perceptive_curriculum_success.bool()
    demote = raw.perceptive_curriculum_demote.bool()
    for index, label in enumerate(("success", "low_progress", "partial_hold", "fall")):
        print(
            f"CURRICULUM_SCENARIO={label} BEFORE={int(before[index])} AFTER={int(after[index])} "
            f"SUCCESS={bool(success[index])} DEMOTE={bool(demote[index])} "
            f"GRAVITY={raw.scene['robot'].data.projected_gravity_b[index].tolist()}",
            flush=True,
        )
    assert int(after[0]) == 4 and bool(success[0]) and not bool(demote[0])
    assert int(after[1]) == 2 and not bool(success[1]) and bool(demote[1])
    assert int(after[2]) == 3 and not bool(success[2]) and not bool(demote[2])
    assert int(after[3]) == 2 and not bool(success[3]) and bool(demote[3])
    print("CURRICULUM_PROMOTION=PASS")
    print("CURRICULUM_LOW_PROGRESS_DEMOTION=PASS")
    print("CURRICULUM_PARTIAL_PROGRESS_HOLD=PASS")
    print("CURRICULUM_FALL_DEMOTION=PASS")


def main():
    cfg = parse_env_cfg("M20-Stair-Perceptive-v0", device=args.device, num_envs=4)
    cfg.scene.terrain.max_init_terrain_level = 7
    cfg.curriculum.terrain_levels = None
    cfg.terminations.bad_orientation_2 = None
    _disable_randomization(cfg)
    env = gym.make("M20-Stair-Perceptive-v0", cfg=cfg)
    env.reset()
    raw = env.unwrapped
    level_one_guidance_test(raw)
    curriculum_decision_test(raw)
    print("M20_PERCEPTIVE_RUNTIME_LOGIC=PASS")
    env.close()


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
