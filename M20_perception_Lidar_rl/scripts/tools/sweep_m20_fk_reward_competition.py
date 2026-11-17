"""Isaac FK sweep: actual wheel lift versus official posture and spatial terms."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.mdp.perceptive_spatial import (
    _semantic_body_ids,
    distance_to_vertical_segment,
    wheel_positions_base,
    wheel_positions_yaw,
)


OUT = Path(__file__).resolve().parents[2] / "artifacts" / "fk_reward_competition"
OUT.mkdir(parents=True, exist_ok=True)


def settle(raw):
    raw.scene.write_data_to_sim()
    for _ in range(args.steps):
        raw.sim.step(render=False)
        raw.scene.update(dt=raw.physics_dt)


def main():
    cfg = parse_env_cfg("M20-Stair-Perceptive-v0", device=args.device, num_envs=2)
    cfg.scene.terrain.max_init_terrain_level = 7
    cfg.curriculum.terrain_levels = None
    cfg.terminations.bad_orientation_2 = None
    env = gym.make("M20-Stair-Perceptive-v0", cfg=cfg)
    env.reset()
    raw = env.unwrapped
    terrain = raw.scene.terrain
    terrain.terrain_levels[:] = torch.tensor((6, 7), device=raw.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    robot = raw.scene["robot"]
    body_ids = _semantic_body_ids(robot)
    leg_ids, names = robot.find_joints(raw.cfg.leg_joint_names, preserve_order=True)
    assert names == raw.cfg.leg_joint_names
    default = robot.data.default_joint_pos.clone()
    z_nom = wheel_positions_base(robot, body_ids)[..., 2].detach().clone()
    # Put both robots before first riser; base scanner remains flat by design.
    state = robot.data.default_root_state.clone()
    state[:, :3] += raw.scene.env_origins
    state[:, 0] += 0.50
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state)
    robot.write_joint_state_to_sim(default, torch.zeros_like(robot.data.default_joint_vel))
    settle(raw)
    z_nom = wheel_positions_base(robot, body_ids)[..., 2].detach().clone()
    raw.episode_length_buf[:] = 1
    raw._perceptive_z_nom = z_nom.clone()
    rows = []
    weight_candidates = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    for alpha in np.linspace(0.0, 0.8, 17):
        q = default.clone()
        # Fold front and hind sagittal chains in their respective sign
        # conventions. Both hip-Y and knee move in the same signed direction.
        for leg in range(4):
            sign = -1.0 if leg < 2 else 1.0
            q[:, leg_ids[3 * leg + 1]] += sign * float(alpha) * 2.0
            q[:, leg_ids[3 * leg + 2]] += sign * float(alpha) * 1.2
        q = torch.maximum(torch.minimum(q, robot.data.soft_joint_pos_limits[..., 1]), robot.data.soft_joint_pos_limits[..., 0])
        # Every FK sample is evaluated from the same precontact root pose;
        # otherwise a few physics settle steps advance the robot past the
        # riser and silently remove the active geometry target.
        robot.write_root_state_to_sim(state, env_ids=torch.arange(raw.num_envs, device=raw.device))
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        raw.scene.write_data_to_sim()
        raw.scene.update(dt=raw.physics_dt)
        # Hold every FK pose at the same true first-riser distance. Joint
        # folding changes wheel X as well as Z; without this correction a pose
        # can appear to satisfy clearance merely by moving the wheel backward
        # out of the collision band. Terrain origin is the configured spawn
        # point and the first riser is +1.10 m from it.
        front_x_rel = wheel_positions_yaw(robot, body_ids)[:, :2, 0].mean(1)
        fixed_d = 0.05
        fixed_state = state.clone()
        fixed_state[:, 0] = raw.scene.env_origins[:, 0] + 1.10 - fixed_d - front_x_rel
        robot.write_root_state_to_sim(fixed_state, env_ids=torch.arange(raw.num_envs, device=raw.device))
        raw.scene.write_data_to_sim()
        raw.scene.update(dt=raw.physics_dt)
        for sensor_name in ("height_scanner", "height_scanner_base"):
            raw.scene[sensor_name].update(raw.physics_dt, force_recompute=True)
        raw.common_step_counter += 1
        raw._perceptive_spatial_step = None
        raw._perceptive_z_nom = z_nom.clone()
        geom = mdp.perceptive_spatial_state(raw)
        hp_cfg = raw.reward_manager.get_term_cfg("hipy_joint_pos_penalty")
        kn_cfg = raw.reward_manager.get_term_cfg("knee_joint_pos_penalty")
        hipy = hp_cfg.func(raw, **hp_cfg.params) * hp_cfg.weight
        knee = kn_cfg.func(raw, **kn_cfg.params) * kn_cfg.weight
        hipx_cfg = raw.reward_manager.get_term_cfg("hipx_joint_pos_penalty")
        hipx = hipx_cfg.func(raw, **hipx_cfg.params) * hipx_cfg.weight
        mirror_cfg = raw.reward_manager.get_term_cfg("joint_mirror")
        mirror = mirror_cfg.func(raw, **mirror_cfg.params) * mirror_cfg.weight
        orientation_cfg = raw.reward_manager.get_term_cfg("flat_orientation_l2")
        orientation = orientation_cfg.func(raw, **orientation_cfg.params) * orientation_cfg.weight
        spatial_raw = mdp.perceptive_spatial_lift(raw)
        spatial = spatial_raw * raw.cfg.rewards.spatial_lift.weight
        lift_fk = wheel_positions_base(robot, body_ids)[..., 2] - z_nom
        assert torch.allclose(geom["actual_lift"], lift_fk, atol=1.0e-5)
        lift = lift_fk.mean(1)
        official_cost = hipy + knee + hipx + mirror + orientation
        active_count = geom["active"].sum(1).clamp_min(1)
        active_progress = torch.where(geom["active"], geom["progress"], 0.0).sum(1) / active_count
        active_z_actual = torch.where(geom["active"], geom["z_actual_world"], 0.0).sum(1) / active_count
        active_z_ref = torch.where(geom["active"], geom["z_ref_world"], 0.0).sum(1) / active_count
        active_distance = torch.where(geom["active"], geom["riser_distance"], 0.0).sum(1) / active_count
        segment_distance = distance_to_vertical_segment(
            geom["riser_distance"], geom["z_actual_world"], geom["z_lower"], geom["z_upper"]
        )
        r_clear = raw.cfg.spatial_guidance.wheel_collision_radius + raw.cfg.spatial_guidance.clearance_margin
        clearance_ok_wheel = (segment_distance >= r_clear - 1.0e-5) & (
            (geom["riser_distance"] > 0.0) | (geom["z_actual_world"] >= geom["z_upper"] + r_clear - 1.0e-5)
        )
        clearance_ok = torch.where(geom["active"], clearance_ok_wheel, torch.ones_like(clearance_ok_wheel)).all(1)
        rows.append(
            torch.stack(
                (
                    torch.full_like(lift, alpha),
                    lift,
                    active_distance,
                    active_z_actual,
                    active_z_ref,
                    active_progress,
                    spatial_raw,
                    spatial,
                    hipy,
                    knee,
                    hipx,
                    mirror,
                    orientation,
                    official_cost,
                    spatial + official_cost,
                    clearance_ok.float(),
                ),
                dim=1,
            ).detach().cpu().numpy()
        )
    rows = np.concatenate(rows)
    path = OUT / "fk_reward_competition.csv"
    np.savetxt(
        path,
        rows,
        delimiter=",",
        header=(
            "alpha,mean_leg_relative_lift,active_distance,z_actual_world,z_ref_world,clearance_progress,"
            "spatial_effective_raw,weighted_spatial,weighted_hipy,weighted_knee,weighted_hipx,weighted_mirror,"
            "weighted_orientation,official_cost,total_relevant,actual_geometry_clearance_ok"
        ),
        comments="",
    )
    print("CSV=" + str(path))
    for env_index, label in enumerate(("17", "20")):
        env_rows = rows[env_index::2]
        nominal = env_rows[0]
        target_candidates = env_rows[(env_rows[:, 5] >= 0.999) & (env_rows[:, 15] > 0.5)]
        if len(target_candidates) == 0:
            print(f"PRELIFT_REWARD_COMPETITION_{label}=FAIL FK_TARGET_NOT_REACHED")
            continue
        target = target_candidates[0]
        delta_official = target[13] - nominal[13]
        print(
            f"PRELIFT_{label} alpha={target[0]:.3f} d={target[2]:.6f} z_actual={target[3]:.6f} "
            f"z_ref={target[4]:.6f} progress={target[5]:.6f} geometry_ok={bool(target[15])} "
            f"delta_official={delta_official:.6f}"
        )
        passing = []
        for weight in weight_candidates:
            delta_spatial = weight * (target[6] - nominal[6])
            delta_net = delta_spatial + delta_official
            print(
                f"WEIGHT_SWEEP_{label} w={weight:.1f} delta_spatial={delta_spatial:.6f} "
                f"delta_official={delta_official:.6f} delta_net={delta_net:.6f}"
            )
            if delta_net > 0.0:
                passing.append(weight)
        if passing:
            print(f"PRELIFT_REWARD_COMPETITION_{label}=PASS MIN_WEIGHT={passing[0]:.1f}")
        else:
            print(f"PRELIFT_REWARD_COMPETITION_{label}=FAIL MIN_WEIGHT_GT_8")
    env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
