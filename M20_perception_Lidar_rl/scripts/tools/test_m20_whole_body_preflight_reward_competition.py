"""Final whole-body pre-lift reward competition on fixed 17/20 cm delivery stairs."""

import argparse
import copy

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--height", type=int, choices=(17, 20), required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.perceptive_terrains import (
    M20_DELIVERY_17_TERRAINS_CFG,
    M20_DELIVERY_20_TERRAINS_CFG,
)
from rl_training.tasks.manager_based.locomotion.velocity.mdp.perceptive_spatial import _semantic_body_ids


def disable_randomization(cfg):
    for name in (
        "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia", "randomize_com_positions", "randomize_apply_external_force_torque",
        "randomize_reset_joints", "randomize_actuator_gains", "randomize_push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def sync_raycasters(raw, root_state):
    pose_xyzw = root_state[:, :7].clone()
    pose_xyzw[:, 3:] = math_utils.convert_quat(pose_xyzw[:, 3:], to="xyzw")
    ids = torch.arange(raw.num_envs, dtype=torch.int32, device=raw.device)
    for name in ("height_scanner", "height_scanner_base"):
        sensor = raw.scene[name]
        if hasattr(sensor, "_view") and hasattr(sensor._view, "set_root_transforms"):
            sensor._view.set_root_transforms(pose_xyzw, indices=ids)
        sensor.reset()
        sensor.update(raw.physics_dt, force_recompute=True)


def set_pose_and_settle(raw, q, root_x=0.72):
    robot = raw.scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += raw.scene.env_origins
    root[:, 0] += root_x
    root[:, 7:] = 0.0
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    robot.set_joint_velocity_target(torch.zeros_like(q))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    sync_raycasters(raw, root)
    for _ in range(int(0.5 / raw.physics_dt)):
        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        raw.scene.update(dt=raw.physics_dt)
    raw.common_step_counter += 1
    raw._perceptive_spatial_step = None


def write_exact_pose(raw, q, root_x=0.72):
    """Write the comparison pose without allowing contact to change it."""
    robot = raw.scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += raw.scene.env_origins
    root[:, 0] += root_x
    root[:, 7:] = 0.0
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    robot.set_joint_velocity_target(torch.zeros_like(q))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.scene.update(dt=0.0)
    sync_raycasters(raw, root)
    raw.common_step_counter += 1
    raw._perceptive_spatial_step = None


def weighted(raw, name):
    cfg = raw.reward_manager.get_term_cfg(name)
    return cfg.func(raw, **cfg.params) * cfg.weight


def reward_snapshot(raw):
    raw.command_manager.get_command("base_velocity")[:, 0] = 0.4
    raw.command_manager.get_command("base_velocity")[:, 1:] = 0.0
    raw._perceptive_spatial_step = None
    geom = mdp.perceptive_spatial_state(raw)
    terms = {
        "spatial": weighted(raw, "spatial_lift"),
        "hipx": weighted(raw, "hipx_joint_pos_penalty"),
        "hipy": weighted(raw, "hipy_joint_pos_penalty"),
        "knee": weighted(raw, "knee_joint_pos_penalty"),
        "mirror": weighted(raw, "joint_mirror"),
        "orientation": weighted(raw, "flat_orientation_l2"),
        "false_lift": weighted(raw, "false_lift"),
    }
    net = sum(terms.values())
    return geom, terms, net


def main():
    height_m = args.height / 100.0
    delivery = M20_DELIVERY_17_TERRAINS_CFG if args.height == 17 else M20_DELIVERY_20_TERRAINS_CFG
    cfg = parse_env_cfg("M20-Stair-Perceptive-v0", device=args.device, num_envs=5)
    disable_randomization(cfg)
    cfg.curriculum.terrain_levels = None
    cfg.terminations.bad_orientation_2 = None
    cfg.scene.terrain.terrain_generator = copy.deepcopy(delivery)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.commands.base_velocity.debug_vis = False
    cfg.seed = 42
    env = gym.make("M20-Stair-Perceptive-v0", cfg=cfg)
    env.reset()
    raw = env.unwrapped
    raw.scene.terrain.terrain_levels[:] = 0
    raw.scene.terrain.terrain_types[:] = 0
    raw.scene.terrain.env_origins[:] = raw.scene.terrain.terrain_origins[0, 0]
    # The deterministic delivery generator has one physical row, but these
    # heights correspond to official curriculum levels 6/7.  Preserve the
    # fixed physical origin while selecting the final task's frozen guide gain.
    raw.scene.terrain.terrain_levels[:] = 6 if args.height == 17 else 7
    robot = raw.scene["robot"]
    wheel_ids = _semantic_body_ids(robot)
    leg_ids, names = robot.find_joints(raw.cfg.leg_joint_names, preserve_order=True)
    assert names == raw.cfg.leg_joint_names
    default = robot.data.default_joint_pos.clone()

    alpha_values = (0.15, 0.175, 0.20, 0.225, 0.25) if args.height == 17 else (0.15, 0.1875, 0.225, 0.2625, 0.30)
    alpha = torch.tensor(alpha_values, device=raw.device)
    q = default.clone()
    q[:, leg_ids[1]] -= alpha * 2.0
    q[:, leg_ids[2]] -= alpha * 1.2
    limits = robot.data.soft_joint_pos_limits
    unclipped_in_limits = ((q >= limits[..., 0]) & (q <= limits[..., 1])).all(dim=1)
    q = torch.maximum(torch.minimum(q, limits[..., 1]), limits[..., 0])
    # Match the already accepted whole-body setup: one default-pose physics
    # step before commanding the candidate pose.
    write_exact_pose(raw, default)
    raw.scene.write_data_to_sim()
    raw.sim.step(render=False)
    raw.scene.update(dt=raw.physics_dt)

    # First prove that each candidate is dynamically reachable with real
    # gravity, official PD/actuators and three-wheel support.
    set_pose_and_settle(raw, q)
    dynamic_geom = mdp.perceptive_spatial_state(raw)

    sensor = raw.scene["contact_forces"]
    sensor_names = list(sensor.body_names)
    force_ids = torch.tensor([sensor_names.index(n) for n in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")], device=raw.device)
    support = sensor.data.net_forces_w[:, force_ids].norm(dim=-1)[:, 1:] > 1.0
    gravity = robot.data.projected_gravity_b
    attitude_ok = gravity[:, :2].abs().amax(dim=1) < 0.55
    all_z = robot.data.body_pos_w[:, :, 2] - raw.scene.env_origins[:, None, 2]
    nonwheel = torch.ones(all_z.shape[1], dtype=torch.bool, device=raw.device)
    nonwheel[wheel_ids] = False
    penetration_ok = all_z[:, nonwheel].amin(dim=1) > 0.015
    clearance = dynamic_geom["progress"][:, 0] >= 0.999
    feasible = unclipped_in_limits & (support.sum(dim=1) == 3) & attitude_ok & penetration_ok & clearance

    # Reward competition compares the two exact poses at the identical root
    # geometry.  The dynamic test above is the reachability gate; freezing the
    # comparison pose prevents the nominal wheel/riser collision impulse from
    # physically lifting the nominal pose before its reward is measured.
    raw.episode_length_buf[:] = 1
    if hasattr(raw, "_perceptive_z_nom"):
        del raw._perceptive_z_nom
    write_exact_pose(raw, default)
    nominal_geom, nominal_terms, nominal_net = reward_snapshot(raw)
    nominal_z_cache = raw._perceptive_z_nom.clone()
    write_exact_pose(raw, q)
    raw._perceptive_z_nom = nominal_z_cache
    raw._perceptive_spatial_step = None
    target_geom, target_terms, target_net = reward_snapshot(raw)

    for i in range(raw.num_envs):
        delta_terms = {name: float(target_terms[name][i] - nominal_terms[name][i]) for name in target_terms}
        delta_net = float(target_net[i] - nominal_net[i])
        print(
            f"COMPETITION_{args.height}CM alpha={float(alpha[i]):.2f} feasible={bool(feasible[i])} "
            f"active_nom={int(nominal_geom['active'][i].sum())} active_target={int(target_geom['active'][i].sum())} "
            f"spatial_nom={float(nominal_terms['spatial'][i]):.6f} spatial_target={float(target_terms['spatial'][i]):.6f} "
            f"progress_nom={float(nominal_geom['progress'][i,0]):.6f} progress_target={float(target_geom['progress'][i,0]):.6f} "
            f"supports={int(support[i].sum())} limits={bool(unclipped_in_limits[i])} "
            f"attitude={bool(attitude_ok[i])} penetration={bool(penetration_ok[i])} "
            f"dynamic_clearance={bool(clearance[i])} "
            f"delta_spatial={delta_terms['spatial']:.6f} delta_hipy={delta_terms['hipy']:.6f} "
            f"delta_knee={delta_terms['knee']:.6f} delta_false_lift={delta_terms['false_lift']:.6f} "
            f"DELTA_NET={delta_net:.6f}"
        )

    passing = torch.where(feasible & ((target_net - nominal_net) > 0.0))[0]
    assert passing.numel() > 0, f"{args.height} cm has no feasible clearance pose with positive DeltaNet"
    chosen = int(passing[0])
    print(
        f"PRELIFT_REWARD_COMPETITION_{args.height}=PASS alpha={float(alpha[chosen]):.2f} "
        f"DELTA_NET={float(target_net[chosen]-nominal_net[chosen]):.6f}"
    )
    env.close()


try:
    main()
finally:
    app.close()
