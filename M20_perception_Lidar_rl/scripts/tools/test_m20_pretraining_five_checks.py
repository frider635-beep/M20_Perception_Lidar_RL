"""Pre-training hard checks: official terrain, frames, feasibility, yaw, contact."""

import argparse
import copy
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--suite", choices=("official", "delivery17", "delivery20"), default="official")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.perceptive_terrains import (
    M20_DELIVERY_17_20_TERRAINS_CFG,
    M20_DELIVERY_17_TERRAINS_CFG,
    M20_DELIVERY_20_TERRAINS_CFG,
    M20_DELIVERY_ACCEPTANCE_TERRAINS_CFG,
)
from rl_training.tasks.manager_based.locomotion.velocity.mdp.perceptive_spatial import (
    _semantic_body_ids,
    classify_vertical_riser_contact,
    wheel_positions_yaw,
)


def disable_randomization(cfg):
    for name in (
        "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia", "randomize_com_positions", "randomize_apply_external_force_torque",
        "randomize_reset_joints", "randomize_actuator_gains", "randomize_push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def assert_official_terrain(cfg):
    terrain = cfg.scene.terrain.terrain_generator
    official = ROUGH_TERRAINS_CFG
    assert set(terrain.sub_terrains) == {"pyramid_stairs_inv"}
    for name in terrain.sub_terrains:
        got, ref = terrain.sub_terrains[name], official.sub_terrains[name]
        assert type(got) is type(ref)
        assert got.step_height_range == ref.step_height_range == (0.05, 0.23)
        assert got.step_width == ref.step_width == 0.3
        assert got.platform_width == ref.platform_width == 3.0
        assert got.border_width == ref.border_width == 1.0
    assert terrain.sub_terrains["pyramid_stairs_inv"].proportion == 1.0
    sample = terrain.sub_terrains["pyramid_stairs_inv"]
    num_steps = int(min(
        (terrain.size[0] - 2 * sample.border_width - sample.platform_width) // (2 * sample.step_width) + 1,
        (terrain.size[1] - 2 * sample.border_width - sample.platform_width) // (2 * sample.step_width) + 1,
    ))
    assert num_steps == 6
    delivery = M20_DELIVERY_ACCEPTANCE_TERRAINS_CFG.sub_terrains["full_width_ascending_stairs"]
    assert delivery.step_depth == 0.31 and 0.17 in delivery.step_heights and 0.20 in delivery.step_heights
    exact = M20_DELIVERY_17_20_TERRAINS_CFG.sub_terrains
    assert exact["stairs_17cm_x_31cm"].step_heights == (0.17,)
    assert exact["stairs_20cm_x_31cm"].step_heights == (0.20,)
    assert all(item.step_depth == 0.31 for item in exact.values())
    print("OFFICIAL_MESH_PYRAMID_REUSE=PASS")
    print("OFFICIAL_STAIRS_NUM_STEPS=6 STEP_WIDTH=0.30 STEP_HEIGHT_RANGE=0.05:0.23")
    print("DELIVERY_17_20_X_31CM_TERRAIN=PASS")


def write_pose(raw, yaw_degrees, root_z_offset=0.0, root_x_offset=0.0, root_y_offset=0.0, *, settle=True):
    robot = raw.scene["robot"]
    state = robot.data.default_root_state.clone()
    state[:, :3] += raw.scene.env_origins
    state[:, 0] += root_x_offset
    state[:, 1] += root_y_offset
    state[:, 2] += root_z_offset
    yaw_deg = torch.as_tensor(yaw_degrees, device=raw.device, dtype=state.dtype)
    if yaw_deg.ndim == 0:
        yaw_deg = yaw_deg.repeat(raw.num_envs)
    if yaw_deg.shape != (raw.num_envs,):
        raise ValueError(f"yaw must be scalar or [{raw.num_envs}], got {tuple(yaw_deg.shape)}")
    yaw = torch.deg2rad(yaw_deg)
    state[:, 3:7] = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    if settle:
        raw.sim.step(render=False)
        raw.scene.update(dt=raw.physics_dt)
    else:
        raw.scene.update(dt=0.0)
    for sensor_name in ("height_scanner", "height_scanner_base"):
        sensor = raw.scene[sensor_name]
        # RayCaster owns an independent PhysX articulation view.  Direct test
        # teleports update the robot view immediately but this second view can
        # retain the reset transform.  Synchronize it explicitly before the
        # forced ray cast; normal environment stepping does not need this.
        if hasattr(sensor, "_view") and hasattr(sensor._view, "set_root_transforms"):
            sensor_pose_xyzw = state[:, :7].clone()
            sensor_pose_xyzw[:, 3:] = math_utils.convert_quat(sensor_pose_xyzw[:, 3:], to="xyzw")
            indices = torch.arange(raw.num_envs, dtype=torch.int32, device=raw.device)
            sensor._view.set_root_transforms(sensor_pose_xyzw, indices=indices)
        sensor.reset()
        sensor.update(raw.physics_dt, force_recompute=True)
    raw.common_step_counter += 1
    raw._perceptive_spatial_step = None


def coordinate_frame_check(raw):
    # Use two actual terrain rows with different world-Z origins.  This is a
    # runtime check, not a single-origin algebra test.
    terrain = raw.scene.terrain
    levels = torch.tensor([0, 7, 0, 7, 0], device=raw.device)
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
    write_pose(raw, 0.0)
    s = mdp.perceptive_spatial_state(raw)
    world_progress = ((s["z_actual_world"] - s["z_nominal_world"]) /
                      (s["z_ref_world"] - s["z_nominal_world"]).clamp_min(1e-6)).clamp(0, 1)
    relative_progress = ((s["z_actual_relative"] - raw.cfg.spatial_guidance.wheel_collision_radius) /
                         (s["z_ref_relative"] - raw.cfg.spatial_guidance.wheel_collision_radius).clamp_min(1e-6)).clamp(0, 1)
    assert torch.allclose(s["z_actual_relative"], s["z_actual_world"] - s["z_lower"], atol=1e-6)
    assert torch.allclose(s["z_ref_relative"], s["z_ref_world"] - s["z_lower"], atol=1e-6)
    assert torch.allclose(s["progress"], world_progress, atol=1e-6)
    assert torch.allclose(s["progress"], relative_progress, atol=1e-6)
    assert torch.unique(raw.scene.env_origins[:, 2]).numel() > 1
    print("WORLD_RELATIVE_COORDINATE_INVARIANCE=PASS runtime_env_origin_z=" + repr(raw.scene.env_origins[:, 2].tolist()))


def yaw_check(raw, expected_height_m):
    terrain = raw.scene.terrain
    if raw.num_envs != 5:
        raise ValueError("yaw acceptance requires five envs for five yaw values")
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    yaw_values = torch.tensor([0.0, 5.0, -5.0, 10.0, -10.0], device=raw.device)
    # Advance one physics step so the RayCaster's independent PhysX
    # articulation view observes the manually written root transform.  A
    # zero-step scene update leaves that view at the randomized reset pose.
    write_pose(raw, yaw_values, root_x_offset=0.60, settle=True)
    state = mdp.perceptive_spatial_state(raw)
    robot = raw.scene["robot"]
    wheel_ids = _semantic_body_ids(robot)
    wheel_local = wheel_positions_yaw(robot, wheel_ids)
    sensor = raw.scene["height_scanner"]
    hit_delta = sensor.data.ray_hits_w - robot.data.root_pos_w[:, None, :]
    quat = robot.data.root_quat_w[:, None, :].expand(-1, hit_delta.shape[1], -1).reshape(-1, 4)
    hit_local = math_utils.quat_apply_inverse(quat, hit_delta.reshape(-1, 3)).reshape_as(hit_delta)
    hit_local_sem = hit_local.reshape(raw.num_envs, 15, 21, 3).transpose(1, 2).flip(dims=(1, 2))
    for env_index in range(raw.num_envs):
        height_cm = int(round(expected_height_m * 100))
        print(
            f"YAW_DIAG terrain={height_cm} yaw={yaw_values[env_index].item():+.0f} "
            f"root_rel={(robot.data.root_pos_w[env_index] - raw.scene.env_origins[env_index]).tolist()} "
            f"base_link_rel={(robot.data.body_pos_w[env_index, 0] - raw.scene.env_origins[env_index]).tolist()} "
            f"sensor_rel={(sensor.data.pos_w[env_index] - robot.data.root_pos_w[env_index]).tolist()} "
            f"drift={sensor.drift[env_index].tolist()} ray_cast_drift={sensor.ray_cast_drift[env_index].tolist()} "
            f"env_origin={raw.scene.env_origins[env_index].tolist()} "
            f"hit_x_center={hit_local_sem[env_index, :, 7, 0].tolist()} "
            f"hit_z_center={hit_local_sem[env_index, :, 7, 2].tolist()} "
            f"wheel_local={wheel_local[env_index].tolist()} "
            f"selected_x={state['riser_x'][env_index].tolist()} "
            f"height={state['riser_height'][env_index].tolist()} "
            f"distance={state['riser_distance'][env_index].tolist()} "
            f"valid={state['valid'][env_index].tolist()} active={state['active'][env_index].tolist()} "
            f"z_ref_relative={state['z_ref_relative'][env_index].tolist()}"
        )
        elevated = hit_local_sem[env_index, :, :, 2] > (robot.data.root_pos_w[env_index, 2] - 0.50)
        if elevated.any():
            ij = torch.where(elevated)
            print(
                f"YAW_ELEVATED_COUNT={int(elevated.sum())} "
                f"x_range=({float(hit_local_sem[env_index, :, :, 0][elevated].min()):.4f},"
                f"{float(hit_local_sem[env_index, :, :, 0][elevated].max()):.4f}) "
                f"y_range=({float(hit_local_sem[env_index, :, :, 1][elevated].min()):.4f},"
                f"{float(hit_local_sem[env_index, :, :, 1][elevated].max()):.4f})"
            )
        else:
            print("YAW_ELEVATED_COUNT=0")
    # Analytic yaw-frame intersection of the world-X first-riser plane with
    # each wheel's local lateral coordinate: x = (x_riser-x_base)/cos(yaw) + y*tan(yaw).
    yaw_rad = torch.deg2rad(yaw_values)
    x_world_delta = torch.full_like(yaw_rad, 0.50)  # first riser 1.10 - root offset 0.60
    expected_x = x_world_delta[:, None] / torch.cos(yaw_rad)[:, None] + wheel_local[..., 1] * torch.tan(yaw_rad)[:, None]
    expected_d = expected_x - wheel_local[..., 0]
    distance_error = torch.abs(state["riser_distance"] - expected_d)
    x_error = torch.abs(state["riser_x"] - expected_x)
    assert state["valid"].all(), state["valid"]
    assert distance_error.max() <= 0.075, distance_error
    assert x_error.max() <= 0.075, x_error
    expected_height = torch.full_like(state["riser_height"], expected_height_m)
    assert torch.max(torch.abs(state["riser_height"] - expected_height)) <= 0.015
    expected_active = (expected_d < raw.cfg.spatial_guidance.d_start) & (expected_d > raw.cfg.spatial_guidance.d_end)
    assert torch.equal(state["active"], expected_active)
    assert torch.isfinite(state["z_ref_relative"]).all()
    print(f"YAW_FRAME_MAX_X_ERROR={float(x_error.max()):.6f}")
    print(f"YAW_FRAME_MAX_D_ERROR={float(distance_error.max()):.6f}")
    print("YAW_ALIGNED_RISER_GEOMETRY=PASS yaw=0,+-5,+-10")


def whole_body_feasibility(raw, expected_height_m):
    robot = raw.scene["robot"]
    body_ids = _semantic_body_ids(robot)
    leg_ids, _ = robot.find_joints(raw.cfg.leg_joint_names, preserve_order=True)
    default = robot.data.default_joint_pos.clone()
    terrain = raw.scene.terrain
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    # Put the robot immediately before the first delivery riser.
    write_pose(raw, 0.0, root_x_offset=0.72)
    alpha_candidates = torch.linspace(0.15, 0.55, raw.num_envs, device=raw.device)
    for height in (expected_height_m,):
        q = default.clone()
        # Lift FL only. FR/HL/HR remain at the official default pose and must
        # form the requested three-wheel support set.
        q[:, leg_ids[1]] -= alpha_candidates * 2.0
        q[:, leg_ids[2]] -= alpha_candidates * 1.2
        limits = robot.data.soft_joint_pos_limits
        in_limits = ((q >= limits[..., 0]) & (q <= limits[..., 1])).all()
        q = torch.maximum(torch.minimum(q, limits[..., 1]), limits[..., 0])
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        robot.set_joint_velocity_target(torch.zeros_like(q))
        raw.scene.write_data_to_sim()
        for _ in range(int(0.5 / raw.physics_dt)):
            raw.sim.step(render=False)
            raw.scene.update(dt=raw.physics_dt)
        wheels = robot.data.body_pos_w[:, body_ids]
        support_z = wheels[:, :, 2] - raw.scene.env_origins[:, None, 2]
        base_clear = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        gravity = robot.data.projected_gravity_b
        roll_pitch_ok = gravity[:, :2].abs().amax(dim=1) < 0.55
        sensor = raw.scene["contact_forces"]
        names = list(sensor.body_names)
        wheel_sensor_ids = torch.tensor([names.index(n) for n in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")], device=raw.device)
        forces = sensor.data.net_forces_w[:, wheel_sensor_ids].norm(dim=-1)
        support_contacts = forces[:, 1:] > 1.0
        all_body_z = robot.data.body_pos_w[:, :, 2] - raw.scene.env_origins[:, None, 2]
        nonwheel = torch.ones(all_body_z.shape[1], dtype=torch.bool, device=raw.device)
        nonwheel[body_ids] = False
        no_link_penetration = all_body_z[:, nonwheel].min() > 0.015
        support_ok = support_contacts.sum(1) == 3
        base_ok = base_clear > 0.05
        geom = mdp.perceptive_spatial_state(raw)
        fl_clearance = geom["progress"][:, 0] >= 0.999
        selected = in_limits & support_ok & base_ok & roll_pitch_ok & fl_clearance
        print(
            f"WHOLE_BODY_DIAG_{int(height*100)}CM in_limits={bool(in_limits)} "
            f"support_contacts={support_contacts.tolist()} force={forces.tolist()} "
            f"alpha={alpha_candidates.tolist()} support_ok={support_ok.tolist()} "
            f"no_link_penetration={bool(no_link_penetration)} roll_pitch_ok={roll_pitch_ok.tolist()} "
            f"base_ok={base_ok.tolist()} fl_clearance={fl_clearance.tolist()} "
            f"base_clear={base_clear.tolist()} support_z={support_z.tolist()}"
        )
        assert bool(in_limits) and bool(no_link_penetration) and selected.any()
        chosen = int(torch.where(selected)[0][0])
        print(
            f"WHOLE_BODY_FEASIBILITY_{int(height*100)}=PASS alpha={float(alpha_candidates[chosen]):.6f} "
            f"support_contacts={int(support_contacts[chosen].sum())} progress={float(geom['progress'][chosen,0]):.6f} "
            f"nonwheel_min_clear={float(all_body_z[chosen, nonwheel].min()):.6f} "
            f"base_clear={float(base_clear[chosen]):.6f}"
        )


def contact_classifier_check(raw, expected_height_m):
    """Run three actual contact-sensor scenes on fixed 17 cm delivery stairs."""
    robot = raw.scene["robot"]
    sensor = raw.scene["contact_forces"]
    names = list(sensor.body_names)
    wheel_sensor_ids = torch.tensor([names.index(n) for n in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")], device=raw.device)
    wheel_body_ids = _semantic_body_ids(robot)
    terrain = raw.scene.terrain
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    riser_x = raw.scene.env_origins[:, 0] + 1.10
    lower = raw.scene.env_origins[:, 2]
    upper = lower + expected_height_m
    radius = raw.cfg.spatial_guidance.wheel_collision_radius

    def scene_result(x_center, z_center, *, vx=0.0, vz=0.0):
        # Restore the official default joint pose first.  The previous
        # whole-body acceptance leaves FL lifted; using that stale wheel
        # offset would place the new contact scene at the wrong height/x.
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.default_joint_vel))
        raw.scene.write_data_to_sim(); raw.sim.forward(); raw.scene.update(dt=0.0)
        state = robot.data.default_root_state.clone(); state[:, :3] += raw.scene.env_origins
        front_rel = wheel_positions_yaw(robot, wheel_body_ids)[:, 0]
        state[:, 0] = x_center - front_rel[:, 0]
        state[:, 2] = z_center - front_rel[:, 2]
        state[:, 7:] = 0.0
        state[:, 7] = vx
        state[:, 9] = vz
        robot.write_root_state_to_sim(state)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.default_joint_vel))
        raw.scene.write_data_to_sim(); raw.sim.forward(); raw.scene.update(dt=0.0)
        hit = torch.zeros((raw.num_envs, 4), dtype=torch.bool, device=raw.device)
        force_peak = torch.zeros((raw.num_envs, 4, 3), device=raw.device)
        for _ in range(40):
            raw.scene.write_data_to_sim(); raw.sim.step(render=False); raw.scene.update(dt=raw.physics_dt)
            force = sensor.data.net_forces_w[:, wheel_sensor_ids]
            force_peak = torch.where(force.norm(dim=-1, keepdim=True) > force_peak.norm(dim=-1, keepdim=True), force, force_peak)
            xy = robot.data.body_pos_w[:, wheel_body_ids, :2]
            z = robot.data.body_pos_w[:, wheel_body_ids, 2]
            rx = torch.full_like(z, 1.10) + raw.scene.env_origins[:, 0:1]
            zl = raw.scene.env_origins[:, 2:3].expand_as(z)
            zu = zl + expected_height_m
            hit |= classify_vertical_riser_contact(xy, z, force, rx, zl, zu, wheel_radius=radius)
        return hit, force_peak

    # Contact-sensor force is reported after dynamics has reacted; place the
    # wheel just inside the face and allow a short driven impact window.
    a, fa = scene_result(riser_x - radius + 0.005, lower + radius, vx=1.0)
    b, fb = scene_result(riser_x + 0.16, upper + radius + 0.08, vz=-0.5)
    c, fc = scene_result(riser_x + 0.30, upper + radius + 0.08, vx=1.0)
    print(f"CONTACT_SCENE_A_VERTICAL_FACE result={a[:,0].tolist()} force={fa[:,0].tolist()}")
    print(f"CONTACT_SCENE_B_UPPER_TREAD result={b[:,0].tolist()} force={fb[:,0].tolist()}")
    print(f"CONTACT_SCENE_C_CROSSING result={c[:,0].tolist()} force={fc[:,0].tolist()}")
    assert a[:, 0].any(), "scene A did not produce BAD vertical-riser contact"
    assert not b[:, 0].any(), "scene B upper tread was misclassified as riser contact"
    assert not c[:, 0].any(), "scene C crossing was misclassified as riser contact"
    print("VERTICAL_RISER_CONTACT_A=PASS True")
    print("UPPER_TREAD_CONTACT_B=PASS False")
    print("CORRECT_CROSSING_C=PASS False")
    print("VERTICAL_RISER_CONTACT_CLASSIFIER=PASS")


def main():
    if args.suite in ("delivery17", "delivery20"):
        expected_height_m = 0.17 if args.suite == "delivery17" else 0.20
        delivery_cfg = M20_DELIVERY_17_TERRAINS_CFG if args.suite == "delivery17" else M20_DELIVERY_20_TERRAINS_CFG
        cfg = parse_env_cfg("M20-Stair-Perceptive-v0", device=args.device, num_envs=5)
        disable_randomization(cfg)
        cfg.curriculum.terrain_levels = None
        cfg.scene.terrain.terrain_generator = copy.deepcopy(delivery_cfg)
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.commands.base_velocity.debug_vis = False
        cfg.seed = 42
        env = gym.make("M20-Stair-Perceptive-v0", cfg=cfg)
        env.reset()
        raw = env.unwrapped
        yaw_check(raw, expected_height_m)
        whole_body_feasibility(raw, expected_height_m)
        contact_classifier_check(raw, expected_height_m)
        print(f"PRETRAINING_DELIVERY_{int(expected_height_m*100)}CM_CHECKS=PASS")
        env.close()
        return

    cfg = parse_env_cfg("M20-Stair-Perceptive-v0", device=args.device, num_envs=5)
    disable_randomization(cfg)
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.max_init_terrain_level = 9
    cfg.commands.base_velocity.debug_vis = False
    assert_official_terrain(cfg)
    env = gym.make("M20-Stair-Perceptive-v0", cfg=cfg)
    env.reset()
    raw = env.unwrapped
    levels = torch.tensor((0, 2, 4, 7, 9), device=raw.device)
    raw.scene.terrain.terrain_levels[:] = levels
    raw.scene.terrain.env_origins[:] = raw.scene.terrain.terrain_origins[levels, raw.scene.terrain.terrain_types]
    coordinate_frame_check(raw)
    print("PRETRAINING_OFFICIAL_CHECKS=PASS")
    env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
