"""Measure nominal M20 wheel centres and mathematical shortening bounds."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, default=17)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.mdp.perceptive_spatial import _semantic_body_ids


def settle(raw, physics_steps=4):
    raw.scene.write_data_to_sim()
    for _ in range(physics_steps):
        raw.sim.step(render=False)
        raw.scene.update(dt=raw.physics_dt)


def main():
    print("STAGE=parse_config", flush=True)
    cfg = parse_env_cfg("M20-Stair-Perceptive-v0", device=args.device, num_envs=1)
    cfg.scene.terrain.max_init_terrain_level = 0
    print("STAGE=create_env", flush=True)
    env = gym.make("M20-Stair-Perceptive-v0", cfg=cfg)
    print("STAGE=reset_env", flush=True)
    env.reset()
    print("STAGE=resolve_workspace", flush=True)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    body_ids = _semantic_body_ids(robot)
    wheel0 = mdp.wheel_positions_base(robot, body_ids)
    z_nom = wheel0[0, :, 2].clone()
    print("WHEEL_ORDER=FL,FR,HL,HR")
    print("Z_NOM_BASE_M=" + ",".join(f"{v:.6f}" for v in z_nom.tolist()))

    leg_names = raw.cfg.leg_joint_names
    leg_ids, resolved = robot.find_joints(leg_names, preserve_order=True)
    assert resolved == leg_names, (resolved, leg_names)
    limits = robot.data.soft_joint_pos_limits[0, leg_ids]
    default = robot.data.default_joint_pos.clone()
    best = torch.zeros(4, device=raw.device)
    best_q = default[0, leg_ids].reshape(4, 3).clone()
    # Scan each leg independently over simulator soft limits. This is a numeric
    # reachability search, not a physical-safe or deployment-safe workspace.
    for leg in range(4):
        hipy_axis = torch.linspace(limits[3 * leg + 1, 0], limits[3 * leg + 1, 1], args.samples, device=raw.device)
        knee_axis = torch.linspace(limits[3 * leg + 2, 0], limits[3 * leg + 2, 1], args.samples, device=raw.device)
        for hipy in hipy_axis:
            for knee in knee_axis:
                q = default.clone()
                q[0, leg_ids[3 * leg + 1]] = hipy
                q[0, leg_ids[3 * leg + 2]] = knee
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                settle(raw, physics_steps=1)
                z = mdp.wheel_positions_base(robot, body_ids)[0, leg, 2]
                lift = z - z_nom[leg]
                if lift > best[leg]:
                    best[leg] = lift
                    best_q[leg] = q[0, leg_ids[3 * leg:3 * leg + 3]]
    print("WORKSPACE_MATH_RANGE_M=" + ",".join(f"{v:.6f}" for v in best.tolist()))
    print("WORKSPACE_NUMERIC_BOUND_0P85_M=" + ",".join(f"{0.85 * v:.6f}" for v in best.tolist()))
    print("WORKSPACE_NUMERIC_BOUND_NOT_PHYSICAL_SAFE=True")
    print("BEST_Q_HIPX_HIPY_KNEE=" + repr(best_q.tolist()))
    print("SCAN_RANGE=OFFICIAL_SOFT_JOINT_LIMITS")

    # Sign test: move FL knee toward the best shortening pose.
    q = default.clone()
    q[0, leg_ids[2]] = best_q[0, 2]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    settle(raw)
    fl_lift = mdp.wheel_positions_base(robot, body_ids)[0, 0, 2] - z_nom[0]
    print(f"FL_KNEE_SIGN_TEST_LIFT_M={float(fl_lift):.6f}")
    assert fl_lift > 0.0, "FL knee shortening must increase base-frame actual lift"
    print("WORKSPACE_MATHEMATICAL_SCAN=PASS")
    env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
