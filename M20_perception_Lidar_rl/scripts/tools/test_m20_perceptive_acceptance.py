"""Isaac integration acceptance and 64-env finite-value smoke."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _canonical(value):
    """Turn manager config values into stable, recursively comparable data."""
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_canonical(item) for item in value)
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    return value


def main():
    print("STAGE=parse_config", flush=True)
    cfg = parse_env_cfg("M20-Stair-Perceptive-v0", device=args.device, num_envs=args.num_envs)
    rough_cfg = parse_env_cfg("Rough-Deeprobotics-M20-v0", device=args.device, num_envs=args.num_envs)
    # Official-locked M20 controller/action parameters must remain identical.
    assert cfg.leg_joint_names == rough_cfg.leg_joint_names
    assert cfg.wheel_joint_names == rough_cfg.wheel_joint_names
    assert cfg.actions.joint_pos.scale == rough_cfg.actions.joint_pos.scale == {
        ".*_hipx_joint": 0.125, "^(?!.*_hipx_joint).*": 0.25
    }
    assert cfg.actions.joint_vel.scale == rough_cfg.actions.joint_vel.scale == 5.0
    official_joint = cfg.scene.robot.actuators["joint"]
    official_wheel = cfg.scene.robot.actuators["wheel"]
    reference_joint = rough_cfg.scene.robot.actuators["joint"]
    reference_wheel = rough_cfg.scene.robot.actuators["wheel"]
    assert _canonical(official_joint) == _canonical(reference_joint)
    assert _canonical(official_wheel) == _canonical(reference_wheel)
    assert (official_joint.stiffness, official_joint.damping) == (80.0, 2.0)
    assert (official_wheel.stiffness, official_wheel.damping) == (0.0, 0.6)
    assert (official_joint.effort_limit, official_joint.velocity_limit) == (76.4, 22.4)
    assert (official_wheel.effort_limit, official_wheel.velocity_limit) == (21.6, 79.3)
    locked_reward_count = 0
    for name in rough_cfg.rewards.__dataclass_fields__:
        official = getattr(rough_cfg.rewards, name, None)
        current = getattr(cfg.rewards, name, None)
        if official is None:
            assert current is None, name
        else:
            assert current is not None, name
            assert current.weight == official.weight, f"{name}.weight"
            assert current.func == official.func, f"{name}.func"
            assert _canonical(current.params) == _canonical(official.params), f"{name}.params"
            locked_reward_count += 1
    assert cfg.rewards.spatial_lift is not None and cfg.rewards.false_lift is not None
    for name in ("hipy_joint_pos_penalty", "knee_joint_pos_penalty"):
        params = getattr(cfg.rewards, name).params
        assert params["terrain_height_threshold"] == 0.06, name
        assert params["high_terrain_penalty_scale"] == 0.1, name
        assert params["sensor_cfg"].name == "height_scanner_base", name
    print("OFFICIAL_FROZEN_HIGH_TERRAIN_RELAXATION=threshold:0.06,scale:0.1 PASS", flush=True)
    print(f"OFFICIAL_LOCKED_REWARDS={locked_reward_count} FULL_PARAMS=PASS", flush=True)
    print("TASK_CALIBRATED_REWARDS=spatial_lift,false_lift", flush=True)
    print("OFFICIAL_LOCKED_PARAMS=PASS", flush=True)
    print("STAGE=create_env", flush=True)
    env = gym.make("M20-Stair-Perceptive-v0", cfg=cfg)
    print("STAGE=reset_env", flush=True)
    obs, _ = env.reset()
    print("STAGE=inspect_interface", flush=True)
    raw = env.unwrapped
    dimensions = (
        raw.observation_manager.group_obs_dim["policy"][0],
        raw.observation_manager.group_obs_dim["critic"][0],
        raw.action_manager.total_action_dim,
    )
    print(f"POLICY_DIM={dimensions[0]} CRITIC_DIM={dimensions[1]} ACTION_DIM={dimensions[2]}")
    assert dimensions == (372, 247, 16), dimensions
    assert raw.observation_manager.active_terms["critic"] == [
        "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
        "joint_pos", "joint_vel", "actions", "height_scan",
    ]
    assert raw.observation_manager._group_obs_term_dim["critic"] == [
        (3,), (3,), (3,), (3,), (16,), (16,), (16,), (187,),
    ]
    print("OFFICIAL_ACTOR_DIM=57 OFFICIAL_CRITIC_DIM=247")
    print("NEW_ACTOR_DIM=372 NEW_CRITIC_DIM=247 ACTION_DIM=16")
    print("CRITIC_COMPOSITION=base_lin_vel:3,base_ang_vel:3,projected_gravity:3,velocity_commands:3,joint_pos:16,joint_vel:16,actions:16,height_scan:187")
    expected_terms = [
        "base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions", "height_scan"
    ]
    assert raw.observation_manager.active_terms["policy"] == expected_terms
    assert not any("contact" in term.lower() for term in expected_terms)
    print("ACTOR_TERMS=" + ",".join(expected_terms))

    expected_action_order = raw.cfg.leg_joint_names + raw.cfg.wheel_joint_names
    actual_order = []
    for term in raw.action_manager._terms.values():
        actual_order.extend(term._joint_names)
    assert actual_order == expected_action_order, (actual_order, expected_action_order)
    print("ACTION_ORDER=" + ",".join(actual_order))

    action = torch.zeros((args.num_envs, 16), device=raw.device)
    # Known 16-D action test: only FL knee (index 2) is non-zero.
    action[:, 2] = 0.25
    raw.action_manager.process_action(action)
    position_term = raw.action_manager.get_term("joint_pos")
    changed = (position_term.processed_actions - position_term._offset).abs() > 1e-8
    assert changed[:, 2].all() and changed.sum(dim=1).eq(1).all(), changed
    assert position_term._joint_names[2] == "fl_knee_joint"
    print("ACTION_TARGET_TEST=only_fl_knee PASS")
    for _ in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
        state = raw.perceptive_spatial_state
        assert torch.isfinite(state["spatial_raw"]).all()
        assert torch.isfinite(state["actual_lift"]).all()
    print(f"SMOKE_ENVS={args.num_envs} STEPS={args.steps} NO_NAN=True NO_INF=True")
    print(f"GUIDE_ACTIVE_FRACTION={float(state['active'].float().mean()):.6f}")
    print(f"SPATIAL_RAW_MEAN={float(state['spatial_raw'].mean()):.6f}")
    print(f"FALSE_LIFT_COST_MEAN={float(state['false_cost'].mean()):.6f}")
    print(f"CLAMP_WARNING_COUNT={int(state['clamp_warning'].sum())}")

    task_dir = Path(__file__).resolve().parents[2] / "source" / "rl_training" / "rl_training" / "tasks" / "manager_based" / "locomotion" / "velocity" / "config" / "wheeled" / "deeprobotics_m20"
    new_files = [task_dir / "perceptive_env_cfg.py", task_dir / "perceptive_terrains.py"]
    forbidden = ("SWING" + "_ACTIVE", "AR" + "MED", "NORMAL" + "_STATE")
    lowered = "\n".join(path.read_text() for path in new_files)
    assert not any(token.lower() in lowered.lower() for token in forbidden)
    print("NO_EXPLICIT_GAIT_STATE=PASS")
    env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
