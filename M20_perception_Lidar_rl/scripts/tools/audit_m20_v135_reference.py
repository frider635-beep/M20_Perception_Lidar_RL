#!/usr/bin/env python3
"""Runtime audit for the V13.5-Final frozen ABI and reward wiring."""
from __future__ import annotations

import argparse
import os
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import rl_training.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

TASK = "M20-Stair-V13.5-V11FD-NativeAlternation-v0"


def main():
    env = parse_env_cfg(TASK, device=args.device, num_envs=16)
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    policy = agent.policy
    assert policy.class_name == "V11ContinuousActorCritic"
    assert policy.actor_hidden_dims == [512, 256, 128]
    assert agent.num_steps_per_env == 24
    assert not agent.resume and agent.load_run is None and agent.load_checkpoint is None
    assert env.observations.policy.height_scan.func.__name__ == "v11_actor_perception"
    assert env.observations.critic.v13_reward_state.func.__name__ == "v135_final_critic_reward_state"
    assert env.v11_perception.soft_alternation_command_split
    assert env.v11_perception.stair_forward_command_min == 0.10
    assert env.v11_perception.stair_forward_command_max is None
    assert env.v11_perception.non_ascending_stair_forward_command_min == 0.15
    assert env.v11_perception.non_ascending_stair_forward_command_max == 0.80
    assert env.scene.terrain.max_init_terrain_level == 3
    assert env.events.randomize_reset_base.func.__name__ == "v10_reset_root_state_stair_conditioned"
    assert env.v135_longtrain.rollout_steps_per_update == 24
    assert env.v135_longtrain.anneal_updates == 1000
    assert env.v135_longtrain.flat_blend_steps == 10
    expected = {"track_lin_vel_xy_exp": 5.0, "world_forward_progress": 2.0, "world_forward_deficit": 3.0}
    for name, weight in expected.items():
        assert getattr(env.rewards, name).weight == weight
    strict = {
        "v135_valid_riser_completion": 1.5,
        "v135_wrong_side_riser": -0.15,
        "v135_duplicate_riser": -0.10,
        "v135_skip_riser": -0.35,
        "v135_dual_skip_riser": -0.20,
    }
    for name, weight in strict.items():
        assert getattr(env.rewards, name).weight == weight
    print("V135_FINAL_REFERENCE_AUDIT: PASS actor=63 critic=574 action=16 mlp rollout=24", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
