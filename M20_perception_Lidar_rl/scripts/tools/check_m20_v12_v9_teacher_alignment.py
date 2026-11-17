"""Check that the V12 teacher keeps the effective V9 network/training settings."""

import re
from types import SimpleNamespace

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_ppo_cfg import (
    DeeproboticsM20V12V9Teacher6DPPORunnerCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v12_teacher_env_cfg import (
    DeeproboticsM20V12V9Teacher6DEnvCfg,
)
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.v12_teacher_moe import MoeLayer

cfg = DeeproboticsM20V12V9Teacher6DPPORunnerCfg()
env_cfg = DeeproboticsM20V12V9Teacher6DEnvCfg()

assert cfg.num_steps_per_env == 24
assert cfg.max_iterations == 30_000
assert cfg.save_interval == 200
assert cfg.clip_actions == 5.0

assert cfg.policy.class_name == "V12Teacher6DMoEActorCritic"
assert cfg.policy.init_noise_std == 0.5
assert cfg.policy.noise_std_type == "log"
assert cfg.policy.actor_hidden_dims == [512, 256, 128]
assert cfg.policy.critic_hidden_dims == [512, 256, 128]
assert cfg.policy.activation == "elu"
assert cfg.policy.num_moe_experts == 4
assert cfg.policy.moe_gate_hidden_dims == [128]
assert cfg.policy.output_init_scale == 0.01

assert cfg.algorithm.num_learning_epochs == 5
assert cfg.algorithm.num_mini_batches == 8
assert cfg.algorithm.learning_rate == 3.0e-4
assert cfg.algorithm.entropy_coef == 5.0e-4
assert cfg.algorithm.desired_kl == 0.006
assert cfg.algorithm.gamma == 0.99
assert cfg.algorithm.lam == 0.95

assert env_cfg.rewards.front_action_mirror is None
assert env_cfg.rewards.rear_action_mirror is None
assert env_cfg.rewards.action_mirror.func is mdp.v12_v9_action_term_mirror
assert env_cfg.rewards.action_mirror.weight == -0.15
assert env_cfg.rewards.action_mirror.params["mirror_joint_names"] == [
    ["fl_(hipx|hipy|knee).*", "fr_(hipx|hipy|knee).*"],
    ["hl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
]
assert env_cfg.rewards.wheel_action_mirror.func is mdp.v12_v9_action_term_mirror
assert env_cfg.rewards.wheel_action_mirror.weight == -0.05

for term, joint_names in (
    (env_cfg.rewards.hipy_joint_pos_penalty, env_cfg.hipy_joint_names),
    (env_cfg.rewards.knee_joint_pos_penalty, env_cfg.knee_joint_names),
):
    assert term.func is mdp.joint_pos_penalty
    assert term.weight == -0.1
    assert term.params["asset_cfg"].joint_names == joint_names
    assert term.params["stand_still_scale"] == 5.0
    assert term.params["velocity_threshold"] == 0.5
    assert term.params["command_threshold"] == 0.1

reset = env_cfg.events.randomize_reset_base
assert reset.func is mdp.reset_root_state_uniform
assert not ({"stair_distance_range", "stair_lateral_range", "stair_yaw_jitter_range"} & reset.params.keys())
assert reset.params["pose_range"] == {
    "x": (-0.50, 0.50), "y": (-0.50, 0.50), "z": (0.0, 0.0),
    "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-3.14, 3.14),
}
assert reset.params["velocity_range"] == {
    "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05),
    "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
}

assert env_cfg.curriculum.terrain_levels.func is mdp.v12_v9_teacher_terrain_levels
assert env_cfg.curriculum.terrain_platform_success_rate is None
assert env_cfg.curriculum.terrain_levels.params == {
    "lin_vel_reward_term_name": "track_lin_vel_xy_exp",
    "ang_vel_reward_term_name": "track_ang_vel_z_exp",
    "forward_progress_reward_term_name": "world_forward_progress_score",
    "move_up_tracking_threshold": 0.50,
    "move_down_tracking_threshold": 0.32,
    "move_up_forward_progress_threshold": 0.45,
    "move_down_forward_progress_threshold": 0.20,
    "random_reset_at_max": True,
    "allow_move_down": True,
}

ascending = tuple(env_cfg.v11_perception.ascending_stair_terrain_type_ids)
stairs = tuple(env_cfg.v11_perception.stair_terrain_type_ids)
assert ascending
assert ascending == tuple(range(ascending[0], ascending[-1] + 1))
assert set(ascending).issubset(stairs)
assert env_cfg.v10_perception.single_stair_column_start == -1
assert env_cfg.v10_perception.single_stair_column_end == -1
assert env_cfg.v10_perception.full_stair_column_start == ascending[0]
assert env_cfg.v10_perception.full_stair_column_end == ascending[-1] + 1
assert env_cfg.v10_perception.num_terrain_columns == env_cfg.scene.terrain.terrain_generator.num_cols


class _FakeActionAsset:
    def __init__(self, joint_names):
        self.joint_names = joint_names

    def find_joints(self, pattern, preserve_order=True):
        matches = [(index, name) for index, name in enumerate(self.joint_names) if re.fullmatch(pattern, name)]
        return [item[0] for item in matches], [item[1] for item in matches]


leg_names = [
    f"{leg}_{joint}_joint"
    for leg in ("fl", "fr", "hl", "hr")
    for joint in ("hipx", "hipy", "knee")
]
leg_actions = torch.ones(2, len(leg_names))
leg_actions[1, leg_names.index("fr_hipy_joint")] = 2.0
fake_action_term = SimpleNamespace(
    _joint_names=leg_names,
    _asset=_FakeActionAsset(leg_names),
    raw_actions=leg_actions,
)
fake_mirror_env = SimpleNamespace(
    num_envs=2,
    device=torch.device("cpu"),
    action_manager=SimpleNamespace(get_term=lambda name: fake_action_term),
)
mirror_cost = mdp.v12_v9_action_term_mirror(
    fake_mirror_env,
    "joint_pos",
    env_cfg.rewards.action_mirror.params["mirror_joint_names"],
)
assert mirror_cost[0] == 0.0
assert mirror_cost[1] > 0.0


class _FakeTerrain:
    def __init__(self):
        self.terrain_levels = torch.zeros(4, dtype=torch.long)
        self.last_update = None

    def update_env_origins(self, ids, move_up, move_down):
        self.last_update = (ids.clone(), move_up.clone(), move_down.clone())


fake_terrain = _FakeTerrain()
reward_weights = {
    "track_lin_vel_xy_exp": 2.5,
    "track_ang_vel_z_exp": 3.0,
    "world_forward_progress_score": 2.0,
}
scores = {
    "track_lin_vel_xy_exp": torch.tensor([0.60, 0.40, 0.80, 0.80]),
    "track_ang_vel_z_exp": torch.tensor([0.60, 0.40, 0.80, 0.80]),
    "world_forward_progress_score": torch.tensor([0.50, 0.50, 0.90, 0.10]),
}
fake_curriculum_env = SimpleNamespace(
    device=torch.device("cpu"),
    common_step_counter=1,
    max_episode_length_s=20.0,
    scene=SimpleNamespace(terrain=fake_terrain),
    termination_manager=SimpleNamespace(time_outs=torch.tensor([True, True, False, True])),
    reward_manager=SimpleNamespace(
        _episode_sums={name: value * reward_weights[name] * 20.0 for name, value in scores.items()},
        get_term_cfg=lambda name: SimpleNamespace(weight=reward_weights[name]),
    ),
    # V9 promotion must not inspect or require this V11 state.
    _v11_landing={"platform_confirmed": torch.zeros(4, dtype=torch.bool)},
)
mdp.v12_v9_teacher_terrain_levels(
    fake_curriculum_env,
    torch.arange(4),
    **env_cfg.curriculum.terrain_levels.params,
)
_, move_up, move_down = fake_terrain.last_update
assert torch.equal(move_up, torch.tensor([True, False, False, False]))
assert torch.equal(move_down, torch.tensor([False, False, True, True]))

for input_dim, output_dim, hidden_dims in (
    (63, 16, cfg.policy.actor_hidden_dims),
    (568, 1, cfg.policy.critic_hidden_dims),
):
    moe = MoeLayer(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=cfg.policy.num_moe_experts,
        gate_hidden_dims=cfg.policy.moe_gate_hidden_dims,
        expert_hidden_dims=hidden_dims,
        activation=cfg.policy.activation,
    )
    output = moe(torch.zeros(2, input_dim))
    assert output.shape == (2, output_dim)
    assert len(moe.experts) == 4
    assert torch.allclose(moe.last_gate_scores.sum(dim=-1), torch.ones(2))

print("PASS: V12 network, rewards, reset, curriculum, and terrain indices match the V9 contract.", flush=True)
simulation_app.close()
