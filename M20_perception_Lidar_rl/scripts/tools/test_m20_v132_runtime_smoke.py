#!/usr/bin/env python3
"""Small Isaac runtime smoke for V13.2 (no learning)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_ppo_cfg import (
    DeeproboticsM20V13V11FDAlternationPPORunnerCfg,
    DeeproboticsM20V132V11FDBarrierAlternationPPORunnerCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v13_v11fd_alternation_env_cfg import (
    DeeproboticsM20V13V11FDAlternationEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v132_v11fd_barrier_alternation_env_cfg import (
    DeeproboticsM20V132V11FDBarrierAlternationEnvCfg,
)


TASK = "M20-Stair-V13.2-V11FD-BarrierAlternation-v0"
V13_STYLE_TERMS = {
    "v13_opposite_progress", "v13_velocity_coupled_opposite_progress",
    "v13_same_side_progress", "v13_simultaneous_progress",
    "v13_alternating_completion", "v13_same_side_completion",
}


def assert_declared_config_delta() -> None:
    """Prove that V13.2 changes only its declared observation/style/runner ABI."""
    v13 = DeeproboticsM20V13V11FDAlternationEnvCfg().to_dict()
    v132 = DeeproboticsM20V132V11FDBarrierAlternationEnvCfg().to_dict()
    for section in ("scene", "actions", "commands", "events", "terminations", "v11_perception", "v13_alternation"):
        assert v132[section] == v13[section], f"undeclared V13.2 drift in {section}"

    obs13, obs132 = v13["observations"], v132["observations"]
    assert obs132["critic"] == obs13["critic"]
    policy132 = dict(obs132["policy"])
    phase_term = policy132.pop("v132_event_phase")
    assert policy132 == obs13["policy"]
    assert str(phase_term["func"]).rsplit(":", 1)[-1] == "v132_event_phase_observation"

    reward13, reward132 = dict(v13["rewards"]), dict(v132["rewards"])
    style = reward132.pop("v132_style")
    assert str(style["func"]).rsplit(":", 1)[-1] == "v132_style_objective"
    assert float(style["weight"]) == 1.0
    for name in V13_STYLE_TERMS:
        assert reward132.pop(name) is None
        reward13.pop(name)
    assert reward132 == reward13

    curriculum13, curriculum132 = dict(v13["curriculum"]), dict(v132["curriculum"])
    for name in tuple(curriculum132):
        if name.startswith("v132_"):
            curriculum132.pop(name)
    for name in tuple(curriculum13):
        if name.startswith("v13_"):
            curriculum13.pop(name)
            assert curriculum132.pop(name) is None
    assert curriculum132 == curriculum13

    agent13 = DeeproboticsM20V13V11FDAlternationPPORunnerCfg().to_dict()
    agent132 = DeeproboticsM20V132V11FDBarrierAlternationPPORunnerCfg().to_dict()
    assert agent132["algorithm"] == agent13["algorithm"]
    policy132_agent = dict(agent132["policy"])
    policy13_agent = dict(agent13["policy"])
    assert policy132_agent.pop("class_name") == "V132BarrierActorCritic"
    policy13_agent.pop("class_name")
    assert policy132_agent == policy13_agent
    for name in ("num_steps_per_env", "max_iterations", "save_interval", "resume", "load_run", "load_checkpoint"):
        assert agent132[name] == agent13[name]


def main() -> None:
    assert_declared_config_delta()
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env = gym.make(TASK, cfg=cfg)
    obs, _ = env.reset()
    raw = env.unwrapped
    assert raw.action_manager.total_action_dim == 16
    assert obs["policy"].shape == (args.num_envs, 68)
    style_cfg = raw.reward_manager.get_term_cfg("v132_style")
    assert style_cfg.func is mdp.v132_style_objective and float(style_cfg.weight) == 1.0

    actions = torch.zeros((args.num_envs, 16), device=raw.device)
    for _ in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(actions)
        reward_state = mdp.v132_alternation_state(raw)
        deploy_state = mdp.v132_deployable_phase_state(raw)
        deploy_obs = mdp.v132_event_phase_observation(raw)
        style = mdp.v132_style_objective(raw) * raw.step_dt
        assert obs["policy"].shape == (args.num_envs, 68)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(reward).all() and torch.isfinite(style).all()
        torch.testing.assert_close(obs["policy"][:, -5:], deploy_obs)
        assert reward_state is not deploy_state
        aligned = reward_state["deploy_target_agreement"]
        assert (
            reward_state["expected_side"][aligned]
            == deploy_state["settlement_expected_side"][aligned]
        ).all()
        assert torch.isin(
            deploy_state["phase"],
            torch.tensor(
                [mdp.APPROACH_B1, mdp.ALTERNATING_RISERS,
                 mdp.TOP_PLATFORM_FOLLOW, mdp.PLATEAU_ROLL],
                device=raw.device,
            ),
        ).all()
        assert torch.isin(
            deploy_state["expected_side"], torch.tensor([-1, 0, 1], device=raw.device),
        ).all()
        assert not (terminated & truncated).any()

    # Isaac computes terminal reward, resets selected environments, and then
    # recomputes observations without incrementing common_step_counter. Force
    # that exact ordering and prove the next episode cannot receive stale 5-D
    # parity/phase state from the previous episode.
    deploy_state = mdp.v132_deployable_phase_state(raw)
    deploy_state["phase"][0] = mdp.PLATEAU_ROLL
    deploy_state["expected_side"][0] = 1
    reset_id = torch.tensor([0], device=raw.device, dtype=torch.long)
    raw._reset_idx(reset_id)
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    reset_obs = raw.observation_manager.compute()["policy"][0, -5:]
    torch.testing.assert_close(
        reset_obs, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device=raw.device),
    )

    print(
        f"V132_RUNTIME_STEP_SMOKE: PASS envs={args.num_envs} steps={args.steps} "
        "actor_dim=68 action_dim=16 style_term=live",
        flush=True,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
