#!/usr/bin/env python3
"""CPU/static contracts for V13.2 barrier alternation and dual returns."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v132_barrier_alternation.py"
V13_MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v13_alternation.py"
CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v132_v11fd_barrier_alternation_env_cfg.py"
V13_CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v13_v11fd_alternation_env_cfg.py"
MDP_INIT = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/__init__.py"
REGISTRY = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py"
RUNNER_CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py"
ALGORITHM = ROOT / "source/rl_training/rl_training/v132_dual_critic.py"
RUNNER = ROOT / "source/rl_training/rl_training/v132_dual_critic_runner.py"
LAUNCH = ROOT / "scripts/train_m20_v132_v11fd_barrier_alternation_from_scratch.sh"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mdp = load(MDP, "v132_contract_mdp")
algorithm = load(ALGORITHM, "v132_contract_algorithm")

# Relaxed barriers are finite in violation, maximal at the band centre, and
# preserve gradients on both sides instead of becoming a binary gate.
values = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5], requires_grad=True)
score = mdp.v132_barrier_band_score(values, 0.0, 1.0, 0.05)
assert torch.isfinite(score).all()
assert score[2] == score.max() and score[2] > score[0] and score[2] > score[-1]
score.sum().backward()
assert torch.isfinite(values.grad).all()

# B1 is never silently renamed after a skipped 30-cm face.
reference = torch.tensor([[1.0, 0.0], [1.3, 0.0], [1.6, 0.0]])
b1 = torch.tensor([[1.0, 0.0]]).expand_as(reference)
axis = torch.tensor([[1.0, 0.0]]).expand_as(reference)
index = mdp.v132_target_index(reference, b1, axis, torch.ones(3, dtype=torch.bool), step_width=0.30)
assert index.tolist() == [1, 2, 3]
choice = mdp._v132_choose_deploy_target(
    positive=torch.tensor([[True, True], [True, True]]),
    world_xy=torch.tensor([[[1.3, 0.0], [1.6, 0.0]], [[1.6, 0.0], [1.9, 0.0]]]),
    upper_z=torch.tensor([[0.2, 0.3], [0.3, 0.4]]),
    step_height=torch.full((2, 2), 0.1), distance=torch.tensor([[0.3, 0.6], [0.6, 0.9]]),
    b1_world_xy=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    forward_world_xy=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    required_index=torch.tensor([2, 2]), b1_valid=torch.tensor([True, True]),
    step_width=0.30,
)
assert choice["index"].tolist() == [2, 3]  # a missing B2 is preserved as a B3 skip

# Either mirror seed is legal; thereafter target parity is fixed for arbitrary
# stair count, including resynchronisation at B3 after a missed B2.
target = torch.tensor([1, 2, 3, 4])
for seed, expected in ((0, [0, 1, 0, 1]), (1, [1, 0, 1, 0])):
    got = mdp.v132_expected_side(
        torch.full((4,), seed), target, torch.ones(4, dtype=torch.bool),
    )
    assert got.tolist() == expected

latched = mdp.v132_latch_first_lift(
    torch.tensor([-1, -1, -1]), torch.tensor([False, False, False]),
    torch.tensor([[True, False], [False, True], [True, True]]),
)
assert latched["seed_side"].tolist() == [0, 1, -1]
assert latched["unique_event"].tolist() == [True, True, False]
assert latched["simultaneous_event"].tolist() == [False, False, True]

# Reset detection covers both Isaac's same-common-step post-reset observation
# and a one-action episode whose next reward also has length one.
current = torch.tensor([8, 0, 1])
previous = torch.tensor([7, 8, 1])
assert mdp.v132_episode_reset_mask(current, previous, same_step=True).tolist() == [False, True, False]
assert mdp.v132_episode_reset_mask(current, previous, same_step=False).tolist() == [False, True, True]

# Deployment parity onset mirrors V11's forward-qualified normalized
# clearance: height without forward progress is neutral, while either unique
# moving side can seed B1 without contact-force input.
clearance = mdp.v132_deploy_clearance_onset(
    max_progress=torch.zeros((3, 2)),
    previous_forward=torch.full((3, 2), -0.30),
    previous_valid=torch.ones((3, 2), dtype=torch.bool),
    wheel_forward=torch.tensor([[-0.29, -0.29], [-0.29, -0.29], [-0.30, -0.30]]),
    wheel_bottom_z=torch.tensor([[0.031, 0.01], [0.01, 0.031], [0.10, 0.10]]),
    target_lower_z=torch.zeros(3), step_height=torch.full((3,), 0.10),
    target_valid=torch.ones(3, dtype=torch.bool),
    clearance_margin=0.05, onset_progress=0.20,
)
assert clearance["lift_ready"].tolist() == [[True, False], [False, True], [False, False]]

# Positive platform evidence requires a real same-height patch with longitudinal
# extent. Merely seeing no riser/receiving no points stays negative, and a B2
# surface above the current top vetoes the one-step-platform transition.
x = torch.linspace(0.16, 0.46, 5)
y = torch.linspace(-0.15, 0.15, 4)
yy, xx = torch.meshgrid(y, x, indexing="ij")
points = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)[None]
face = torch.zeros((1, 2))
top = torch.ones(1)
kwargs = dict(
    probe_start=0.14, probe_length=0.48, corridor_half_width=0.24,
    height_tolerance=0.03, min_points=8, min_forward_span=0.18,
)
positive = mdp.v132_positive_plateau_evidence(points, face, top, **kwargs)
assert positive["positive"].item()
missing = points.clone()
missing[..., 2] = float("nan")
assert not mdp.v132_positive_plateau_evidence(missing, face, top, **kwargs)["positive"].item()
higher = points.clone()
higher[0, 1, 2, 2] = 1.20
blocked = mdp.v132_positive_plateau_evidence(higher, face, top, **kwargs)
assert blocked["higher_surface"].item() and not blocked["positive"].item()

# The last-follower target is the same upper plane, not a fabricated B(n+1).
top_progress = mdp.v132_top_follow_potential(
    wheel_bottom_z=torch.tensor([[0.8, 1.0]]),
    forward_distance=torch.tensor([[0.0, 0.14]]),
    support=torch.tensor([[False, True]]),
    top_height=torch.tensor([1.0]), step_height=torch.tensor([0.2]),
    forward_target=0.14,
)
torch.testing.assert_close(top_progress, torch.tensor([[0.0, 1.0]]))

# Penalty buckets are disjoint, so a simultaneous skip cannot be double charged.
buckets = mdp.v132_violation_buckets(
    torch.tensor([True, True, True]),
    torch.tensor([False, True, True]),
    torch.tensor([False, False, True]),
)
assert buckets["wrong"].tolist() == [True, False, False]
assert buckets["simultaneous"].tolist() == [False, True, False]
assert buckets["skip"].tolist() == [False, False, True]
assert (sum(value.long() for value in buckets.values()) == 1).all()

# Separate standardisation prevents the much larger native speed return from
# numerically erasing style advantages before the actor update.
task = torch.tensor([100.0, 110.0, 120.0, 130.0])
style = torch.tensor([-0.1, -0.1, 0.0, 0.2])
task_n = algorithm.v132_normalize_advantage(task)
style_n = algorithm.v132_normalize_advantage(style)
combined = algorithm.v132_combine_advantages(task_n, style_n, 1.0, 1.0)
torch.testing.assert_close(task_n.mean(), torch.tensor(0.0), atol=1e-6, rtol=0.0)
torch.testing.assert_close(style_n.mean(), torch.tensor(0.0), atol=1e-6, rtol=0.0)
torch.testing.assert_close(combined, task_n + style_n)
torch.testing.assert_close(
    algorithm.v132_combine_advantages(task_n, torch.zeros_like(style_n), 1.0, 1.0),
    task_n,
)
phantom = algorithm.v132_normalize_style_advantage(
    torch.tensor([0.3, -0.2, 0.7]), style_nonzero_count=0,
)
assert not phantom.any(), "zero style cashflow must not create normalized actor gradients"

# Exercise the installed RSL-RL ABI, not only source strings: collect a tiny
# rollout, split total reward into task/style, compute two GAEs, update three
# independent networks, and verify checkpoint coverage.
n, horizon = 4, 3
obs = TensorDict(
    {"policy": torch.randn(n, 68), "critic": torch.randn(n, 11)},
    batch_size=[n],
)
groups = {"actor": ["policy"], "critic": ["critic"]}
actor = MLPModel(
    obs, groups, "actor", 16, hidden_dims=[16], activation="elu",
    distribution_cfg={
        "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
        "init_std": 0.5, "std_type": "log",
    },
)
critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16], activation="elu")
storage = RolloutStorage("rl", n, horizon, obs, [16], "cpu")
dual = algorithm.V132DualCriticPPO(
    actor, critic, storage, num_learning_epochs=1, num_mini_batches=1,
    learning_rate=1.0e-3, schedule="fixed", desired_kl=None,
)
style_linears = [module for module in dual.style_critic.modules() if isinstance(module, torch.nn.Linear)]
assert style_linears and not style_linears[-1].weight.any() and not style_linears[-1].bias.any()
raw = SimpleNamespace(num_envs=n, step_dt=0.02, style_rate=torch.zeros(n))
raw.reward_manager = SimpleNamespace(
    active_terms=["v132_style"],
    _step_reward=torch.zeros((n, 1)),
)
raw._v132_alternation = {
    name: torch.zeros(n, dtype=torch.bool)
    for name in (
        "encounter_event", "physical_event", "ordered_event", "wrong_event",
        "simultaneous_event", "skip_event", "top_entry_event", "top_follow_event",
        "target_active", "deploy_target_agreement",
    )
}
raw._v132_alternation["target_index"] = torch.zeros(n, dtype=torch.long)
raw._v132_deploy = {
    "target_valid": torch.zeros(n, dtype=torch.bool),
    "target_index": torch.zeros(n, dtype=torch.long),
    "settlement_target_valid": torch.zeros(n, dtype=torch.bool),
    "settlement_target_index": torch.zeros(n, dtype=torch.long),
}
dual.env = SimpleNamespace(unwrapped=raw)
initial_style = {key: value.clone() for key, value in dual.style_critic.state_dict().items()}
for step in range(horizon):
    dual.act(obs)
    raw.style_rate = torch.tensor([0.0, 1.0, -1.0, 0.5]) * (step + 1)
    raw.reward_manager._step_reward[:, 0] = raw.style_rate
    task_reward = torch.tensor([0.2, 0.3, 0.4, 0.5])
    total_reward = task_reward + raw.style_rate * raw.step_dt
    next_obs = TensorDict(
        {"policy": torch.randn(n, 68), "critic": torch.randn(n, 11)}, batch_size=[n],
    )
    dual.process_env_step(next_obs, total_reward, torch.zeros(n), {})
    torch.testing.assert_close(storage.rewards[step, :, 0], task_reward)
    torch.testing.assert_close(storage.style_rewards[step, :, 0], raw.style_rate * raw.step_dt)
    obs = next_obs
dual.compute_returns(obs)
assert torch.isfinite(storage.advantages).all()
assert torch.isfinite(storage.style_returns).all()
losses = dual.update()
assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
assert any(
    not torch.equal(value, initial_style[key])
    for key, value in dual.style_critic.state_dict().items()
)
checkpoint = dual.save()
assert checkpoint["v132_dual_return"] is True
assert "actor_state_dict" in checkpoint and "critic_state_dict" in checkpoint
assert "style_critic_state_dict" in checkpoint and "optimizer_state_dict" in checkpoint

source = MDP.read_text(encoding="utf-8")
cfg = CFG.read_text(encoding="utf-8")
algorithm_source = ALGORITHM.read_text(encoding="utf-8")
runner_source = RUNNER.read_text(encoding="utf-8")
registry = REGISTRY.read_text(encoding="utf-8")
runner_cfg = RUNNER_CFG.read_text(encoding="utf-8")

# Isolation and physical reward inheritance: V13 is the base, its generic V11
# stair rewards are not repointed, and no absorbing failure state exists.
assert "(\n    DeeproboticsM20V13V11FDAlternationEnvCfg\n)" in cfg
for forbidden in (
    "landing_event.func =", "clearance_progress.func =",
    "front_hipy_participation_settlement.func =", "landing_center.func =",
    "SEQUENCE_FAILED",
):
    assert forbidden not in cfg + source
assert "v132_" not in V13_CFG.read_text(encoding="utf-8")
assert "v132_" not in V13_MDP.read_text(encoding="utf-8")

# All three new code paths are live: MDP export -> config term/observation -> Gym
# registration -> custom dual-critic runner.
assert "from .v132_barrier_alternation import *" in MDP_INIT.read_text(encoding="utf-8")
assert "func=mdp.v132_style_objective" in cfg
assert "func=mdp.v132_event_phase_observation" in cfg
assert 'id="M20-Stair-V13.2-V11FD-BarrierAlternation-v0"' in registry
assert "DeeproboticsM20V132V11FDBarrierAlternationPPORunnerCfg" in registry
assert "class DeeproboticsM20V132V11FDBarrierAlternationPPORunnerCfg" in runner_cfg
assert "rl_training.v132_dual_critic_runner:V132DualCriticRunner" in runner_cfg
assert "rl_training.v132_dual_critic:V132DualCriticPPO" in runner_source
assert 'style_reward_term: str = "v132_style"' in algorithm_source
assert "task_reward = rewards.reshape(-1) - style_reward" in algorithm_source
assert "reward_manager._step_reward" in algorithm_source
assert "term.func(raw" not in algorithm_source
assert "task_normalized = st.task_advantages" in algorithm_source
assert "value_loss = task_value_loss + style_value_loss" in algorithm_source
assert "clip_grad_norm_(self.actor.parameters()" in algorithm_source
assert 'saved["style_critic_state_dict"]' in algorithm_source
assert "(actor.obs_dim, action_dim) != (68, 16)" in runner_source
assert "--task M20-Stair-V13.2-V11FD-BarrierAlternation-v0" in LAUNCH.read_text(encoding="utf-8")
assert "audit_m20_v13_v11fd_reference.py --headless" in LAUNCH.read_text(encoding="utf-8")

# The extra actor state has its own contact-free ledger.  It must not call the
# reward ledger or read V11's contact-backed landing settlement.
deploy_start = source.index("def v132_deployable_phase_state(")
deploy_end = source.index("\ndef v132_event_phase_observation(", deploy_start)
deploy_source = source[deploy_start:deploy_end]
for forbidden in (
    'base["landing"]', "contact_forces", "v132_alternation_state(",
    "terrain_generator", "landing_event", "reward_manager",
):
    assert forbidden not in deploy_source
observation_start = source.index("def v132_event_phase_observation(")
observation_end = source.index("\ndef v132_style_objective(", observation_start)
observation_source = source[observation_start:observation_end]
assert "v132_deployable_phase_state(" in observation_source
assert "v132_alternation_state(" not in observation_source
reward_start = source.index("def v132_alternation_state(")
reward_end = source.index("\ndef _v132_initialize_deploy_state(", reward_start)
reward_source = source[reward_start:reward_end]
assert "deploy = v132_deployable_phase_state(" in reward_source
assert 'state["seed_side"] = deploy["seed_side"]' in reward_source
assert 'expected = deploy["settlement_expected_side"]' in reward_source
assert "target_agreement & state[\"parity_seeded\"]" in reward_source

print("V132_BARRIER_ALTERNATION_CONTRACT: PASS")
