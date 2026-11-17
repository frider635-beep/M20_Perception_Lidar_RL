#!/usr/bin/env python3
"""CPU/static contracts for native-63D V13.3 chain settlement."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v133_adjacent_transition.py"
RUNNER = ROOT / "source/rl_training/rl_training/v133_transition_runner.py"
DUAL = ROOT / "source/rl_training/rl_training/v132_dual_critic.py"
CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v133_v11fd_adjacent_transition_env_cfg.py"
AGENT = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py"
CURRICULUM = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py"
REGISTRY = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py"
CONTRACT = ROOT / "docs/v133_adjacent_transition_contract.md"
ACCEPTANCE = ROOT / "docs/v133_adjacent_transition_acceptance.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mdp = load(MDP, "v133_contract_mdp")
rl_training_stub = ModuleType("rl_training")
rl_training_stub.__path__ = []
sys.modules.setdefault("rl_training", rl_training_stub)
sys.modules["rl_training.v132_dual_critic"] = load(DUAL, "rl_training.v132_dual_critic")
runner = load(RUNNER, "v133_contract_runner")


# Official reconstruction and chain teaching own disjoint terrain columns.
fake = SimpleNamespace(
    device="cpu",
    cfg=SimpleNamespace(v11_perception=SimpleNamespace(
        stair_terrain_type_ids=(2, 3), ascending_stair_terrain_type_ids=(3,),
    )),
    scene=SimpleNamespace(terrain=SimpleNamespace(terrain_types=torch.tensor([0, 2, 3, 4]))),
)
torch.testing.assert_close(
    mdp.v133_descending_stair_mask(fake), torch.tensor([False, True, False, False]),
)
torch.testing.assert_close(
    mdp.v133_ascending_stair_mask(fake), torch.tensor([False, False, True, False]),
)
native_up = torch.tensor([True, False, True, False])
native_down = torch.tensor([False, True, False, True])
routed_up, routed_down = mdp.v133_route_official_descent_curriculum(
    native_up, native_down, torch.tensor([False, True, False, True]),
    torch.tensor([False, True, False, False]),
    torch.tensor([True, False, True, True]),
)
torch.testing.assert_close(routed_up, torch.tensor([True, True, True, False]))
torch.testing.assert_close(routed_down, torch.tensor([False, False, False, True]))

# Exact-adjacent shaping is paid while progress is made, cannot be replayed by
# lowering, and is fully clawed back unless the adjacent completion succeeds.
progress = mdp.v133_update_adjacent_progress(
    torch.tensor([0.0, 0.4, 0.4]), torch.tensor([0.0, 0.4, 0.4]),
    torch.tensor([0.4, 0.2, 0.7]), torch.tensor([True, True, True]),
    correct_completion=torch.tensor([False, False, True]),
    failed_completion=torch.tensor([False, True, False]),
    terminal=torch.tensor([False, False, False]),
)
torch.testing.assert_close(progress["delta"], torch.tensor([0.4, 0.0, 0.3]))
torch.testing.assert_close(progress["clawback"], torch.tensor([0.0, 0.4, 0.0]))
torch.testing.assert_close(progress["escrow"], torch.tensor([0.4, 0.0, 0.0]))
terminal = mdp.v133_update_adjacent_progress(
    torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.7]), torch.tensor([True]),
    correct_completion=torch.tensor([False]), failed_completion=torch.tensor([False]),
    terminal=torch.tensor([True]),
)
torch.testing.assert_close(terminal["delta"], torch.tensor([0.2]))
torch.testing.assert_close(terminal["clawback"], torch.tensor([0.7]))


def initial(n=1):
    return {
        "chain_length": torch.zeros(n, dtype=torch.long),
        "last_side": torch.full((n,), -1, dtype=torch.long),
        "last_riser_world_xy": torch.zeros((n, 2)),
        "last_upper_z": torch.zeros(n),
        "forward_world_xy": torch.zeros((n, 2)),
        "chain_valid": torch.zeros(n, dtype=torch.bool),
    }


def land(state, side, x, z, *, simultaneous=False):
    return mdp.v133_update_three_step_chain(
        **{key: state[key] for key in initial()},
        completion_event=torch.tensor([True]), completed_side=torch.tensor([side]),
        riser_world_xy=torch.tensor([[x, 0.0]]), upper_z=torch.tensor([z]),
        simultaneous_completion=torch.tensor([simultaneous]),
        heading_world_xy=torch.tensor([[1.0, 0.0]]), step_width=0.30,
        longitudinal_tolerance=0.08, lateral_tolerance=0.15, minimum_height_gain=0.03,
    )


# Both mirror chains become eligible only on the third adjacent completion.
for sides in ((0, 1, 0), (1, 0, 1)):
    state = initial()
    for index, side in enumerate(sides, start=1):
        state = land(state, side, 0.30 * index, 0.10 * index)
        assert state["chain_candidate_event"].item() is (index == 3)
    assert state["chain_length"].item() == 3

# Once a valid chain reaches length three, every later correct adjacent switch
# is its own overlapping three-riser window and emits its own candidate.
state = initial()
for index, side in enumerate((0, 1, 0, 1, 0), start=1):
    state = land(state, side, 0.30 * index, 0.10 * index)
    assert state["chain_candidate_event"].item() is (index >= 3)

# A same-tread follower is observable but cannot advance, reseed or erase the
# last distinct completion.  A later adjacent opposite-side completion can
# therefore continue the natural chain.
state = land(initial(), 0, 0.30, 0.10)
state = land(state, 1, 0.30, 0.10)
assert state["follower_event"].item() and state["chain_valid"].item()
assert state["chain_length"].item() == 1 and state["last_side"].item() == 0
state = land(state, 1, 0.60, 0.20)
assert state["adjacent_event"].item() and state["chain_length"].item() == 2
state = land(state, 0, 0.90, 0.30)
assert state["chain_candidate_event"].item()

# Repeated side, skip and simultaneous completion remain disjoint failures.
state = land(initial(), 0, 0.30, 0.10)
state = land(state, 0, 0.60, 0.20)
assert state["repeat_event"].item() and state["chain_length"].item() == 1
state = land(state, 1, 0.90, 0.30)
assert not state["chain_candidate_event"].item()

state = land(initial(), 0, 0.30, 0.10)
state = land(state, 1, 0.90, 0.30)
assert state["skip_event"].item() and not state["chain_candidate_event"].item()

state = land(initial(), 0, 0.30, 0.10)
state = land(state, 1, 0.60, 0.20, simultaneous=True)
assert state["simultaneous_event"].item() and not state["chain_valid"].item()

quality = mdp.v133_episode_chain_quality(
    torch.tensor([0, 1, 1, 2]), torch.tensor([0, 0, 2, 1]), minimum_precision=0.50,
)
torch.testing.assert_close(quality, torch.tensor([False, True, False, True]))
# Live stable support emits once per side/physical face, including a same-tread
# follower, without relying on the stricter V11 landing event.
steps = torch.zeros((1, 2, 2), dtype=torch.long)
tracked_xy = torch.zeros((1, 2, 2, 2))
tracked_valid = torch.zeros((1, 2, 2), dtype=torch.bool)
credited_forward = torch.zeros((1, 2))
credited_valid = torch.zeros((1, 2), dtype=torch.bool)
riser_xy = torch.tensor([[[0.30, 0.0], [0.60, 0.0]]])
riser_forward = torch.tensor([[0.30, 0.60]])
support_now = torch.tensor([[[True, False], [False, False]]])
for _ in range(3):
    live = mdp.v133_confirm_distinct_front_support(
        steps, support_now, riser_xy, riser_forward, tracked_xy, tracked_valid,
        credited_forward, credited_valid,
        confirmation_frames=3, distinct_tolerance=0.08,
    )
    steps = live["steps"]
    tracked_xy, tracked_valid = live["tracked_world_xy"], live["tracked_valid"]
    credited_forward, credited_valid = live["credited_forward"], live["credited_valid"]
assert live["side_event"].tolist() == [[True, False]]
repeated = mdp.v133_confirm_distinct_front_support(
    steps, support_now, riser_xy, riser_forward, tracked_xy, tracked_valid,
    credited_forward, credited_valid,
    confirmation_frames=3, distinct_tolerance=0.08,
)
assert not repeated["side_event"].any()

# Reusing scanner slot 0 for B2 starts confirmation from frame one.  Once B2
# is credited, returning to B1 cannot replay an event for the same side.
b2_xy = torch.tensor([[[0.60, 0.0], [0.90, 0.0]]])
b2_forward = torch.tensor([[0.60, 0.90]])
switched = mdp.v133_confirm_distinct_front_support(
    repeated["steps"], support_now, b2_xy, b2_forward,
    repeated["tracked_world_xy"], repeated["tracked_valid"],
    repeated["credited_forward"], repeated["credited_valid"],
    confirmation_frames=3, distinct_tolerance=0.08,
)
assert switched["steps"][0, 0, 0].item() == 1 and not switched["side_event"].any()
for _ in range(2):
    switched = mdp.v133_confirm_distinct_front_support(
        switched["steps"], support_now, b2_xy, b2_forward,
        switched["tracked_world_xy"], switched["tracked_valid"],
        switched["credited_forward"], switched["credited_valid"],
        confirmation_frames=3, distinct_tolerance=0.08,
    )
assert switched["side_event"].tolist() == [[True, False]]
replay = switched
for _ in range(3):
    replay = mdp.v133_confirm_distinct_front_support(
        replay["steps"], support_now, riser_xy, riser_forward,
        replay["tracked_world_xy"], replay["tracked_valid"],
        replay["credited_forward"], replay["credited_valid"],
        confirmation_frames=3, distinct_tolerance=0.08,
    )
assert not replay["side_event"].any()
torch.testing.assert_close(
    mdp.v133_episode_reset_mask(torch.tensor([1, 2]), torch.tensor([1, 1]), same_step=False),
    torch.tensor([True, False]),
)
qualification = dict(
    chain_count=10, fl_count=5, fr_count=5, chain_precision=0.6,
    velocity_tracking=0.7, finite=True, minimum_chain_count=10,
    minimum_each_final_side=1, minimum_chain_precision=0.5,
    minimum_velocity_tracking=0.65,
)
assert mdp.v133_checkpoint_qualified(**qualification)
assert not mdp.v133_checkpoint_qualified(**dict(qualification, chain_count=0))
assert not mdp.v133_checkpoint_qualified(**dict(qualification, fr_count=0))
assert not mdp.v133_checkpoint_qualified(**dict(qualification, finite=False))
torch.testing.assert_close(
    mdp.v133_episode_reset_mask(torch.tensor([1, 2]), torch.tensor([1, 1]), same_step=True),
    torch.tensor([False, False]),
)

with tempfile.TemporaryDirectory(prefix="v133_native63_") as directory:
    runner.V133TransitionRunner._write_actor_manifest(directory)
    manifest = (Path(directory) / "actor_observation_manifest.txt").read_text()
    assert "63D" in manifest and "No expected-side" in manifest

cfg_text = CFG.read_text()
agent_text = AGENT.read_text()
runner_text = RUNNER.read_text()
curriculum_text = CURRICULUM.read_text()
registry_text = REGISTRY.read_text()
contract_text = CONTRACT.read_text()
acceptance_text = ACCEPTANCE.read_text()
assert "DeeproboticsM20V13V11FDAlternationEnvCfg" in cfg_text
assert "M20V13RewardsCfg" in cfg_text
assert "ObsTerm" not in cfg_text and "v133_task_state" not in cfg_text
assert "v133_three_step_chain_reward" in cfg_text
assert "V133_ACTOR_OBS_DIM=63" in cfg_text
assert "enable_persistent_descent_rear_phase = False" in cfg_text
assert "self.rewards.descent_step_complete = None" in cfg_text
assert "self.rewards.descent_rear_support_progress = None" in cfg_text
assert "self.rewards.descent_rear_follow_delay = None" in cfg_text
assert "self.rewards.descent_rear_roll_consistency = None" in cfg_text
assert "self.rewards.flat_orientation_l2.func = mdp.v11_phase_flat_orientation_l2" in cfg_text
assert "self.rewards.lin_vel_z_l2.func = mdp.v11_phase_lin_vel_z_l2" in cfg_text
assert "V133_DESCENT_OFFICIAL_M20_REWARD_ROUTING=YES" in cfg_text
assert "V133_DESCENT_OFFICIAL_COMMIT=6d317dfb" in cfg_text
assert "official_descent_curriculum" in cfg_text
assert "v133_nonofficial_illegal_contact_termination" in cfg_text
assert "v133_ascent_front_riser_collision_termination" in cfg_text
assert "v133_live_descent_diagonal_action_penalty" not in cfg_text
assert "repeat_transition_cost" in cfg_text and "skip_transition_cost" in cfg_text
assert "adjacent_progress_weight" in cfg_text
assert "reference_riser_world_xy" not in MDP.read_text().split("def v133_episode_chain_quality", 1)[0]
assert "descent_front_follow_active" not in MDP.read_text()
assert "no persistent tread identity" in contract_text
assert "no persistent descending-tread identity" in acceptance_text
assert "num_steps_per_env = 32" in agent_text
assert "self.algorithm.num_mini_batches = 8" in agent_text
assert "save_interval = 250" in agent_text
assert "(actor.obs_dim, action_dim) != (63, 16)" in runner_text
assert '"selected_checkpoint": selected' in (ROOT / "scripts/tools/evaluate_m20_v133_frozen.py").read_text()
assert "v133_three_step_gate" in curriculum_text
assert "M20-Stair-V13.3-V11FD-AdjacentTransition-Acquisition-v0" not in registry_text
assert 'id="M20-Stair-V13.3-V11FD-FromScratch-v0"' in registry_text

print("V133_NATIVE63_THREE_STEP_CHAIN_CONTRACT: PASS", flush=True)
