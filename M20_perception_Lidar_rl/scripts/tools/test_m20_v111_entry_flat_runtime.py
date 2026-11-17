"""Isaac wiring, saved ABI and post-entry velocity parity on the repair task."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', default='interview_prep/model_35300.pt')
p.add_argument('--num_envs', type=int, default=40)
p.add_argument('--steps', type=int, default=240)
p.add_argument('--adjacent', action='store_true')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import gymnasium as gym
import torch
from unittest.mock import patch
import rl_training.tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rl_training.v11_continuous_runner import V11ContinuousRunner
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_entry_flat as repair
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_rear_posture as native
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_terrain_geometry import v135_terrain_geometry

TASK = 'M20-Stair-V11.1-AdjacentRepair-v0' if a.adjacent else 'M20-Stair-V11.1-EntryFlatRepair-v0'
cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs)
old = parse_env_cfg('M20-Stair-V11.1-V11FD-RearPosture-v0', device=a.device, num_envs=a.num_envs)
cfg.seed = 42
for section in ('actions', 'observations', 'commands', 'terminations', 'events'):
    if section == 'commands' and a.adjacent:
        import copy
        normalized = copy.deepcopy(cfg.commands)
        normalized.base_velocity.rel_only_lin_x_envs = old.commands.base_velocity.rel_only_lin_x_envs
        normalized.base_velocity.rel_zero_vel_envs = old.commands.base_velocity.rel_zero_vel_envs
        assert repr(normalized.to_dict()) == repr(old.commands.to_dict()), section
        continue
    assert repr(getattr(cfg, section).to_dict()) == repr(getattr(old, section).to_dict()), section
assert cfg.v11_perception.to_dict() == old.v11_perception.to_dict()
assert not cfg.v111_rear_posture.timed_support_cycle
assert not cfg.v111_rear_posture.same_tread_handoff
assert not cfg.v111_rear_posture.adjacent_landing_credit
assert not hasattr(old, 'v111_entry_flat')
from v111_native_landing_parity import check_native_landing
check_native_landing(old)
agent = load_cfg_from_registry(TASK, 'rsl_rl_cfg_entry_point')
env = gym.make(TASK, cfg=cfg)
raw = env.unwrapped
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = V11ContinuousRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
runner.load(str(Path(a.checkpoint).resolve()), load_cfg={'actor': True, 'critic': True, 'optimizer': False,
    'iteration': True, 'rnd': False})
raw.common_step_counter = runner.current_learning_iteration*agent.num_steps_per_env
wrapped.reset()
assert raw.observation_manager.compute()['policy'].shape == (a.num_envs, 63)
assert raw.action_manager.total_action_dim == 16
geometry = v135_terrain_geometry(raw)
assert set(round(float(w), 2) for w in geometry['step_width'][geometry['ascent']]) == set(cfg.v111_entry_flat.ascent_treads)
# Production wrappers must give the native value once entry and planar-ground
# gates are off, including an episode that retains a failed-entry history.
zero = torch.zeros(a.num_envs, device=raw.device)
with patch.object(repair, 'flat_state', return_value=dict(workspace_cost=zero+9, blend=zero)), \
     patch.object(repair, 'entry_state', return_value=dict(entry_near=zero.bool(), entry_failed=~zero.bool())):
    torch.testing.assert_close(repair.track_velocity(raw, .5, 'base_velocity'),
        native.v111_track_lin_vel_with_descent_support(raw, .5, 'base_velocity'), rtol=0, atol=0)
    torch.testing.assert_close(repair.forward_progress(raw),
        native.v111_world_forward_with_descent_support(raw), rtol=0, atol=0)
    torch.testing.assert_close(repair.forward_deficit(raw),
        native.v111_world_forward_deficit_with_descent_support(raw), rtol=0, atol=0)
print('V111_REPAIR_ABI_GEOMETRY_NATIVE_POST_ENTRY_PARITY: PASS', flush=True)
policy = runner.get_inference_policy(device=raw.device)
weights = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
positive_flat = 0
adjacent_contacts = 0
with torch.inference_mode():
    obs = wrapped.get_observations()
    for _ in range(a.steps):
        obs, rewards, done, info = wrapped.step(policy(obs))
        assert torch.isfinite(rewards).all()
        s = repair.entry_state(raw)
        finished = s['done'] & ~s['entry_complete_event']
        assert not s['entry_near'][finished].any()
        flat = repair.flat_state(raw)
        positive_flat += int(((flat['workspace_cost'] > 0) & flat['active']).sum())
        if a.adjacent:
            from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_adjacent
            sequence = v111_adjacent.state(raw)
            adjacent_contacts += int(sequence['support_event'].sum())
            assert torch.isfinite(sequence['overshoot']).all()
            assert torch.isfinite(v111_adjacent.straight_stability_penalty(raw)).all()
for key, value in runner.alg.actor.state_dict().items():
    assert torch.equal(value, weights[key])
assert positive_flat > 0, 'flat workspace penalty never recognized the baseline motion'
print(f'V111_REPAIR_FINITE_FROZEN_ROLLOUT: PASS positive_flat_samples={positive_flat}', flush=True)
if a.adjacent:
    assert adjacent_contacts > 0, 'adjacent support reward never recognized a real contact'
    print(f'V111_ADJACENT_PHYSICAL_WIRING: PASS supported_treads={adjacent_contacts}', flush=True)
wrapped.close()
app.close()
