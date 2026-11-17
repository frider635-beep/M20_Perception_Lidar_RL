"""V11.2 reset, reward wiring and frozen-policy rollout; no training/update."""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--num_envs', type=int, default=40)
p.add_argument('--rollout_steps', type=int, default=24)
p.add_argument('--checkpoint_path', default='logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch/model_29999.pt')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import gymnasium as gym
import torch
import rl_training.tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rl_training.v11_continuous_runner import V11ContinuousRunner
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_terrain_geometry import v135_terrain_geometry
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v112_flat_posture import flat_state

TASK = 'M20-Stair-V11.2-VariableTread-v0'
cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs)
cfg.seed = 42
old = parse_env_cfg('M20-Stair-V11.1-V11FD-RearPosture-v0', device=a.device, num_envs=a.num_envs)
for section in ('actions', 'observations', 'commands', 'terminations', 'events'):
    assert repr(getattr(cfg, section).to_dict()) == repr(getattr(old, section).to_dict()), section
assert cfg.v111_rear_posture.timed_support_cycle
assert not old.v111_rear_posture.timed_support_cycle
from v112_handoff_integration import check_handoff
check_handoff(cfg)
from v111_native_landing_parity import check_native_landing
check_native_landing(old)
assert old.scene.terrain.terrain_generator.num_cols == 20
assert old.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs'].step_width == .3
assert 'v112_level_posture' not in old.rewards.to_dict()
agent = load_cfg_from_registry(TASK, 'rsl_rl_cfg_entry_point')
env = gym.make(TASK, cfg=cfg)
raw = env.unwrapped
assert 'v112_level_posture' in raw.reward_manager.active_terms
assert 'v112_support_cycle' in raw.reward_manager.active_terms
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = V11ContinuousRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
runner.load(a.checkpoint_path)
raw.common_step_counter = int(runner.current_learning_iteration) * int(agent.num_steps_per_env)
wrapped.reset()
assert runner.alg.optimizer.state_dict()['state']
assert raw.observation_manager.compute()['policy'].shape == (a.num_envs, 63)
geometry = v135_terrain_geometry(raw)
assert set(round(float(v), 2) for v in geometry['step_width'][geometry['ascent']]) == set(cfg.v112_tread_widths)
types = raw.scene.terrain.terrain_types
stair_ids = torch.tensor(cfg.v11_perception.stair_terrain_type_ids, device=raw.device)
stairs = (types[:, None] == stair_ids[None]).any(1)
assert set(round(float(v), 2) for v in geometry['step_width'][stairs & ~geometry['ascent']]) == set(cfg.v112_tread_widths)
offset = (raw.scene['robot'].data.root_pos_w[:, :2] - raw.scene.env_origins[:, :2]).abs()
assert (offset[geometry['ascent']] < geometry['radii'][geometry['ascent']]).all(), 'reset outside ascent approach platform'
assert flat_state(raw)['active'].any(), 'flat reward has no real scanner evidence'
print('V112_RUNTIME_CONFIG_RESET_GEOMETRY_ABI: PASS', flush=True)
policy = runner.get_inference_policy(device=raw.device)
weights = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
with torch.inference_mode():
    obs = wrapped.get_observations()
    positive_workspace_samples = 0
    for _ in range(a.rollout_steps):
        action = policy(obs)
        obs, rewards, dones, extras = wrapped.step(action)
        assert torch.isfinite(rewards).all()
        s = flat_state(raw)
        assert torch.isfinite(s['cost']).all()
        positive_workspace_samples += int(((s['workspace_cost'] > 0) & s['active']).sum())
for k, v in runner.alg.actor.state_dict().items():
    assert torch.equal(v, weights[k]), f'policy mutated: {k}'
print('V112_FROZEN_POLICY_ROLLOUT_NO_TRAINING: PASS', flush=True)
print('V112_WORKSPACE_POSITIVE_SAMPLES', positive_workspace_samples, flush=True)
if a.rollout_steps >= 240:
    assert positive_workspace_samples > 0, 'workspace cost never activated in real rollout'
print('V112_FLAT_EVIDENCE_COUNT', int(flat_state(raw)['active'].sum()), flush=True)
wrapped.close()
app.close()
