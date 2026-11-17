#!/usr/bin/env python3
"""V11.1 descent/flat transplant runtime and checkpoint ABI smoke."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=16)
p.add_argument('--checkpoint_path', default='logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch/model_29999.pt')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import gymnasium as gym
import torch
import rl_training.tasks
from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rl_training.v11_continuous_runner import V11ContinuousRunner
TASK='M20-Stair-V11.1-V11FD-RearPosture-v0'
cfg=parse_env_cfg(TASK,device=a.device,num_envs=a.num_envs)
assert cfg.v11_perception.front_completion_mode=='bilateral_same_riser'
assert not cfg.v11_perception.soft_alternation
assert cfg.observations.policy.height_scan.func.__name__=='v11_actor_perception'
import copy
from pathlib import Path
import yaml
def _remove(mapping, path):
    node = mapping
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]


def _diff(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return [f"{path}: keys {sorted(set(left) ^ set(right))}"]
        out = []
        for key in sorted(left):
            out.extend(_diff(left[key], right[key], f"{path}.{key}" if path else key))
        return out
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out.extend(_diff(a, b, f"{path}[{index}]"))
        return out
    try:
        if isinstance(left == right, bool) and left == right:
            return []
    except Exception:
        pass
    return [] if repr(left) == repr(right) else [f"{path}: {left!r} != {right!r}"]



saved=yaml.load(Path('logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch/params/env.yaml').read_text(),Loader=yaml.UnsafeLoader)
candidate=copy.deepcopy(cfg.to_dict())
for section in ('actions','commands','observations','events','terminations'):
    assert not _diff(saved[section],candidate[section],section), _diff(saved[section],candidate[section],section)
wrappers={'track_lin_vel_xy_exp','world_forward_progress','world_forward_deficit'}
descent={'descent_step_complete','descent_rear_follow_delay','descent_rear_roll_consistency'}
for key,ref in saved['rewards'].items():
    cur=candidate['rewards'].get(key)
    if key in descent: continue
    if key in wrappers:
        cur=copy.deepcopy(cur);cur['func']=ref['func']
    assert not _diff(ref,cur,key), _diff(ref,cur,key)
for key,cur in candidate['rewards'].items():
    if key not in saved['rewards'] and cur is not None:
        assert key.startswith(('v135_descent_','v135_post_stair_flat_')),key
for key,value in saved['v11_perception'].items():
    assert not _diff(value,candidate['v11_perception'][key],key), key
for path in (('scene','terrain','terrain_generator'),('scene','terrain','max_init_terrain_level')):
    left,right=saved,candidate
    for key in path:left,right=left[key],right[key]
    assert not _diff(left,right,str(path)),_diff(left,right,str(path))
assert not cfg.v111_rear_posture.timed_support_cycle
assert not cfg.v111_rear_posture.same_tread_handoff
assert not cfg.v111_rear_posture.adjacent_landing_credit
assert cfg.v111_rear_posture.physical_first_tread
print('V111_SAVED_FLAT_ASCENT_CONFIG_PARITY: PASS',flush=True)
from v111_native_landing_parity import check_native_landing
check_native_landing(cfg)
agent=load_cfg_from_registry(TASK,'rsl_rl_cfg_entry_point')
env=gym.make(TASK,cfg=cfg);raw=env.unwrapped
obs=raw.observation_manager.compute()
assert obs['policy'].shape==(a.num_envs,63)
assert raw.action_manager.total_action_dim==16
terms=set(raw.reward_manager.active_terms)
assert 'v111_support_cycle' not in terms and 'v111_level_posture' not in terms
for name in ('v135_descent_rear_transfer_progress','v135_descent_rear_transfer_completion',
             'v135_descent_rear_recovery','v135_post_stair_flat_progress',
             'v135_post_stair_flat_completion','v135_post_stair_flat_crawl'):
    assert name in terms,name
for name in ('v111_descent_rear_extension','v111_descent_rear_knee_flexion','v111_descent_rear_roll',
             'v111_level_exit_body_clearance','descent_step_complete','descent_rear_follow_delay',
             'descent_rear_roll_consistency'):
    assert name not in terms,name
assert not any(name.startswith(('v13_','v135_ascent_','v135_valid_','v135_adjacent_')) for name in terms)
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_descent_support_transfer import v135_descent_rear_support_transfer_state
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_descent_stance import v135_post_stair_flat_state
rear=v135_descent_rear_support_transfer_state(raw)
flat=v135_post_stair_flat_state(raw)
assert rear['P_rear'].shape==(a.num_envs,)
assert flat['quality'].shape==(a.num_envs,)
wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
runner=V11ContinuousRunner(wrapped,agent.to_dict(),log_dir=None,device=agent.device)
checkpoint=a.checkpoint_path
runner.load(checkpoint)
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert runner.current_learning_iteration==saved['iter']
assert runner.alg.optimizer.state_dict()['state'], 'optimizer state not restored'
print(f"V111_CHECKPOINT_RESTORE: PASS iter={runner.current_learning_iteration}",flush=True)
with torch.inference_mode():
    observations=wrapped.get_observations()
    for _ in range(agent.num_steps_per_env):
        actions=runner.alg.act(observations)
        observations,rewards,dones,extras=wrapped.step(actions)
        assert torch.isfinite(rewards).all()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import v11_continuous as native
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as baseline
        from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_descent_support_transfer import v135_descent_rear_support_transfer_state
        transfer=v135_descent_rear_support_transfer_state(raw)
        mask=~transfer['monitor_active'] & ~transfer['active']
        for name,func in [('track_lin_vel_xy_exp',baseline.track_lin_vel_xy_exp),('world_forward_progress',native.v11_world_forward_progress_reward),('world_forward_deficit',native.v11_world_forward_speed_deficit)]:
            term=raw.reward_manager.get_term_cfg(name)
            torch.testing.assert_close(term.func(raw,**term.params)[mask],func(raw,**term.params)[mask])
        runner.alg.process_env_step(observations,rewards,dones,extras)
    runner.alg.compute_returns(observations)
runner.alg.update()
print('V111_NATIVE_ASCENT_REWARD_PARITY_AND_PPO: PASS',flush=True)
wrapped.close();app.close()
