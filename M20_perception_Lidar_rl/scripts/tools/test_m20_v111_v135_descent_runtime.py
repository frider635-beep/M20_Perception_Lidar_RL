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
from v111_adjacent_regression import check_adjacent_landing
check_adjacent_landing(cfg)
from v111_cycle_integration import check_cycle_integration
check_cycle_integration(cfg)
agent=load_cfg_from_registry(TASK,'rsl_rl_cfg_entry_point')
env=gym.make(TASK,cfg=cfg);raw=env.unwrapped
obs=raw.observation_manager.compute()
assert obs['policy'].shape==(a.num_envs,63)
assert raw.action_manager.total_action_dim==16
terms=set(raw.reward_manager.active_terms)
for name in ('v135_descent_rear_transfer_progress','v135_descent_rear_transfer_completion',
             'v135_descent_rear_recovery','v135_post_stair_flat_progress',
             'v135_post_stair_flat_completion','v135_post_stair_flat_crawl',
             'v111_support_cycle','v111_level_posture'):
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
        runner.alg.process_env_step(observations,rewards,dones,extras)
    runner.alg.compute_returns(observations)
runner.alg.update()
print('V111_V135_DESCENT_FLAT_RESUME_PPO_SMOKE: PASS',flush=True)
wrapped.close();app.close()
