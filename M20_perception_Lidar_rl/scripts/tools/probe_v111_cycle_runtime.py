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
cfg.seed=42
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
counts={}
def count(k,x): counts[k]=counts.get(k,0)+int(x.sum().item())
with torch.inference_mode():
    observations=wrapped.get_observations()
    for _ in range(256):
        actions=runner.alg.act(observations)
        observations,rewards,dones,extras=wrapped.step(actions)
        assert torch.isfinite(rewards).all()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp.v11_continuous import v11_state
        s=v11_state(raw)['landing']; asset=raw.scene['robot'];ids=raw._v11_front_body_ids
        v=asset.data.body_lin_vel_w[:,ids];w=asset.data.body_ang_vel_w[:,ids]
        r=torch.zeros_like(v);r[...,2]=-raw.cfg.v11_perception.wheel_radius
        slip=torch.linalg.vector_norm((v+torch.linalg.cross(w,r,dim=-1))[...,:2],dim=-1)
        f=raw.scene.sensors['contact_forces'].data.net_forces_w[:,raw._v11_front_contact_ids,2]
        active=s['target_active'][:,None]
        support=(s['support_steps']>=3)&active
        quiet=(slip<=raw.cfg.v111_rear_posture.cycle_slip_limit)&(f>=raw.cfg.v111_rear_posture.cycle_contact_force)
        upper=support&quiet&s['crossing']
        lower=(s['wheel_yaw'][...,0]<s['region']['x_min'][:,None]-raw.cfg.v11_perception.wheel_radius-raw.cfg.v11_perception.front_margin)&quiet&((s['wheel_bottom']-raw._v111_cycle_risers['riser_lower_world'][:,:1]).abs()<=raw.cfg.v111_rear_posture.cycle_plane_tolerance)
        for k,x in [('upper_before_collision',upper),('upper_clean_history',upper&~s['vertical_contact']),('upper_clean_now',upper&~s['vertical_contact_now']),('lower',lower),('choice_before_collision',upper&lower.flip(1)),('choice_clean_history',upper&~s['vertical_contact']&lower.flip(1)),('choice_clean_now',upper&~s['vertical_contact_now']&lower.flip(1))]:count(k,x)

        for k,x in [('active',s['target_active']),('candidate',s['candidate_ready']&active),('support',s['support']&active),('current_support', (s['support_steps']>=3)&active),('crossing',s['crossing']&active),('clearance',s['clearance']&active),('slip_ok', (slip<=.25)&active),('force_ok',(f>=15)&active),('leader',s['cycle_leader']>=0),('delta',s['cycle_delta']>0),('ready',s['cycle_ready']),('done',s['cycle_done']),('bad',s['cycle_bad'])]:count(k,x)
print('CYCLE_PROBE_COUNTS',counts,flush=True)
wrapped.close();app.close()
