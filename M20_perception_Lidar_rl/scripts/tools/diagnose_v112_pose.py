"""Frozen Isaac rollout; isolate deployment nominal-pose mismatch without training."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--deploy-offset',action='store_true');p.add_argument('--forward-condition',action='store_true');p.add_argument('--steps',type=int,default=750)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import gymnasium as gym
import torch,numpy as np,json
from pathlib import Path
import rl_training.tasks
from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rl_training.v11_continuous_runner import V11ContinuousRunner
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v111_support_cycle import flat_state
T='M20-Stair-V11.2-VariableTread-v0';cfg=parse_env_cfg(T,device=a.device,num_envs=40);cfg.seed=42
cfg.observations.policy.enable_corruption=False
agent=load_cfg_from_registry(T,'rsl_rl_cfg_entry_point');env=gym.make(T,cfg=cfg);raw=env.unwrapped
w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);r=V11ContinuousRunner(w,agent.to_dict(),log_dir=None,device=agent.device);r.load(a.checkpoint)
raw.common_step_counter=int(r.current_learning_iteration)*agent.num_steps_per_env;w.reset();policy=r.get_inference_policy(device=raw.device)
robot=raw.scene['robot'];names=cfg.observations.policy.joint_pos.params['asset_cfg'].joint_names
ids=[robot.joint_names.index(n) for n in names];nom=robot.data.default_joint_pos[:,ids].clone();dep=nom.new_tensor([0,-.6,1,0,-.6,1,0,.6,-1,0,.6,-1,0,0,0,0]);scale=nom.new_tensor([.125,.25,.25]*4+[5]*4)
meta={'checkpoint':a.checkpoint,'deploy_offset':a.deploy_offset,'forward_condition':a.forward_condition,'joint_names':names,'nominal':nom[0].tolist(),'deployment_nominal':dep.tolist(),'reward_names':raw.reward_manager.active_terms,'num_envs':40,'steps':a.steps,'dt':raw.step_dt}
print('POSE_CONTRACT',json.dumps(meta),flush=True)
records={};last=torch.zeros_like(nom)
with torch.inference_mode():
 obs=w.get_observations()
 for i in range(a.steps):
  if a.forward_condition:
   term=raw.command_manager.get_term('base_velocity')
   term.time_left[:]=10000
   term.is_heading_env[:]=False
   term.is_standing_env[:]=False
   term.vel_command_b[:]=term.vel_command_b.new_tensor([.7,0,0])
   obs['policy'][:,6:9]=term.vel_command_b
  if a.deploy_offset:
   obs['policy'][:,9:25]+=nom-dep
   obs['policy'][:,41:57]=last
  act=policy(obs);last=act.clone()
  applied=act+(dep-nom)/scale if a.deploy_offset else act
  obs,reward,done,extra=w.step(applied)
  s=flat_state(raw)
  if i%5==0:
   values={'q':robot.data.joint_pos[:,ids],'root':robot.data.root_pos_w,'quat':robot.data.root_quat_w,'vel':robot.data.root_lin_vel_b,'command':raw.command_manager.get_command('base_velocity'),'flat_active':s['active'],'clearance':s['clearance'],'cost':s['cost'],'blend':s['blend'],'tilt':s['tilt'],'episode_length':raw.episode_length_buf,'terrain_type':raw.scene.terrain.terrain_types,'terrain_level':raw.scene.terrain.terrain_levels,'terms':raw.reward_manager._step_reward,'body':robot.data.body_pos_w,'obs':obs['policy']}
   for k,v in values.items():records.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
  if i%250==0:print('STEP',i,flush=True)
meta['body_names']=robot.body_names
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**{k:np.stack(v) for k,v in records.items()});out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('POSE_DIAG_COMPLETE',flush=True)
w.close();app.close()
