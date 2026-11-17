"""Isolated flat-ground physics A/B, ONNX fixed, only inference nominal differs.
Uses deployment robot MJCF and PD gains; excludes ROS, radar and keyboard timing.
"""
import numpy as np,mujoco,onnxruntime as ort,json
from pathlib import Path
root=Path('artifacts/v112_crawl_diagnosis_20260927');meta=json.loads((root/'native.json').read_text());names=meta['joint_names'];true=np.array(meta['nominal']);dep=np.array(meta['deployment_nominal'])
xml='/home/ubuntu/桌面/m20_1/M20_perception_deploy_63d/src/M20_sdk_deploy/M20_description/m20_mjcf/mjcf/M20.xml'
onnx='/tmp/m20_perception_deploy_63d_sim2sim/models/model_54300.20260927_200305_2930716.v11_signed_edges_v2.onnx'
opt=ort.SessionOptions();opt.intra_op_num_threads=1;session=ort.InferenceSession(onnx,sess_options=opt,providers=['CPUExecutionProvider'])
results={};scale=np.array([.125,.25,.25]*4+[5]*4)
for label,nom in [('matched',true),('deployed',dep)]:
 m=mujoco.MjModel.from_xml_path(xml);m.opt.timestep=.001;d=mujoco.MjData(m)
 jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];aid=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names];bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base_link')
 d.qpos[:3]=[0,0,.58];d.qpos[3:7]=[1,0,0,0];d.qpos[qa]=true;mujoco.mj_forward(m,d)
 last=np.zeros(16);rows=[];kp=np.array([80]*12+[0]*4);kd=np.array([2]*12+[.6]*4)
 for step in range(1250):
  command=np.array([.7 if 100<=step<1100 else 0,0,0]);q=d.qpos[qa].copy();v=d.qvel[va].copy();rel=q-nom;rel[12:]=0
  rot=d.xmat[bid].reshape(3,3);gravity=rot.T@np.array([0,0,-1]);omega=d.qvel[3:6]
  obs=np.r_[omega*.25,gravity,command,rel,v*.05,last,np.zeros(6)].astype(np.float32)[None]
  action=session.run(None,{'obs':obs})[0][0].astype(float);last=action.copy();target=nom+scale*action;vt=np.r_[np.zeros(12),target[12:]]
  for _ in range(20):
   torque=kp*(target-d.qpos[qa])+kd*(vt-d.qvel[va]);d.ctrl[aid]=torque;mujoco.mj_step(m,d)
  rows.append(np.r_[step*.02,d.qpos[:3],gravity,q,command,d.qvel[:3]])
 rows=np.array(rows);np.savez_compressed(root/('mujoco_'+label+'.npz'),samples=rows)
 use=(rows[:,0]>5)&(rows[:,0]<21)
 results[label]={'height_mean':float(rows[use,3].mean()),'height_min':float(rows[use,3].min()),'gravity_xy_mean':float(np.linalg.norm(rows[use,4:6],axis=1).mean()),'joint_mean':rows[use,7:23].mean(0).tolist(),'end_position':rows[-1,1:4].tolist(),'mean_vx':float(rows[use,-3].mean()),'samples':int(use.sum())}
print(json.dumps(results,indent=2));(root/'mujoco_ab.json').write_text(json.dumps(results,indent=2))
