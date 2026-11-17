"""FK only: reconstruct logged support geometry, not a dynamics acceptance."""
import mujoco,numpy as np,json
from pathlib import Path
xml='/home/ubuntu/桌面/m20_1/M20_perception_deploy_63d/src/M20_sdk_deploy/M20_description/m20_mjcf/mjcf/M20_stair.xml'
m=mujoco.MjModel.from_xml_path(xml);d=mujoco.MjData(m)
names=[f'{leg}_{j}_joint' for leg in ('fl','fr','hl','hr') for j in ('hipx','hipy','knee','wheel')]
poses={'training_nominal':[0,-.3,.6,0]*2+[0,.3,-.6,0]*2,'deployment_nominal':[0,-.6,1,0]*2+[0,.6,-1,0]*2,'logged_forward_extension':[-.0751,-.511,1.49,0,-.00692,-.0902,1.29,0,-.107,-.257,-.514,0,.2,.111,-.821,0]}
report={}
for label,q in poses.items():
 d.qpos[:3]=0;d.qpos[3:7]=[1,0,0,0]
 for name,value in zip(names,q):d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)]]=value
 mujoco.mj_forward(m,d)
 wheels=np.array([d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,leg+'_wheel')] for leg in ('fl','fr','hl','hr')]);plane=np.linalg.lstsq(np.c_[wheels[:,:2],np.ones(4)],wheels[:,2],rcond=None)[0]
 report[label]={'wheel_centers_body':wheels.tolist(),'wheel_plane':plane.tolist(),'root_clearance_estimate':float(-plane[2]/np.sqrt(1+np.sum(plane[:2]**2))+.09),'wheel_plane_fit_error':float(np.max(np.abs(np.c_[wheels[:,:2],np.ones(4)]@plane-wheels[:,2]))),'assumption':'FK support-plane estimate, not measured ground or torso orientation'}
p=Path('artifacts/v112_crawl_diagnosis_20260927/logged_pose_geometry.json');p.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
