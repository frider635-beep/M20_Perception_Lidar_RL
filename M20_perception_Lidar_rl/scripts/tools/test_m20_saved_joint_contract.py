"""Exercise compiled deployment on independent synthetic contract fixtures."""
import argparse,json,subprocess,re,sys,copy
from pathlib import Path
import numpy as np,onnx,torch
from m20_export_contract import from_checkpoint
from export_onnx_fast import M20_CFG
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--smoke',required=True);a=p.parse_args()
out=Path('artifacts/v112_pose_repair_20260927/contract_tests');out.mkdir(parents=True,exist_ok=True)
cfg=from_checkpoint(a.checkpoint,M20_CFG);assert cfg['default_joint_pos']==[0,-.3,.6,0,-.3,.6,0,.3,-.6,0,.3,-.6,0,0,0,0]
try:from_checkpoint(out/'missing.pt',M20_CFG)
except ValueError:pass
else:raise AssertionError('missing config accepted')
class Fixed(torch.nn.Module):
 def forward(self,obs):return obs[:,:16]*0+1
base=out/'fixture.onnx';torch.onnx.export(Fixed(),torch.zeros(1,63),str(base),input_names=['obs'],output_names=['actions'],opset_version=11)
scale=np.arange(1,17)/100;nom=np.array(cfg['default_joint_pos']);perm=[0,1,2,12,3,4,5,13,6,7,8,14,9,10,11,15]
def make(name,patch):
 m=onnx.load(base);meta={'joint_names':','.join(cfg['joint_names']),'default_joint_pos':','.join(map(str,nom)),'action_scale':','.join(map(str,scale)),'m20_feature_contract':'v11_signed_edges_v2',**cfg['contract_metadata']};meta.update(patch)
 for k,v in meta.items():
  if v is None:continue
  e=m.metadata_props.add();e.key=k;e.value=v
 dst=out/(name+'.onnx');onnx.save(m,dst);return dst
import os
def run(path):
 return subprocess.run([a.smoke,str(path)],env={**os.environ,'M20_ZERO_CMD_WHEEL_STOP':'0','M20_POLICY_DEBUG':'1','M20_V11_FEATURE_CONTRACT':'v11_signed_edges_v2'},capture_output=True,text=True)
r=run(make('valid',{}));(out/'valid.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
line=next(l for l in r.stdout.splitlines() if '[M20_POLICY_DEBUG] step=0' in l)
rel=np.fromstring(line.split('joint_pos_rel=')[1].split('joint_vel_scaled=')[0],sep=' ');assert np.allclose(rel,-nom,atol=1e-6)
command=np.fromstring(line.split('robot_cmd=')[1],sep=' ');expected=nom.copy();expected+=scale;assert np.allclose(command,expected[perm],atol=1e-6),(command,expected[perm])
# Test-only wheel-stop override verifies all sixteen action scales.
# Production wheel-stop behavior is unchanged and checked by launcher smoke.
for name,patch in [('missing',{'m20_joint_contract':None}),('nan',{'default_joint_pos':'nan,'+','.join(map(str,nom[1:]))}),('order',{'joint_names':','.join(reversed(cfg['joint_names']))}),('negative_scale',{'action_scale':','.join(map(str,-scale))})]:
 r=run(make(name,patch));(out/(name+'.log')).write_text(r.stdout+r.stderr);assert r.returncode!=0,name
print('COMPILED_SAVED_JOINT_CONTRACT_OBSERVATION_POSITION_SCALE_NEGATIVES: PASS')
