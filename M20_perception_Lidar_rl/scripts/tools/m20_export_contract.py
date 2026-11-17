"""Resolve joint semantics from the checkpoint's saved training configuration."""
import hashlib
import math
import re
from pathlib import Path
import yaml

class ConfigLoader(yaml.SafeLoader):
    pass
ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node:loader.construct_sequence(node))
# Saved configuration includes slice objects unrelated to the fields we read.
ConfigLoader.add_constructor('tag:yaml.org,2002:python/object/apply:builtins.slice',lambda loader,node:loader.construct_sequence(node))

def resolve(value,names,label):
    if isinstance(value,(int,float)):
        result=[float(value)]*len(names)
    elif isinstance(value,dict):
        result=[]
        for name in names:
            matches=[v for pattern,v in value.items() if re.fullmatch(pattern,name)]
            if len(matches)!=1:raise ValueError(f'{label}: {name} must match exactly one rule, got {matches}')
            result.append(float(matches[0]))
    else:raise ValueError(f'{label}: unsupported mapping')
    if not all(math.isfinite(v) for v in result):raise ValueError(f'{label}: nonfinite')
    return result

def from_checkpoint(checkpoint,base):
    path=Path(checkpoint).resolve().parent/'params/env.yaml'
    if not path.is_file():raise ValueError(f'M20 export requires saved training config: {path}; refusing guessed joint semantics')
    raw=path.read_bytes();cfg=yaml.load(raw,Loader=ConfigLoader)
    pos=cfg['actions']['joint_pos'];vel=cfg['actions']['joint_vel']
    names=pos['joint_names']+vel['joint_names']
    if names!=base['joint_names'] or len(set(names))!=16:raise ValueError('Unsupported M20 policy action joint order')
    obs=cfg['observations']['policy']['joint_pos']['params']['asset_cfg']
    if obs['joint_names']!=names or not obs['preserve_order']:raise ValueError('Observation/action joint ordering mismatch')
    if not pos['preserve_order'] or not vel['preserve_order'] or not pos['use_default_offset'] or not vel['use_default_offset']:
        raise ValueError('Unsupported action offset/order semantics')
    if resolve(cfg['scene']['robot']['init_state']['joint_vel'],names,'joint_vel')!=[0.]*16:raise ValueError('Nonzero velocity offsets unsupported')
    if pos.get('offset',0)!=0 or vel.get('offset',0)!=0:raise ValueError('Explicit action offsets unsupported')
    result=dict(base)
    result['default_joint_pos']=resolve(cfg['scene']['robot']['init_state']['joint_pos'],names,'default_joint_pos')
    result['action_scale']=resolve(pos['scale'],pos['joint_names'],'position scale')+resolve(vel['scale'],vel['joint_names'],'velocity scale')
    if any(x<=0 for x in result['action_scale']):raise ValueError('Action scales must be positive')
    if any(result['default_joint_pos'][12:]):raise ValueError('Wheel position observation must have zero nominal')
    result['contract_metadata']={'m20_joint_contract':'m20_saved_joint_semantics_v1','m20_training_config_sha256':hashlib.sha256(raw).hexdigest(),'m20_training_config_path':str(path)}
    return result
