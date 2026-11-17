"""CPU geometry regressions: passed, under-wheel, ahead and rotated edges."""
import importlib.util
from pathlib import Path
import torch
path=Path(__file__).resolve().parents[2]/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v111_exit_evidence.py'
spec=importlib.util.spec_from_file_location('exit_geometry',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=torch.zeros(3,2);forward=torch.tensor([[1.,0.]]).expand(3,-1)
wheels=torch.tensor([[[.3,.2],[.3,-.2],[-.3,.2],[-.3,-.2]]]).expand(3,-1,-1)
edges=torch.tensor([[[-.6,0.]], [[-.2,0.]], [[.6,0.]]])
valid=torch.ones(3,1,dtype=torch.bool)
no,new=m.riser_masks(edges,valid,wheels,root,forward,.09)
assert no.tolist()==[True,False,False]
assert new.tolist()==[False,False,True]
rotation=torch.tensor([[0.,-1.],[1.,0.]])
rno,rnew=m.riser_masks(edges@rotation.T,valid,wheels@rotation.T,root,forward@rotation.T,.09)
assert torch.equal(no,rno) and torch.equal(new,rnew)
print('V111_DIRECTIONAL_EXIT: PASS')
