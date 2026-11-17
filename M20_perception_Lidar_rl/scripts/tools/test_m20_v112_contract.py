"""CPU regression: actual stair meshes, column lookup and late flat crouching."""
import ast
import importlib.util
import sys
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity'
pkg = types.ModuleType('v112_fixture')
pkg.__path__ = [str(BASE / 'mdp')]
sys.modules[pkg.__name__] = pkg

def load(name):
    spec = importlib.util.spec_from_file_location('v112_fixture.' + name, BASE / 'mdp' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

geo = load('v135_terrain_geometry')
terrain = load('v112_terrain')
flat = load('v111_support_cycle')
metric = load('v112_flat_posture')
tree = ast.parse((BASE / 'config/wheeled/deeprobotics_m20/v111_v11fd_rear_posture_env_cfg.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'M20V111RearPostureCfg')
posture = NS(**{n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)})
tree = ast.parse((BASE / 'config/wheeled/deeprobotics_m20/v112_variable_tread_env_cfg.py').read_text())
widths = next(ast.literal_eval(n.value) for n in ast.walk(tree)
              if isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', '') == 'v112_tread_widths')
assert widths == (.20, .24, .28, .32, .36, .40, .44)

class Cfg(NS):
    def copy(self):
        return deepcopy(self)

def generator(cols):
    def stair(weight, function):
        return Cfg(proportion=weight, step_width=.3, border_width=1., platform_width=3., holes=False,
                   function=NS(__name__=function))
    return Cfg(size=(8., 8.), curriculum=True, num_cols=cols, sub_terrains={
        'flat': Cfg(proportion=.2),
        'pyramid_stairs': stair(.2, 'pyramid_stairs_terrain'),
        'pyramid_stairs_inv': stair(.3, 'inverted_pyramid_stairs_terrain'),
        'boxes': Cfg(proportion=.1), 'rough': Cfg(proportion=.1),
        'slope': Cfg(proportion=.05), 'slope_inv': Cfg(proportion=.05),
    })

# Negative control: twenty columns must not silently omit tread variants.
try:
    terrain.configure_treads(generator(20), widths)
except ValueError:
    pass
else:
    raise AssertionError('missing-width detector failed its negative control')
cfg = generator(40)
stairs, ascent = terrain.configure_treads(cfg, widths)
assert stairs == tuple(range(8, 28)) and ascent == tuple(range(16, 28))
env = NS(device='cpu', cfg=NS(scene=NS(terrain=NS(terrain_generator=cfg))),
         scene=NS(terrain=NS(terrain_types=torch.arange(40))))
g = geo.v135_terrain_geometry(env)
assert g['ascent'].nonzero().flatten().tolist() == list(ascent)
names = geo.column_names(cfg)
for col in stairs:
    assert abs(g['step_width'][col].item() - cfg.sub_terrains[names[col]].step_width) < 1e-6
env.scene.terrain.terrain_types = torch.tensor([27, 16, 10])
assert torch.allclose(geo.v135_terrain_geometry(env)['step_width'], g['step_width'][[27, 16, 10]])
print('V112_COLUMN_COVERAGE_AND_RESET_LOOKUP: PASS', flush=True)

# Use the installed Isaac mesh generators, independent of our geometry lookup.
import __editable___isaaclab_0_54_4_finder as finder
mesh_dir = Path(finder.MAPPING['isaaclab']) / 'terrains/trimesh'
pkg = types.ModuleType('v112_actual_mesh'); pkg.__path__ = [str(mesh_dir)]
sys.modules[pkg.__name__] = pkg
spec = importlib.util.spec_from_file_location('v112_actual_mesh.mesh_terrains', mesh_dir / 'mesh_terrains.py')
mesh = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mesh; spec.loader.exec_module(mesh)
for width in widths:
    for function in (mesh.pyramid_stairs_terrain, mesh.inverted_pyramid_stairs_terrain):
        for difficulty in (0., .5, 1.):
            c = NS(size=(8., 8.), border_width=1., platform_width=3., step_width=width,
                   step_height_range=(.05, .23), holes=False)
            meshes, center = function(difficulty, c)
            samples = np.linspace(center[0], 7.9, 15001)
            surface = np.full_like(samples, -np.inf)
            for part in meshes:
                lo, hi = part.bounds
                inside = (samples >= lo[0]) & (samples <= hi[0]) & (center[1] >= lo[1]) & (center[1] <= hi[1])
                surface[inside] = np.maximum(surface[inside], hi[2])
            idx = np.where(np.abs(np.diff(surface)) > .025)[0]
            edges = (samples[idx] + samples[idx + 1]) / 2
            assert len(edges) >= 3
            assert np.allclose(np.diff(edges), width, atol=.001), (function.__name__, width, edges)
print('V112_ACTUAL_MESH_ALL_SEVEN_TREADS_BOTH_DIRECTIONS: PASS', flush=True)

# Exercise the real stateful flat reward; only simulator inputs are fixtures.
base = dict(riser_pool_world_xy=torch.zeros(1, 2, 2),
            riser_pool=dict(valid=torch.zeros(1, 2, dtype=torch.bool)))
native = types.ModuleType('v112_fixture.v11_continuous')
native.v11_state = lambda env: base
sys.modules[native.__name__] = native
wheel = torch.tensor([[[.3, .2, .1], [.3, -.2, .1], [-.3, .2, .1], [-.3, -.2, .1]]])
hits = torch.tensor([[[.3, .2, 0.], [.3, -.2, 0.], [-.3, .2, 0.], [-.3, -.2, 0.], [.05, 0., 0.], [.55, 0., 0.]]])
data = NS(body_pos_w=wheel, root_pos_w=torch.tensor([[0., 0., .45]]),
          projected_gravity_b=torch.tensor([[0., 0., -1.]]), root_ang_vel_b=torch.zeros(1, 3))
robot = NS(data=data, find_bodies=lambda name: ([('fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel').index(name)], [name]))
class Scene(dict):
    pass
scene = Scene(robot=robot)
scene.sensors = dict(v10_pgtt_scanner=NS(data=NS(ray_hits_w=hits)))
env = NS(scene=scene, num_envs=1, device='cpu', common_step_counter=0, step_dt=.02,
         episode_length_buf=torch.zeros(1, dtype=torch.long),
         command_manager=NS(get_term=lambda name: NS(v11_path_heading_w=torch.zeros(1))),
         cfg=NS(v111_rear_posture=posture, v11_perception=NS(wheel_radius=.1)))
def tick():
    env.common_step_counter += 1
    env.episode_length_buf += 1
    return flat.level_posture_penalty(env)

for _ in range(1500):
    assert tick().item() == 0
data.root_pos_w[:, 2] = .30
for _ in range(1500):
    assert tick().item() < -.9, 'late crouch escaped the persistent constraint'
assert metric.flat_metric(env, slice(None), 'low_rate').item() == 1
frames = env._v111_level_posture['frames'].clone()
flat.level_posture_penalty(env)
assert torch.equal(frames, env._v111_level_posture['frames']), 'same-step cache advanced time'
hits[:, :2, 2] = .18
assert tick().item() == 0, 'stair pose was penalized as flat'
hits[:] = torch.nan
assert tick().item() == 0
assert metric.flat_metric(env, slice(None), 'inactive_rate').item() == 1
hits[:] = torch.tensor([[[.3, .2, 0.], [.3, -.2, 0.], [-.3, .2, 0.], [-.3, -.2, 0.], [.05, 0., 0.], [.55, 0., 0.]]])
env.episode_length_buf[:] = 0
assert -.3 < tick().item() < 0, 'reset retained old confirmation'
for _ in range(5): tick()
assert flat.level_posture_penalty(env).item() < -.9
print('V112_LATE_CROUCH_60S_MISSING_RAYS_STAIRS_RESET_CACHE: PASS', flush=True)
