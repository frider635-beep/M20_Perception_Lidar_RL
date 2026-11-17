"""Persistent planar-ground posture including the four legs' support workspace."""
import torch
from .v111_support_cycle import flat_state as base_flat_state


def workspace_cost(wheel_from_hip_x, joint_delta, cfg):
    """A free workspace, not a command to hold one joint pose.

    Per-leg maximum prevents one trailing leg being diluted by three good legs.
    All inputs are body-relative, independent of world origin/course direction.
    """
    if not (torch.isfinite(wheel_from_hip_x).all() and torch.isfinite(joint_delta).all()):
        raise RuntimeError('V112_NONFINITE_WORKSPACE: invalid robot pose; refusing reward masking')
    reach = ((wheel_from_hip_x.abs() - cfg.flat_reach_free).clamp_min(0)
             / cfg.flat_reach_scale).square().clamp_max(4)
    joints = ((joint_delta.abs() - cfg.flat_joint_free).clamp_min(0)
              / cfg.flat_joint_scale).square().clamp_max(4)
    return reach.mean(1) + .5 * reach.amax(1) + cfg.flat_joint_cost_scale * joints.mean(1)


def flat_state(env):
    base = base_flat_state(env)
    cfg = env.cfg.v112_handoff
    # Old saved V11.2 config objects remain usable for historical evaluation.
    if not getattr(cfg, 'flat_workspace_enabled', False):
        return {**base, **{key: torch.zeros_like(base['cost']) for key in
            ('workspace_cost', 'reach_max', 'joint_deviation', 'extension_rate')}}
    step = int(env.common_step_counter)
    cached = getattr(env, '_v112_workspace_state', None)
    if cached is not None and cached['step'] == step:
        return cached
    robot = env.scene['robot']
    if not hasattr(env, '_v112_workspace_ids'):
        legs = ('fl', 'fr', 'hl', 'hr')
        wheel = [robot.find_bodies(leg + '_wheel')[0][0] for leg in legs]
        hip = [robot.find_bodies(leg + '_hipy')[0][0] for leg in legs]
        joint = [robot.joint_names.index(leg + '_' + kind + '_joint')
                 for leg in legs for kind in ('hipy', 'knee')]
        env._v112_workspace_ids = wheel, hip, joint
    wheel, hip, joint = env._v112_workspace_ids
    q = robot.data.root_quat_w
    # First rotation-matrix column: world-space body forward axis.
    forward = torch.stack((1 - 2 * (q[:, 2].square() + q[:, 3].square()),
                           2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3]),
                           2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])), 1)
    reach = ((robot.data.body_pos_w[:, wheel] - robot.data.body_pos_w[:, hip])
             * forward[:, None]).sum(-1)
    delta = robot.data.joint_pos[:, joint] - robot.data.default_joint_pos[:, joint]
    leg_cost = workspace_cost(reach, delta, cfg)
    state = dict(base)
    state.update(cost=base['cost'] + leg_cost, workspace_cost=leg_cost,
                 reach_max=reach.abs().amax(1), joint_deviation=delta.abs().mean(1),
                 extension_rate=(reach.abs() > cfg.flat_reach_free).any(1).float())
    env._v112_workspace_state = state
    return state


def level_posture_penalty(env):
    state = flat_state(env)
    return -state['cost'] * state['blend']


def level_metric(env, env_ids, key):
    state = flat_state(env)
    active = state['active'][env_ids].float()
    if key == 'active':
        return active.mean()
    return (state[key][env_ids] * active).sum() / active.sum().clamp_min(1)


def flat_metric(env, env_ids, key):
    state = base_flat_state(env)
    active = state['active'][env_ids]
    if key == 'inactive_rate':
        return (~active).float().mean()
    if key == 'low_rate':
        low = state['clearance'][env_ids] < env.cfg.v111_rear_posture.flat_clearance_min
        return (low & active).float().sum() / active.float().sum().clamp_min(1)
    raise ValueError(f'Unknown flat posture metric: {key}')
