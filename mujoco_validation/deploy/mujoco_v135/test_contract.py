"""Differential checks against training code and the installed RSL-RL model."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from perception import extract_risers, actor_feature, Perception
from policy import SavedPolicy
from run import Robot, default_checkpoint


@pytest.fixture(scope="module")
def policy():
    torch.set_num_threads(1)
    return SavedPolicy(default_checkpoint())


def test_detector_matches_training(policy):
    from reference_risers import extract_two_risers_points
    rng = np.random.default_rng(73099)
    x, y = np.meshgrid(np.linspace(1.25, -.75, 21), np.linspace(.7, -.7, 15), indexing="ij")
    points = []
    for _ in range(256):
        edges = np.sort(rng.uniform(-.5, 1.2, 4))
        heights = rng.choice([-1., 1.]) * rng.uniform(.05, .3, 4)
        z = sum(h * (x > edge + rng.uniform(-.3, .3)*y) for edge, h in zip(edges, heights))
        z += rng.normal(0., .001, z.shape)
        z[rng.random(z.shape) < .03] = np.nan
        points.append(np.stack((x, y, z), -1))
    c = policy.perception
    reference = extract_two_risers_points(
        torch.tensor(np.asarray(points)), corridor_half_width=c["corridor_half_width"],
        h_min=c["min_step_height"]-c["height_valid_tolerance"], d_end=c["passed_distance"],
        min_separation=c["min_riser_separation"], signed_edges=True)
    for index, grid in enumerate(points):
        actual = extract_risers(grid, c)
        for key in reference:
            np.testing.assert_allclose(actual[key], reference[key][index].numpy(), rtol=0, atol=1.e-12)
    flat = np.stack((x, y, np.zeros_like(x)), -1)
    assert not extract_risers(flat, c)["valid"].any()


def test_actor_matches_rsl(policy):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    ref = MLPModel(TensorDict({"policy": torch.zeros(1, 63)}, batch_size=[1]),
                   {"actor": ["policy"]}, "actor", 16, hidden_dims=[512, 256, 128],
                   activation="elu", obs_normalization=False,
                   distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "log"})
    state = torch.load(policy.path, map_location="cpu", weights_only=True)["actor_state_dict"]
    ref.load_state_dict(state, strict=True)
    ref.eval()
    observations = torch.randn(128, 63, generator=torch.Generator().manual_seed(42))
    with torch.inference_mode():
        expected = ref(TensorDict({"policy": observations}, batch_size=[128]))
        actual = policy.actor(observations)
    torch.testing.assert_close(actual, expected, rtol=1.e-6, atol=1.e-6)


def test_missing_saved_config_is_rejected(tmp_path):
    from checkpoint_config import resolve_configs
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint without configuration")
    with pytest.raises(FileNotFoundError, match="--config-dir"):
        resolve_configs(checkpoint, search_roots=[])


def test_relocated_config_requires_identical_checkpoint(tmp_path):
    from checkpoint_config import resolve_configs
    checkpoint = tmp_path / "model_35300.pt"
    checkpoint.write_bytes(b"original checkpoint")
    logs = tmp_path / "logs/rsl_rl"
    wrong = logs / "wrong/run"
    right = logs / "right/run"
    for run_dir, content in ((wrong, b"different checkpoint"), (right, checkpoint.read_bytes())):
        (run_dir / "params").mkdir(parents=True)
        (run_dir / checkpoint.name).write_bytes(content)
        for name in ("env.yaml", "agent.yaml"):
            (run_dir / "params" / name).write_text(str(run_dir))
    params, mode, source = resolve_configs(checkpoint, search_roots=[logs])
    assert params == right / "params" and mode == "checkpoint_sha256"
    assert source == right / checkpoint.name
    (right / checkpoint.name).write_bytes(b"also different")
    with pytest.raises(FileNotFoundError, match="--config-dir"):
        resolve_configs(checkpoint, search_roots=[logs])


def test_ambiguous_config_and_incomplete_adjacent_are_rejected(tmp_path):
    from checkpoint_config import resolve_configs
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"same model")
    logs = tmp_path / "logs/rsl_rl"
    for label in ("a", "b"):
        run_dir = logs / label / "run"
        (run_dir / "params").mkdir(parents=True)
        (run_dir / checkpoint.name).write_bytes(checkpoint.read_bytes())
        for name in ("env.yaml", "agent.yaml"):
            (run_dir / "params" / name).write_text(label)
    with pytest.raises(ValueError, match="conflicting"):
        resolve_configs(checkpoint, search_roots=[logs])
    explicit = logs / "b/run/params"
    assert resolve_configs(checkpoint, explicit)[0] == explicit
    (tmp_path / "params").mkdir()
    with pytest.raises(FileNotFoundError, match="Missing saved"):
        resolve_configs(checkpoint, search_roots=[logs])


def test_real_relocated_v111_policy_uses_original_semantics(policy, tmp_path):
    # Regression for the user's copied checkpoint: same iteration number also
    # exists in V13.5, so matching only the filename selects the wrong run.
    import yaml
    import shutil
    from saved_config import ConfigLoader
    workspace = HERE.parents[2]
    copied = workspace / "M20_perception_Lidar_rl/interview_prep/model_35300.pt"
    params = copied.parent / "params"
    moved = tmp_path / "model_35300.pt"
    shutil.copy2(copied, moved)
    relocated = SavedPolicy(copied)
    explicit = SavedPolicy(moved, params)
    assert relocated.metadata["config_source_checkpoint"] is None
    assert relocated.metadata["config_resolution"] == "adjacent"
    assert relocated.iteration == explicit.iteration == 35300
    # Read the selected YAML independently; ensure the semantics helper uses
    # that file too, rather than still looking beside the moved checkpoint.
    cfg = yaml.load((params / "env.yaml").read_text(), Loader=ConfigLoader)
    assert relocated.cfg == cfg
    for field in ("nominal", "action_scale", "kp", "kd", "effort", "armature"):
        np.testing.assert_array_equal(getattr(relocated, field), getattr(policy, field))
        np.testing.assert_array_equal(getattr(explicit, field), getattr(policy, field))
    obs = np.linspace(-1., 1., 63, dtype=np.float32)
    np.testing.assert_array_equal(relocated(obs), policy(obs))
    np.testing.assert_array_equal(explicit(obs), policy(obs))


def test_incompatible_actor_is_rejected(policy, tmp_path):
    import shutil
    params = tmp_path / "params"
    params.mkdir()
    for name in ("env.yaml", "agent.yaml"):
        shutil.copy2(policy.path.parent / "params" / name, params / name)
    checkpoint = torch.load(policy.path, map_location="cpu", weights_only=True)
    checkpoint["actor_state_dict"]["mlp.0.weight"] = checkpoint["actor_state_dict"]["mlp.0.weight"][:, :-1]
    target = tmp_path / "model.pt"
    torch.save(checkpoint, target)
    with pytest.raises(RuntimeError, match="size mismatch"):
        SavedPolicy(target)


def test_observation_order_scaling_and_wheel_mask(policy):
    q = policy.nominal + np.arange(16)*.01
    obs = policy.observation(np.array([1., 2., 3.]), np.array([0., 0., -1.]), np.array([.4, -.2, .3]),
                             q, np.arange(16), np.arange(16)*.02, np.array([1., -.16, .5, 1., -.16, .8]))
    np.testing.assert_allclose(obs[:3], [.25, .5, .75])
    np.testing.assert_allclose(obs[6:9], [.4, -.2, .3])
    np.testing.assert_allclose(obs[9:21], np.arange(12)*.01, atol=1.e-7)
    np.testing.assert_array_equal(obs[21:25], 0.)
    np.testing.assert_allclose(obs[25:41], np.arange(16)*.05, atol=1.e-7)
    np.testing.assert_allclose(obs[41:57], np.arange(16)*.02, atol=1.e-7)
    np.testing.assert_allclose(obs[57:], [1., -.16, .5, 1., -.16, .8])


def test_promotion_and_command_gate(policy):
    r = dict(valid=np.array([True, True]), riser_height=np.array([.16, .16]), riser_distance=np.array([-.1, .2]))
    np.testing.assert_allclose(actor_feature(r, .5, policy.perception), [1., .16, .2, 0., 0., 0.])
    np.testing.assert_array_equal(actor_feature(r, 0., policy.perception), 0.)


def test_confirmation_hold_and_delay(policy):
    p = Perception.__new__(Perception)
    p.cfg, p.mode, p.rng = policy.perception, "clean", np.random.default_rng(42)
    p.reset()
    sample = np.array([1., .16, .5, 0., 0., 0.], dtype=np.float32)
    sequence = [p.filter(sample) for _ in range(7)]
    np.testing.assert_array_equal(np.asarray(sequence)[:3], 0.)
    np.testing.assert_allclose(sequence[3], sample)
    sequence = [p.filter(np.zeros(6)) for _ in range(5)]
    assert sequence[3][0] == 1.
    np.testing.assert_array_equal(sequence[4], 0.)
    p.reset()
    np.testing.assert_array_equal(p.filter(np.zeros(6)), 0.)


def test_joint_mapping_pd_and_real_terrain_rays(policy):
    args = SimpleNamespace(robot_xml=HERE/"assets/m20/mjcf/M20.xml", physics_dt=.001,
                           terrain_xml=HERE/"terrains/stairs_ascent_walls.xml",
                           stair_height=None, tread_depth=None, stair_count=None, stair_start=None,
                           spawn_xy=None, wall_alpha=.25, perception="clean", seed=42)
    r = Robot(policy, args)
    pose = policy.nominal + np.r_[np.linspace(-.01, .01, 12), np.zeros(4)]
    r.data.qpos[r.qids] = pose
    np.testing.assert_array_equal(r.data.qpos[r.qids], pose)
    r.data.qvel[r.vids] = np.arange(16)*.01
    action = np.linspace(-.1, .1, 16)
    r.control(action)
    target = policy.nominal + policy.action_scale*action
    vtarget = np.r_[np.zeros(12), policy.action_scale[12:]*action[12:]]
    expected = np.clip(policy.kp*(target-pose)+policy.kd*(vtarget-r.data.qvel[r.vids]), -policy.effort, policy.effort)
    np.testing.assert_allclose(r.data.ctrl[r.aids], expected)
    r.reset()
    # Move within the actual ray anticipation range, keeping the robot below the first riser.
    r.data.qpos[r.root] = -.4
    import mujoco
    mujoco.mj_forward(r.model, r.data)
    for _ in range(4):
        obs = r.observe(np.array([.5, 0., 0.]), np.zeros(16))
    assert obs[57] == 1.
    np.testing.assert_allclose(obs[58], .16, atol=1.e-6)
    assert obs[59] > 0.
    assert len(r.terrain_geoms) >= 12


def test_prone_stand_policy_and_both_resets(policy):
    from interaction import Interaction, PRONE_JOINT_POS
    import mujoco
    args = SimpleNamespace(robot_xml=HERE/"assets/m20/mjcf/M20.xml", physics_dt=.001,
                           terrain_xml=HERE/"terrains/flat.xml", stair_height=None,
                           tread_depth=None, stair_count=None, stair_start=None,
                           spawn_xy=None, wall_alpha=.25, perception="clean", seed=42)
    r = Robot(policy, args)
    session = Interaction(r)
    prone = r.data.qpos.copy()
    np.testing.assert_array_equal(r.data.qpos[r.qids], PRONE_JOINT_POS)
    session.event("c")
    assert session.mode == "prone"
    session.event("z")
    for _ in range(310):
        session.update()
        target = session.pose_target()
        for _ in range(20):
            r.control_pose(target)
            mujoco.mj_step(r.model, r.data)
        mujoco.mj_forward(r.model, r.data)
    assert session.mode == "ready"
    assert r.data.xpos[r.base_id, 2] > r.ground() + .42
    standing = session.standing_snapshot.copy()
    session.event("c")
    assert session.mode == "policy"
    for _ in range(50):
        obs = r.observe(np.array([.5, 0., 0.]), session.previous)
        action = session.action(obs)
        for _ in range(20):
            r.control(action)
            mujoco.mj_step(r.model, r.data)
        mujoco.mj_forward(r.model, r.data)
        session.previous = action
    session.event("r")
    assert session.mode == "ready" and not session.pending_policy
    np.testing.assert_array_equal(r.data.qpos, standing)
    np.testing.assert_array_equal(r.data.qvel, 0.)
    np.testing.assert_array_equal(session.previous, 0.)
    np.testing.assert_array_equal(r.perception.last, 0.)
    session.event("reset")
    assert session.mode == "prone" and session.standing_snapshot is None
    np.testing.assert_allclose(r.data.qpos, prone, atol=1.e-12)
    np.testing.assert_array_equal(r.data.qvel, 0.)
    # R remains available even before a stand has completed.
    session.event("r")
    assert session.mode == "ready"


def test_production_loop_keyboard_lifecycle(tmp_path, monkeypatch):
    from run import parse_args, run
    monkeypatch.setattr(sys, "argv", ["run.py", "--headless", "--duration", "8",
                                    "--terrain-xml", str(HERE/"terrains/flat.xml"),
                                    "--output", str(tmp_path/"interaction")])
    args = parse_args()
    events = {10: ["z"], 40: ["c"], 310: ["r"], 330: ["c"], 350: ["reset"], 351: ["c"]}
    assert run(args, event_source=lambda step: events.get(step, [])) == 0
    data = np.load(args.output/"trajectory.npz")
    assert all(data["mode"][:10] == "prone")
    assert data["mode"][40] == "standing"  # early C waits for completed standing
    assert data["mode"][300] == "policy"
    assert data["mode"][310] == "ready"
    np.testing.assert_array_equal(data["command"][310], 0.)
    np.testing.assert_array_equal(data["action"][310], 0.)
    np.testing.assert_array_equal(data["observation"][310, 41:57], 0.)
    assert all(data["mode"][350:] == "prone")


def test_goal_does_not_end_production_loop(tmp_path, monkeypatch):
    from run import parse_args, run
    import json
    monkeypatch.setattr(sys, "argv", ["run.py", "--headless", "--duration", "14",
                                    "--output", str(tmp_path/"goal_continuation"), "--require-goal"])
    args = parse_args()
    assert not args.stop_at_goal
    assert run(args) == 0
    summary = json.loads((args.output/"summary.json").read_text())
    assert summary["goal_reached"] and not summary["fell"]
    assert summary["status"] == "timeout"
    assert summary["control_steps"] == 700
    assert summary["goal_events"][0]["step"] < 699
