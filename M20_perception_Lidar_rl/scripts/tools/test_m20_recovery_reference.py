"""Physical-label anchor checks; no simulator and no optimizer updates."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'source/rl_training/rl_training'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load('m20_reference_math', SOURCE/'recovery_reference.py')
builder = load('m20_reference_builder', ROOT/'scripts/tools/m20_build_recovery_replay.py')


def loss(prediction, scene, observations=None, target=None, **kwargs):
    observations = torch.zeros(len(prediction), 63) if observations is None else observations
    target = torch.zeros_like(prediction) if target is None else target
    return reference.reference_anchor_loss(prediction, target, observations, torch.tensor(scene),
        .04, 2., **kwargs)


def write_replay(path, *, source_sha='teacher', labels=True):
    data = dict(observation=np.zeros((4, 63), dtype=np.float32))
    if labels:
        data.update(scene=np.array([0, 0, 1, 1]), phase=np.array([1, 2, 1, 2]), case_id=np.array([0, 0, 1, 1]))
    np.savez_compressed(path, **data)
    metadata = dict(schema_version=2, samples=4, source_sha256=source_sha,
        replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.json').write_text(json.dumps(metadata))


def test_zero_perception_on_ascent_never_becomes_flat_anchor():
    predicted = torch.ones(2, 16, requires_grad=True)
    loss(predicted, [0, 1]).backward()
    for index in reference.ALL_LEG_SWING:
        assert predicted.grad[0, index] == pytest.approx(.04*predicted.grad[1, index].item())
    assert predicted.grad[0, 0] == predicted.grad[1, 0]


def test_actor_visible_ascent_relaxes_front_and_rear_without_flat_label():
    predicted = torch.ones(2, 16, requires_grad=True)
    observations = torch.zeros(2, 63)
    observations[1, 57:59] = 1
    target = torch.zeros_like(predicted, requires_grad=True)
    loss(predicted, [1, 1], observations, target).backward()
    assert target.grad is None
    for index in reference.FRONT_SWING:
        assert predicted.grad[1, index] == pytest.approx(.04*predicted.grad[0, index].item())
    for index in (7, 8, 10, 11):
        assert predicted.grad[1, index] == pytest.approx(.35*predicted.grad[0, index].item())
    assert predicted.grad[1, 12] == predicted.grad[0, 12]


def test_tolerance_does_not_pull_small_necessary_corrections_back():
    predicted = torch.full((2, 16), .03, requires_grad=True)
    value = loss(predicted, [0, 1])
    value.backward()
    assert value.item() == 0
    assert not predicted.grad.any()


def test_flat_wheel_common_drive_is_stronger_than_differential_bias():
    common, differential = torch.zeros(1, 16), torch.zeros(1, 16)
    common[:, 12:] = 1
    differential[:, 12:] = torch.tensor([1., -1., 1., -1.])
    assert loss(differential, [0]).item() == pytest.approx(.08*loss(common, [0]).item())
    # All-wheel drive on stairs retains its original scale.
    assert loss(common, [0]).item() == pytest.approx(loss(common, [1]).item())


def test_missing_labels_and_wrong_teacher_are_rejected(tmp_path):
    path = tmp_path/'replay.npz'
    write_replay(path, labels=False)
    with pytest.raises(ValueError, match='physical scene'):
        reference.load_labeled_replay(path)
    write_replay(path)
    with pytest.raises(ValueError, match='frozen teacher'):
        reference.load_labeled_replay(path, 'different')
    obs, scene, phase, case_id = reference.load_labeled_replay(path, 'teacher')
    assert obs.shape == (4, 63)
    assert set(scene) == {0, 1}
    assert set(phase) == {1, 2}
    assert set(case_id) == {0, 1}


def test_metadata_detects_modified_replay(tmp_path):
    path = tmp_path/'replay.npz'
    write_replay(path)
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        reference.load_labeled_replay(path)


def test_builder_labels_physical_cases_and_excludes_nonpolicy(tmp_path):
    checkpoint = tmp_path/'teacher.pt'
    checkpoint.write_bytes(b'fixture checkpoint identity')
    source_sha = builder.sha256(checkpoint)
    for terrain, tread in [('flat', None), ('ascent', .26)]:
        case = tmp_path/'baseline'/terrain
        case.mkdir(parents=True)
        (case/'manifest.json').write_text(json.dumps(dict(schema_version=2,
            checkpoint_sha256=source_sha, physics_enabled=True, profile='z_c_8')))
        (case/'results.json').write_text(json.dumps([dict(terrain=terrain, tread=tread, command_speed=.7)]))
        np.savez_compressed(case/f'{terrain}_{tread}_0.7.npz',
            observation=np.zeros((3, 63), dtype=np.float32), policy_active=np.array([False, True, True]),
            phase=np.array([0, 1, 2]))
    output = tmp_path/'reference.npz'
    metadata = builder.build(tmp_path/'baseline', checkpoint, output)
    obs, scene, phase, _ = reference.load_labeled_replay(output, source_sha)
    assert metadata['samples'] == 4
    assert obs.shape == (4, 63)
    assert sorted(scene) == [0, 0, 1, 1]
    assert sorted(phase) == [1, 1, 2, 2]
    with pytest.raises(FileExistsError):
        builder.build(tmp_path/'baseline', checkpoint, output)


def test_real_rsl_constructor_freezes_reference_and_anchor_gradients_are_finite(tmp_path):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    # Import the runner without importing the repository's Isaac task registry.
    package_name = '_m20_cpu_runner_test'
    package = types.ModuleType(package_name)
    package.__path__ = [str(SOURCE)]
    sys.modules[package_name] = package
    runner = load(package_name+'.v111_recovery_runner', SOURCE/'v111_recovery_runner.py')
    observations = TensorDict({'policy': torch.zeros(4, 63), 'critic': torch.zeros(4, 568)}, batch_size=[4])
    groups = {'actor': ['policy'], 'critic': ['critic']}
    actor = MLPModel(observations, groups, 'actor', 16, hidden_dims=[512, 256, 128],
        distribution_cfg={'class_name': 'rsl_rl.modules.distribution:GaussianDistribution',
                          'init_std': 1., 'std_type': 'log'})
    critic = MLPModel(observations, groups, 'critic', 1, hidden_dims=[512, 256, 128])
    checkpoint = ROOT/'interview_prep/model_35300.pt'
    replay = tmp_path/'replay.npz'
    write_replay(replay, source_sha=builder.sha256(checkpoint))
    algorithm = runner.RecoveryPPO(actor, critic, None, teacher_checkpoint=str(checkpoint),
        replay_path=str(replay), anchor_weight=2., anchor_front_stair_weight=.04, anchor_batch_size=8)
    for name, parameter in algorithm.actor.named_parameters():
        if name.startswith(('mlp.0.', 'mlp.2.', 'distribution.')):
            assert not parameter.requires_grad
    assert not any(parameter.requires_grad for parameter in algorithm.teacher.parameters())
    before = {name: parameter.detach().clone() for name, parameter in algorithm.actor.named_parameters()}
    algorithm._anchor_step(algorithm.optimizer, (), {})
    assert len(algorithm.anchor_losses) == 1
    gradients = [parameter.grad for parameter in algorithm.actor.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    for name, parameter in algorithm.actor.named_parameters():
        assert torch.equal(before[name], parameter), 'Gradient check must not update model weights'
