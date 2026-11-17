#!/usr/bin/env python3
"""Create a filtered migration snapshot without changing the source worktrees."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT/'tools/github_export_policy.json'
WEIGHT_SUFFIXES = {'.pt', '.pth', '.ckpt', '.onnx', '.safetensors', '.jit'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reason(path, policy, *, training):
    p = Path(path)
    if training and path == policy['checkpoint']:
        return None
    if training and p.parts[0] in policy['excluded_training_directories']:
        return p.parts[0]
    if any(part in policy['excluded_directory_names'] or part.endswith('.egg-info') for part in p.parts[:-1]):
        return 'cache/output directory'
    if p.name in policy['excluded_names'] or p.name.startswith('.env.'):
        return 'local/archive metadata'
    if p.suffix.lower() in policy['excluded_suffixes'] or 'tfevents' in p.name:
        return 'generated artifact or other weight'
    return None


def collect(root, policy, *, training):
    names = git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')
    files, skipped = {}, Counter()
    tracked = {}
    for entry in git(root, 'ls-files', '-s', '-z').decode().split('\0'):
        if entry:
            info, name = entry.split('\t', 1)
            tracked[name] = info.split()[0]
    for name in sorted(set(filter(None, names))):
        if name == 'deep_robotics_model':
            continue
        if not training and not any(name == scope or name.startswith(scope+'/') for scope in policy['validation_roots']):
            continue  # The validation checkout has a sparse index of other branches.
        why = reason(name, policy, training=training)
        if why:
            skipped[why] += 1
            continue
        source = root/name
        if not source.is_file():
            raise ValueError('Missing export source: '+str(source))
        if source.is_symlink():
            raise ValueError('Review symlink before publishing: '+str(source))
        if source.stat().st_size > policy['maximum_file_bytes']:
            raise ValueError('Export file exceeds size limit: '+str(source))
        if SECRET.search(source.read_bytes()):
            raise ValueError('Credential pattern detected; publication stopped: '+str(source))
        mode = '100755' if source.suffix == '.sh' else tracked.get(name, '100644')
        files[name] = (source, mode)
    if training and policy['checkpoint'] not in files:
        source = root/policy['checkpoint']
        if not source.is_file():
            raise ValueError('Selected checkpoint is missing')
        files[policy['checkpoint']] = (source, '100644')
    return files, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-root', type=Path, default=ROOT.parent/'mujoco_validation')
    parser.add_argument('--output', type=Path, required=True, help='A new directory outside both source worktrees')
    args = parser.parse_args()
    output, validation = args.output.resolve(), args.validation_root.resolve()
    if any(output == r or output.is_relative_to(r) for r in (ROOT, validation)):
        raise ValueError('Output must be outside the source worktrees')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing export: '+str(output))
    policy = json.loads(POLICY_PATH.read_text())
    provenance = json.loads((ROOT/'interview_prep/CHECKPOINT.json').read_text())
    if digest(ROOT/policy['checkpoint']) != provenance['sha256']:
        raise ValueError('Selected checkpoint SHA-256 does not match its provenance')
    training, skipped = collect(ROOT, policy, training=True)
    validators, skipped_validation = collect(validation, policy, training=False)
    for name in ('env.yaml', 'agent.yaml'):
        config = ROOT/'interview_prep/params'/name
        if digest(config) != provenance['config_sha256'][name]:
            raise ValueError('Selected checkpoint saved configuration mismatch: '+name)
        if config.relative_to(ROOT).as_posix() not in training:
            raise ValueError('Saved configuration was filtered out: '+name)
    submodule_sha = git(ROOT/'deep_robotics_model', 'rev-parse', 'HEAD').decode().strip()
    if git(ROOT/'deep_robotics_model', '-c', 'core.filemode=false', 'diff', '--name-only', 'HEAD').strip():
        raise ValueError('Tracked robot assets changed: review and publish their commit first')
    output.mkdir(parents=True)
    records = []
    for prefix, files in [('M20_perception_Lidar_rl', training), ('mujoco_validation', validators)]:
        for name, (source, mode) in files.items():
            target = output/prefix/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o755 if mode == '100755' else 0o644)
            records.append(dict(path=target.relative_to(output).as_posix(), bytes=target.stat().st_size, sha256=digest(target)))
    shutil.copyfile(ROOT/'tools/github_snapshot_README.md', output/'README.md')
    shutil.copyfile(ROOT/'tools/github_snapshot.gitignore', output/'.gitignore')
    shutil.copyfile(ROOT/'LICENSE', output/'LICENSE')
    (output/'.gitattributes').write_text('* text=auto\n*.sh text eol=LF\n*.pt -text\n*.npz -text\n')
    (output/'.gitmodules').write_text('[submodule "deep_robotics_model"]\n\tpath = M20_perception_Lidar_rl/deep_robotics_model\n\turl = https://github.com/DeepRoboticsLab/deep_robotics_model.git\n')
    weights = sorted(p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES)
    if weights != ['M20_perception_Lidar_rl/'+policy['checkpoint']]:
        raise ValueError('Export must contain exactly the selected checkpoint: '+repr(weights))
    report = dict(format_version=1, source_head=git(ROOT, 'rev-parse', 'HEAD').decode().strip(),
                  validation_head=git(validation, 'rev-parse', 'HEAD').decode().strip(),
                  robot_submodule_commit=submodule_sha, policy=policy,
                  selected_checkpoint=provenance, files=records,
                  filtered_training=dict(skipped), filtered_validation=dict(skipped_validation),
                  local_source_files_deleted=0, contains_original_git_history=False)
    (output/'PUBLISH_MANIFEST.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), files=len(records), total_bytes=sum(f['bytes'] for f in records),
                          weights=weights, robot_submodule_commit=submodule_sha, local_source_files_deleted=0), indent=2))


if __name__ == '__main__':
    main()
