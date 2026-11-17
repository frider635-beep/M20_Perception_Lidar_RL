"""Portable recovery assets resolved from the repository and explicit overrides."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'data/m20_recovery'


def resolve_assets():
    metadata = json.loads((DATA/'reference_observations_v2.json').read_text())
    def path(value):
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
    teacher = path(os.environ.get('M20_RECOVERY_TEACHER') or os.environ.get('M20_BASE_CHECKPOINT') or metadata['source_checkpoint'])
    replay = path(os.environ.get('M20_RECOVERY_REPLAY') or metadata['replay_path'])
    return teacher, replay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    teacher, replay = resolve_assets()
    if args.teacher:
        print(teacher)
    elif args.replay:
        print(replay)
    else:
        print(json.dumps(dict(teacher=str(teacher), replay=str(replay))))
