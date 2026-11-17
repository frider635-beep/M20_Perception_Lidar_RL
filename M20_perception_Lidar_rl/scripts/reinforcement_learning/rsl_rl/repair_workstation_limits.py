"""Shared repair-task limits; no Isaac, Torch, CUDA or simulation imports."""
import argparse
import json
import os
from pathlib import Path

TASKS = ('M20-Stair-V11.1-EntryFlatRepair-v0', 'M20-Stair-V11.1-AdjacentRepair-v0',
         'M20-Stair-V11.1-AscentRecovery-v0')
LEDGER = Path(__file__).resolve().parents[2]/'repair_workstation_limits.json'


def validate(num_envs, updates, *, distributed=False, ledger=None):
    ledger = Path(ledger or os.environ.get('M20_REPAIR_LIMITS', LEDGER))
    cfg = json.loads(ledger.read_text())
    for key in ('default_num_envs', 'max_num_envs', 'default_updates', 'max_updates_per_stage'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'Invalid workstation ledger: {key}')
    n = cfg['default_num_envs'] if num_envs is None else num_envs
    k = cfg['default_updates'] if updates is None else updates
    if distributed:
        raise ValueError('Repair stages on this workstation use one GPU process')
    if type(n) is not int or not 1 <= n <= cfg['max_num_envs']:
        raise ValueError(f"Repair environments must be 1..{cfg['max_num_envs']} (requested {n}); outside the configured workstation limit")
    if type(k) is not int or not 1 <= k <= cfg['max_updates_per_stage']:
        raise ValueError(f"Repair stages are limited to {cfg['max_updates_per_stage']} updates (requested {k}); validate MuJoCo before starting another stage")
    return n, k


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--num_envs', type=int)
    parser.add_argument('--max_iterations', type=int)
    parser.add_argument('--distributed', action='store_true')
    args, _ = parser.parse_known_args()
    try:
        print(*validate(args.num_envs, args.max_iterations, distributed=args.distributed))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
