"""Live terminal view of a guarded recovery session; closing it never stops training."""
import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    ansi = re.compile(r'\x1b\[[0-9;]*m')
    while True:
        try:
            state = json.loads((args.session/'status.json').read_text())
        except (OSError, json.JSONDecodeError):
            state = dict(status='initializing', completed_updates=0, total_updates=30000)
        lines = []
        try:
            with args.log.open('rb') as stream:
                stream.seek(max(0, args.log.stat().st_size - 18000))
                lines = ansi.sub('', stream.read().decode(errors='replace')).splitlines()[-24:]
        except OSError:
            pass
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                                  '--format=csv,noheader'], capture_output=True, text=True, timeout=3).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            gpu = 'unavailable'
        disk = shutil.disk_usage(args.session).free/2**30
        print('\033[2J\033[H', end='')
        print(f'M20 RECOVERY | {state.get("total_updates", 30000)} additional PPO updates '
              f'| behavior evaluation every {state.get("segment_updates", 100)} updates')
        print(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %Z'))
        print(f"Status: {state.get('status')} | Completed: {state.get('completed_updates', 0)} / {state.get('total_updates', 30000)}")
        print(f"GPU util / used / total / temp: {gpu} | Disk free: {disk:.1f} GiB")
        print(f"Repair acceptance: {state.get('repair_passed', False)} | PID: {state.get('pid', 'starting')}")
        print(f"Behavior failure policy: {state.get('behavior_failure_policy', 'historical stop policy')}")
        print(f"Latest checkpoint: {state.get('last_checkpoint') or state.get('candidate_checkpoint') or 'starting'}")
        print(f"Last retained: {state.get('last_retained_checkpoint') or 'original model_35300.pt'}")
        print(f"Session: {args.session}")
        if state.get('error'):
            print('STOP REASON:', state['error'])
        for reason in state.get('capability_regressions', [])[:5]:
            print('CAPABILITY REGRESSION (training continues):', reason)
        rejection = state.get('rejection', {})
        for reason in state.get('last_retention_failures', rejection.get('retention_failures', []))[:5]:
            print('RETENTION FAILURE:', reason)
        print('--- training / evaluation log ---')
        print('\n'.join(lines), flush=True)
        time.sleep(3)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
