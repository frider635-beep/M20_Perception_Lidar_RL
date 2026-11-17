#!/usr/bin/env python3
"""Generate V13.5 report from one explicit run directory (no latest-run guessing)."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/rl_training"))
import importlib.util
_module_path = Path(__file__).resolve().parents[2] / "source/rl_training/rl_training/v135_training_report.py"
_spec = importlib.util.spec_from_file_location("v135_training_report", _module_path)
_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_report)
build_report = _report.build_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--status", default="manual_snapshot")
    args = parser.parse_args()
    print(build_report(args.run_dir, args.status))
