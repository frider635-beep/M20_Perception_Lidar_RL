#!/usr/bin/env python3
"""Static contract trace for the M20 interview-prep notes.

This script intentionally avoids importing Isaac Lab.  It scans source text for
the task IDs, ABI gates, feature contracts, and teacher/actor boundaries used by
the interview roadmap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Probe:
    path: str
    pattern: str
    label: str


PROBES = [
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py",
        r'M20-Stair-Continuous-Landing-V11-v0',
        "V11 gym task is registered",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py",
        r'M20-Stair-V12-V9Teacher-6D-v0',
        "V12 gym task is registered",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_continuous_env_cfg.py",
        r'front_completion_mode:\s*str\s*=\s*"alternating_risers"',
        "V11 defaults to alternating risers",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_continuous_env_cfg.py",
        r'feature_contract:\s*str\s*=\s*"v11_signed_edges_v2"',
        "V11/V11-FD signed-edge feature contract is present",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_continuous_env_cfg.py",
        r'self\.observations\.policy\.height_scan\s*=\s*ObsTerm\(\s*func=mdp\.v11_actor_perception',
        "V11 actor observation is the 6D two-riser term",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_continuous_env_cfg.py",
        r'self\.rewards\.first_step_success\s*=\s*None',
        "V10 first-step reward is disconnected in V11",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/perceptive_spatial.py",
        r'def extract_two_risers_points\(',
        "Two-riser extractor exists",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py",
        r'risers\s*=\s*extract_two_risers_points\(',
        "V11 state consumes the two-riser extractor",
    ),
    Probe(
        "source/rl_training/rl_training/v11_continuous_runner.py",
        r'if \(actor\.obs_dim, action_dim\) != \(63, 16\):',
        "V11 runner enforces 63D actor and 16D action",
    ),
    Probe(
        "source/rl_training/rl_training/v11_continuous_runner.py",
        r'Actor has no FL/FR-specific action path',
        "V11 manifest states no FL/FR-specific action path",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v12_teacher_env_cfg.py",
        r'self\.v11_perception\.front_completion_mode\s*=\s*"bilateral_same_riser"',
        "V12 fixes bilateral same-riser completion",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v12_teacher_env_cfg.py",
        r'self\.v11_perception\.feature_contract\s*=\s*"v11_positive_risers_v1"',
        "V12 keeps the positive-riser feature contract",
    ),
    Probe(
        "source/rl_training/rl_training/v12_teacher_moe.py",
        r'class MoeLayer\(nn\.Module\):',
        "V12 soft MoE layer exists",
    ),
    Probe(
        "source/rl_training/rl_training/v12_teacher_runner.py",
        r'num_experts != 4',
        "V12 runner gates the four-expert MoE architecture",
    ),
    Probe(
        "source/rl_training/rl_training/v12_teacher_runner.py",
        r'if \(actor\.obs_dim, action_dim\) != \(63, 16\):',
        "V12 runner enforces 63D actor and 16D action",
    ),
    Probe(
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v12_teacher_rewards.py",
        r'The scanner is never registered in the V12 Actor observation group',
        "V12 reward docs state PGTT is not actor input",
    ),
    Probe(
        "scripts/train_m20_v12_from_scratch.sh",
        r'--task M20-Stair-V12-V9Teacher-6D-v0',
        "V12 launcher targets the V12 teacher task",
    ),
]


def find_line(text: str, pattern: str) -> int | None:
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
    match = regex.search(text)
    if match is None:
        return None
    return text.count("\n", 0, match.start()) + 1


def main() -> int:
    results = []
    ok = True
    for probe in PROBES:
        path = ROOT / probe.path
        if not path.exists():
            ok = False
            results.append({"ok": False, "label": probe.label, "path": probe.path, "line": None, "reason": "missing file"})
            continue
        text = path.read_text(encoding="utf-8")
        line = find_line(text, probe.pattern)
        if line is None:
            ok = False
            results.append({"ok": False, "label": probe.label, "path": probe.path, "line": None, "reason": "pattern not found"})
        else:
            results.append({"ok": True, "label": probe.label, "path": probe.path, "line": line})

    print(f"M20_INTERVIEW_TRACE={'PASS' if ok else 'FAIL'}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
