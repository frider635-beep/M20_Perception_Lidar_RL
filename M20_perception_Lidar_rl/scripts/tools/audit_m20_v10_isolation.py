#!/usr/bin/env python3
"""Source-level privilege and prohibited-architecture audit for V10."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTOR_FILES = [
    ROOT / "source/rl_training/rl_training/v10_unified_policy.py",
    ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v10_unified_env_cfg.py",
]
PROHIBITED = ("spatial_lift", "BridgeActor", "residual", "gait_phase")
SIM_ONLY = ("body_pos_w", "net_forces_w", "vertical_contact", "fr_z")


def main():
    policy = ACTOR_FILES[0].read_text()
    env_cfg = ACTOR_FILES[1].read_text()
    actor_obs_block = env_cfg.split("# Critic remains asymmetric:", 1)[0]
    prohibited_hits = [term for term in PROHIBITED if term in policy or term in env_cfg]
    if re.search(r"\b372D?\b", policy) or re.search(r"\b372D?\b", env_cfg):
        prohibited_hits.append("372D")
    sim_hits = [term for term in SIM_ONLY if term in actor_obs_block]
    graph_ok = "observation[..., 58:60]" in policy and "self.support_head(latent)" in policy
    print(f"PROHIBITED_V10_HITS={prohibited_hits}")
    print(f"ACTOR_SIM_ONLY_HITS={sim_hits}")
    print(f"TEST_D={'PASS' if not prohibited_hits and not sim_hits and graph_ok else 'FAIL'}")
    print(f"SIM_ONLY_Z_ISOLATION={'PASS' if not sim_hits else 'FAIL'}")
    print(f"PRIVILEGED_CRITIC={'PASS' if 'v10_critic_teacher' in env_cfg and 'v10_critic_pgtt' in env_cfg else 'FAIL'}")
    raise SystemExit(0 if not prohibited_hits and not sim_hits and graph_ok else 1)


if __name__ == "__main__":
    main()
