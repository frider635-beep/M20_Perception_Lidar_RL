#!/usr/bin/env bash
# Launch M20 V11 from scratch with the project-standard hssim environment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/ubuntu/.holosoma_deps/miniconda3/bin/activate /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"

exec /home/ubuntu/.holosoma_deps/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-Continuous-Landing-V11-v0 \
  --max_iterations 30000 \
  --num_envs 8192 \
  --headless \
  --run_name v11_from_scratch "$@"
