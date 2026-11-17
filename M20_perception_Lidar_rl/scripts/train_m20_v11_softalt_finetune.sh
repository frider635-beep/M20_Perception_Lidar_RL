#!/usr/bin/env bash
# Fine-tune V11 SoftAlt from an explicitly selected 63-D V11 checkpoint.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${V11_SOFTALT_CHECKPOINT:-}" ]]; then
  echo "Set V11_SOFTALT_CHECKPOINT to the selected ABI-compatible V11 checkpoint." >&2
  exit 2
fi
if [[ ! -f "$V11_SOFTALT_CHECKPOINT" ]]; then
  echo "V11 checkpoint does not exist: $V11_SOFTALT_CHECKPOINT" >&2
  exit 2
fi

source /home/ubuntu/miniconda3/bin/activate /home/ubuntu/miniconda3/envs/m20_wzh
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
export V11_SOFTALT_CHECKPOINT

exec /home/ubuntu/.holosoma_deps/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-SoftAlt-V11-v0 \
  --max_iterations 1000 \
  --num_envs 4096 \
  --headless \
  --run_name v11_softalt_finetune "$@"
