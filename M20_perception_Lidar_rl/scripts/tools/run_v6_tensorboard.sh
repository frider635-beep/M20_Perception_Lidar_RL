#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/ubuntu/m20_stair_official20k
PORT=${1:-6007}
RUN_ROOT=${PROJECT}/rl_training/logs/rsl_rl/m20_stair_official20k_v1

source /home/ubuntu/.holosoma_deps/miniconda3/bin/activate /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim

if [[ $# -ge 2 ]]; then
  LOGDIR=$2
else
  runs=("${RUN_ROOT}"/20*)
  if [[ ! -d "${runs[-1]}" ]]; then
    echo "No V6 TensorBoard run found under ${RUN_ROOT}" >&2
    exit 1
  fi
  LOGDIR=${runs[-1]}
fi

exec python -m tensorboard.main \
  --logdir "${LOGDIR}" \
  --host 0.0.0.0 \
  --port "${PORT}"
