#!/usr/bin/env bash
# Launch the V13.3 native-63D actor and both critics from random initialization.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

M20_PYTHON="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20_wzh/bin/python}"
[[ -x "$M20_PYTHON" ]] || { echo "Training Python not found: $M20_PYTHON" >&2; exit 2; }

TASK="M20-Stair-V13.3-V11FD-FromScratch-v0"
SEED="${V133_SEED:-42}"
MAX_ITERATIONS="${V133_MAX_ITERATIONS:-20000}"
NUM_ENVS="${V133_NUM_ENVS:-8192}"
RUN_NAME="${V133_RUN_NAME:-v133_native63_adjacentprogress_clawback_officialdescent_rollout32_from_scratch_20k}"

if [[ -n "${V133_MOTOR_PRIOR_CHECKPOINT:-}" ]]; then
  echo "V13.3 from-scratch refuses V133_MOTOR_PRIOR_CHECKPOINT." >&2
  exit 2
fi
if [[ "$MAX_ITERATIONS" != "20000" ]]; then
  echo "V13.3 authorized training budget must remain 20000 iterations." >&2
  exit 2
fi

export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
"$M20_PYTHON" scripts/tools/test_m20_v133_adjacent_transition.py
"$M20_PYTHON" scripts/tools/test_m20_v133_runtime_smoke.py --headless --num_envs 16 --steps 8

SOURCE_MANIFEST_TMP="$(mktemp /tmp/v133-source-manifest.XXXXXX)"
trap 'rm -f "$SOURCE_MANIFEST_TMP"' EXIT
declare -A EXISTING_RUNS=()
for candidate in logs/rsl_rl/M20-V13.3-V11FD-FromScratch/*_"$RUN_NAME"; do
  [[ -d "$candidate" ]] && EXISTING_RUNS["$candidate"]=1
done
mapfile -t SOURCE_FILES < <(find source/rl_training/rl_training -type f -name '*.py' -print | sort)
SOURCE_FILES+=(scripts/reinforcement_learning/rsl_rl/train.py scripts/tools/evaluate_m20_v133_frozen.py)
sha256sum "${SOURCE_FILES[@]}" > "$SOURCE_MANIFEST_TMP"

env -u V133_MOTOR_PRIOR_CHECKPOINT "$M20_PYTHON" \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --seed "$SEED" \
  --max_iterations "$MAX_ITERATIONS" \
  --num_envs "$NUM_ENVS" \
  --headless \
  --run_name "$RUN_NAME"

NEW_RUNS=()
for candidate in logs/rsl_rl/M20-V13.3-V11FD-FromScratch/*_"$RUN_NAME"; do
  if [[ -d "$candidate" && -z "${EXISTING_RUNS[$candidate]+present}" ]]; then
    NEW_RUNS+=("$candidate")
  fi
done
[[ "${#NEW_RUNS[@]}" == "1" ]] || {
  echo "V13.3 expected exactly one new run directory, found ${#NEW_RUNS[@]}." >&2
  exit 3
}
RUN_DIR="${NEW_RUNS[0]}"
cp "$SOURCE_MANIFEST_TMP" "$RUN_DIR/source_manifest.sha256"
"$M20_PYTHON" scripts/tools/evaluate_m20_v133_frozen.py \
  --checkpoint-dir "$RUN_DIR" \
  --source-manifest "$RUN_DIR/source_manifest.sha256" \
  --output "$RUN_DIR/frozen_evaluation" \
  --headless
