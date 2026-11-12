#!/usr/bin/env bash
# 5-fold cross-validation of one experiment: make the folds, then for each fold train, evaluate on the
# MSD test split and score the external dataset (atlas box, atlas box + largest piece, MSD-style box +
# largest piece); finally the experiment's results table and the ablation table from MLflow.
#
# The MSD test split and the external dataset are test-only: no fold trains on them or picks weights
# with them. Everything is logged to MLflow (runs/mlflow.db).
#
# Resumable: a fold that finished training (last.pt present) is not trained again; an interrupted one
# continues from its checkpoint. Evaluations always run again (they take a minute or two).
#
# Usage (from the project root, Linux or macOS):
#     scripts/run_cv.sh                                        # configs/baseline.yaml, named "3D U-Net"
#     scripts/run_cv.sh configs/random_trim.yaml "Random position"
#     DRY_RUN=1 scripts/run_cv.sh                              # print the commands only
# Settings (environment variables):
#     FOLDS=5   EXTERNAL=data/mri_dataset   VENV=.venv   LOG=runs/<experiment>/cv_<date>.log
# Research use only, not for diagnosis.
set -euo pipefail

CONFIG="${1:-configs/baseline.yaml}"
NAME="${2:-3D U-Net}"
FOLDS="${FOLDS:-5}"
EXTERNAL="${EXTERNAL:-data/mri_dataset}"
VENV="${VENV:-.venv}"
DRY_RUN="${DRY_RUN:-0}"

cd "$(dirname "$0")/.."    # the project root, wherever the script is started from
if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "No virtual environment at $VENV; using $(command -v python)" >&2
fi
export PYTHONPATH=src
export MLFLOW_DISABLE_AGENT_HINT=1

# the experiment folder is the config's out_dir; each fold goes to <out_dir>/fold<k>
RUN_DIR="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["train"]["out_dir"])' "$CONFIG")"
EXPERIMENT="$(basename "$RUN_DIR")"
if [[ "$DRY_RUN" != "1" ]]; then
    LOG="${LOG:-$RUN_DIR/cv_$(date +%Y%m%d_%H%M%S).log}"
    mkdir -p "$(dirname "$LOG")"
    exec > >(tee -a "$LOG") 2>&1    # everything below also goes to the log file
    echo "Log: $LOG"
fi

run() {    # print a command with the time, then run it (or only print it with DRY_RUN=1)
    echo "[$(date '+%H:%M:%S')] $*"
    if [[ "$DRY_RUN" != "1" ]]; then
        "$@"
    fi
}

echo "Experiment $EXPERIMENT ($CONFIG), $FOLDS folds, results in $RUN_DIR/fold<k>"
run python -m hippo.data --folds "$FOLDS"

if [[ ! -d "$EXTERNAL" ]]; then
    echo "External dataset not found at $EXTERNAL: external scoring is skipped (set EXTERNAL=...)" >&2
fi

for ((k = 0; k < FOLDS; k++)); do
    fold="$RUN_DIR/fold$k"
    echo "=== $EXPERIMENT fold $k of $((FOLDS - 1)) ==="
    if [[ -f "$fold/last.pt" ]]; then
        echo "Fold $k already trained ($fold/last.pt); not training again"
    else
        run python -m hippo.train --config "$CONFIG" --fold "$k" --resume
    fi
    run python -m hippo.evaluate --run "$fold"
    if [[ -d "$EXTERNAL" ]]; then
        run python -m hippo.external --src "$EXTERNAL" --run "$fold"
        run python -m hippo.external --src "$EXTERNAL" --run "$fold" --largest-piece
        # own crop folder per experiment and fold: the MSD-style box depends on that model's first prediction
        run python -m hippo.external --src "$EXTERNAL" --run "$fold" --msd-crop --largest-piece \
            --out "data/external_msd_crop_${EXPERIMENT}_fold$k"
    fi
done

run python -m hippo.results --experiment "$RUN_DIR" --name "$NAME"
run python -m hippo.results --ablation
echo "Done: $RUN_DIR/results/results.md, $RUN_DIR/results/bland_altman.png, runs/ablation.md"
