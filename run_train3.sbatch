#!/bin/bash
#SBATCH --job-name=rtdetr_v7_train3
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/achin/COS40007-Group
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --signal=B:TERM@120

set -euo pipefail

REPO_DIR="/home/achin/COS40007-Group"
NOTEBOOK="Train3.ipynb"
RUN="rtdetr_l_v7"
PROJECT_DIR="runs/defect_detection"
JOB_ID="${SLURM_JOB_ID:-manual}"
TMP_NOTEBOOK="/tmp/${NOTEBOOK%.ipynb}.${JOB_ID}.executed.ipynb"

module load gcc/12.2.0 2>/dev/null || true
# module unload python/3.10.8 2>/dev/null || true

cd "$REPO_DIR"
source .venv/bin/activate
mkdir -p logs

echo "============================================================"
echo "sbatch job $JOB_ID on $(hostname)"
echo "Started:   $(date)"
echo "Notebook:  $NOTEBOOK"
echo "Run:       $RUN"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed')"
echo "CUDA:      $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)')"
echo "Python:    $(python --version) ($(which python))"
echo "Git HEAD:  $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "============================================================"

# Write executed output to /tmp first to avoid same-file nbconvert deadlocks.
set +e
jupyter nbconvert --to notebook --execute "$NOTEBOOK" \
  --output "$TMP_NOTEBOOK" \
  --ExecutePreprocessor.timeout=-1 \
  2>&1 | tee -a "logs/nbconvert-${JOB_ID}.log"
NB_EXIT=${PIPESTATUS[0]}
set -e

if [ -f "$TMP_NOTEBOOK" ]; then
  mv "$TMP_NOTEBOOK" "$NOTEBOOK"
  echo ""
  echo "Notebook updated: $(stat -c %s "$NOTEBOOK" 2>/dev/null) B"
else
  echo "WARNING: no executed notebook produced at $TMP_NOTEBOOK"
  echo "Check logs/nbconvert-${JOB_ID}.log for errors"
fi

if [ "$NB_EXIT" -ne 0 ]; then
  echo ""
  echo "ERROR: nbconvert exited with code $NB_EXIT"
  echo "The iter log and runs/ outputs are the source of truth. Re-submit to resume."
fi

echo ""
echo "============================================================"
echo "Iter log: $PROJECT_DIR/${RUN}_iterlog.csv"
echo "============================================================"
cat "$PROJECT_DIR/${RUN}_iterlog.csv" 2>/dev/null || echo "(no iter log found)"

echo ""
echo "============================================================"
echo "Weights"
echo "============================================================"
for d in "$PROJECT_DIR/${RUN}_iter"*/weights; do
  [ -d "$d" ] || continue
  run_dir=${d%/weights}
  if [ -f "$d/best.pt" ]; then
    echo "  best: $d/best.pt ($(stat -c %s "$d/best.pt") B)"
  else
    echo "  best: $run_dir (not completed)"
  fi
  if [ -f "$d/last.pt" ]; then
    echo "  last: $d/last.pt ($(stat -c %s "$d/last.pt") B)"
  fi
done

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"

exit "$NB_EXIT"
