#!/bin/bash
#SBATCH --job-name=rtdetr_v5_iter
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
# HPC enforces a 2.5h wall-clock cap per submission. Each submission
# completes ~2-3 iters of the 6-iter run; re-submit to resume (the iter
# loop inside the notebook is fully resumable - completed iters are
# skipped, in-progress iters retrain from prev last.pt).
#SBATCH --chdir=/home/achin/COS40007-Group
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=102775928@students.swinburne.edu.my
#SBATCH --signal=B:TERM@120
# SIGTERM is sent 120s before the time-limit hard kill. Bash's default handler
# propagates TERM to children in the same process group, so the python child
# running jupyter nbconvert also gets TERM. Ultralytics 8.4.x installs a SIGTERM
# handler that saves last.pt and exits cleanly, giving the 120s buffer to flush
# the checkpoint to disk.

set -euo pipefail

# --- 1. Environment ---
# Load python module so the venv's python can resolve shared libs.
# The venv itself is activated AFTER module load so its bin/ is on PATH first.
module load python/3.10.8
# gcc is only needed for pip install of C extensions; harmless at training time
# but if the module is missing we still want the job to proceed.
module load gcc/12.2.0 2>/dev/null || true

# cd to repo (also done via --chdir, but explicit for clarity)
cd /home/achin/COS40007-Group
source .venv/bin/activate

# --- 2. Diagnostics ---
echo "============================================================"
echo "sbatch job $SLURM_JOB_ID on $(hostname)"
echo "Started:   $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed')"
echo "CUDA:      $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)')"
echo "Python:    $(python --version)  ($(which python))"
echo "Git HEAD:  $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "============================================================"
echo ""

# --- 3. Run the notebook (resumable) ---
mkdir -p logs

# Use /tmp output then atomic mv to avoid same-file nbconvert deadlock.
# The iter loop inside the notebook will skip any iter whose last.pt AND
# iterlog row both exist, so this script is safe to re-submit any number
# of times - the only cost of an early kill is the in-progress iter.
jupyter nbconvert --to notebook --execute Train2-jason.ipynb \
  --output /tmp/Train2-jason.executed.ipynb \
  2>&1 | tee -a logs/nbconvert-$SLURM_JOB_ID.log
NB_EXIT=${PIPESTATUS[0]}

# --- 4. Move executed notebook back to repo ---
# If a partial .ipynb was produced (e.g., a downstream cell failed after the
# iter loop succeeded), still move it - cells 1-12 outputs are still useful.
NB_OUT=/tmp/Train2-jason.executed.ipynb
if [ -f "$NB_OUT" ]; then
  mv "$NB_OUT" Train2-jason.ipynb
  echo ""
  echo "Notebook updated: $(stat -c %s Train2-jason.ipynb 2>/dev/null) B"
else
  echo "WARNING: no executed notebook produced at $NB_OUT"
  echo "Check logs/nbconvert-$SLURM_JOB_ID.log for errors"
fi

if [ "$NB_EXIT" -ne 0 ]; then
  echo ""
  echo "ERROR: nbconvert exited with code $NB_EXIT"
  echo "The .ipynb above may be partial. The iter log + runs/ on disk are the source of truth."
  echo "Re-submit to resume (completed iters will be skipped):"
  echo "  sbatch run_v5.sbatch"
  # Don't 'exit $NB_EXIT' here - let the final summary run so the user can
  # see how many iters completed in the morning.
fi

# --- 5. Final summary ---
echo ""
echo "============================================================"
echo "Final iterlog: runs/defect_detection/rtdetr_l_v5_iterlog.csv"
echo "============================================================"
cat runs/defect_detection/rtdetr_l_v5_iterlog.csv 2>/dev/null || echo "(no iterlog found)"

echo ""
echo "============================================================"
echo "Per-iter best weights:"
echo "============================================================"
for n in 0 1 2 3 4 5; do
  d="runs/defect_detection/rtdetr_l_v5_iter${n}"
  if [ -f "$d/weights/best.pt" ]; then
    sz=$(stat -c %s "$d/weights/best.pt")
    echo "  iter $n: $d/weights/best.pt ($sz B)"
  else
    echo "  iter $n: (not completed)"
  fi
done

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
