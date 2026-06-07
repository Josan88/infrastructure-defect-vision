#!/bin/bash
#SBATCH --job-name=rtdetr_v5_iter
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/achin/COS40007-Group
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=102775928@students.swinburne.edu.my
#SBATCH --signal=B:USR1@120
# SIGUSR1 120s before wall-time kill - gives Ultralytics a chance to flush
# last.pt cleanly. (Ultralytics itself listens for SIGTERM, not SIGUSR1, but
# the 120s buffer is enough for normal periodic checkpointing to run.)

set -euo pipefail

# --- 1. Environment ---
# Load python module so the venv's python can resolve shared libs.
# The venv itself is activated AFTER module load so its bin/ is on PATH first.
module load python/3.10.8
module load gcc/12.2.0

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

# --- 4. Move executed notebook back to repo ---
if [ -f /tmp/Train2-jason.executed.ipynb ]; then
  mv /tmp/Train2-jason.executed.ipynb Train2-jason.ipynb
  echo ""
  echo "Notebook updated: $(stat -c %s Train2-jason.ipynb 2>/dev/null) B"
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
