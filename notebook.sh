#!/bin/bash
#SBATCH --job-name=jupyter-notebook
#SBATCH --partition=milan-gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/achin/COS40007-Group/logs/%j.out

# Environment
ml gcc/12.2.0
ml python/3.10.8
cd /home/achin/COS40007-Group
source .venv/bin/activate

# Prep
mkdir -p logs

# Pick a random free port to avoid conflicts with other users
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# Resolve the compute node hostname
NODE=$(hostname -f)

# SSH tunnel instructions
echo "============================================================"
echo "  Jupyter is starting on: ${NODE}:${PORT}"
echo ""
echo "  To connect, run this on your LOCAL machine:"
echo "    ssh -N -L ${PORT}:${NODE}:${PORT} achin@tooarrana1.hpc.swin.edu.au"
echo ""
echo "  Then open in your browser:"
echo "    http://localhost:${PORT}"
echo "============================================================"


# Launch Jupiter
jupyter notebook \
    --no-browser \
    --port="${PORT}" \
    --ip=0.0.0.0 \
    --NotebookApp.token='' \
    --NotebookApp.password=''
