# AGENTS.md

## What this is
COS40007 group project (Swinburne "AI for Engineering"): trains an RT-DETR-L defect detector
on a YOLO-format dataset of 3 classes — `crack`, `pothole`, `wall_peeling`. Framework: Ultralytics 8.4.x.

## Read this first — gotchas that bite every new session

1. **Hardcoded Linux paths.** All paths in `dataset/dataset.yaml`, the first cell of every
   `Train*.ipynb`, and the `runs/.../args.yaml` files point at `/home/achin/COS40007-Group/...`.
   Nothing resolves on this Windows box. Edit before running.
2. **`device: '0'` assumes a CUDA GPU.** Original training used an A100 80GB (torch 2.8.0+cu128,
   Python 3.9, ultralytics 8.4.56). The HPC's `milan-gpu` partition is the only one with A100s
   per its "How to request for a GPU" doc, so on the cluster you always get an A100 80GB.
   `batch=16` (or `batch=-1` AutoBatch, which lands on 16 for A100) and `amp=False` or
   `amp=True` both work. On a non-CUDA local machine, set `device='cpu'` on the
   `YOLO(...).train(...)` / `.predict(...)` call.
3. **Three notebooks, not one.**
   - `Train.ipynb`         — first version, target `rtdetr_l_v2`; train body duplicates `Train2-jason.ipynb`'s
   - `Train1.ipynb`        — `rtdetr_l_v3_frozen` experiment (`freeze=21`, `lr0=5e-4`, no aug)
   - `Train2-jason.ipynb`  — **latest**, RUN=`rtdetr_l_v5`, with background-prefix EDA + iterative-training loop
   Use `Train2-jason.ipynb` for new runs unless continuing a specific experiment.
4. **Mixed precision is OFF** (`amp: false`) in the recorded args. AMP on (`amp=True`) is safe
   on A100 and slightly faster; it only matters on smaller GPUs that lack VRAM headroom
   (the HPC always allocates A100s, so this is largely a non-issue there).
5. **No `requirements.txt`.** Deps: `pip install ultralytics matplotlib pillow`
   (ultralytics pulls torch automatically).
6. **Pretrained weights `rtdetr-l.pt`** live at the repo root (committed) and are passed as `model=...`.
7. **Background images** (filenames `road-…` / `wall-…`) have empty `.txt` labels; they are not
   extra classes. `nc: 3` in `dataset/dataset.yaml` is correct. Don't add `road` / `wall` to `names`.
   Note: `wall_peeling-…` is a different prefix and is the defect class; leave the prefix convention alone.
8. **Already-trained checkpoints** in `runs/defect_detection/rtdetr_l_v{2,4,5_iter0..5}/weights/best.pt` —
   use them for inference without retraining.
9. **`runs/` is committed** (the project's 2nd commit is literally "runs"). Don't `git rm` it.
   `.gitignore` only covers `.ipynb_checkpoints/`, `.venv/`, `jenny/`.
10. **Notebook outputs are base64 PNG evidence of results** (training curves, confusion matrices,
    prediction samples, etc.). Do not clear cell outputs to "clean up" — they are the report.
11. **Lmod vs venv conflict.** The HPC exposes Python via Lmod (`ml python/3.10.8`). The project
    venv is at `.venv/`. `launch_jupyter.sh` and `run_v5.sbatch` both `module unload python/3.10.8`
    before `source .venv/bin/activate` to avoid the venv's python resolving to a broken
    symlink. If you do both `ml python/3.10.8` AND `source .venv/bin/activate` manually, the
    module's env vars are needed by the venv's python to find shared libs, so the order matters:
    `ml` first, then `source`. In `~/.bashrc` it's cleanest to NOT add `ml python/3.10.8` so the
    venv's own interpreter always wins.
12. **Same-file nbconvert deadlock risk.** `jupyter nbconvert --to notebook --execute --output X`
    where X is the input file can stall or corrupt the .ipynb under load. `run_v5.sbatch` and
    `RUN_ON_HPC.md` both recommend writing to `/tmp/X.executed.ipynb` then `mv`-ing back. Use
    this pattern for any unattended run.

## How to run

### Smoke-test inference (no retraining)
```python
from ultralytics import RTDETR
m = RTDETR("runs/defect_detection/rtdetr_l_v4/weights/best.pt")
m.predict(source="dataset/test/images", conf=0.25, save=True,
          project="runs/defect_detection", name="infer_smoke")
```

### Retrain (from a notebook)
1. Open `Train2-jason.ipynb` (or the run you want to continue).
2. Edit the **first cell** paths to point at the local repo root.
3. Set `device=` and (if needed) `batch=` on the training call.
4. Run all cells top-to-bottom. 300 epochs @ `imgsz=640`, `batch=16` is hours on an A100, much
   longer on CPU.

### SSH into HPC
Training and inference both happen on the Swinburne HPC. This box is **not** on the
cluster — you must SSH in first, then request a GPU allocation with `srun`. There is an
SSH config entry already, so the host alias works:

```bash
ssh tooarrana1.hpc.swin.edu.au
# (full form: ssh achin@tooarrana1.hpc.swin.edu.au)
```

Auth is by **password**, not SSH key — `~/.ssh` on this Windows box has no private key,
only the `config` and `known_hosts` files. The prompt is the HPC account password, not
the local Windows one. Keys are not strictly required, but generating one and adding
the public key to `~/.ssh/authorized_keys` on the cluster saves you typing the password
on every login (recommended for sbatch-driven runs).

Once on the login node, `pwd` should land you in `$HOME`. Clone or pull the repo there
(the local Windows copy and the HPC copy will diverge — commit/push from the HPC
checkout when training is done so the local box has the latest `runs/` and weights).

### Retrain on the HPC cluster (preferred for full training runs)
Training runs are performed on a remote SLURM-managed GPU server, not on local machines.
After SSH'ing in, request an interactive GPU allocation with `srun` (this is what
actually grabs the GPU — running python on the login node will not see CUDA):

```bash
srun --partition=milan-gpu --time=02:30:00 --gres=gpu:1 --cpus-per-task=4 --mem=48G --pty bash -i
```

Notes:
- `--partition=milan-gpu` — milan-gpu partition (AMD Milan + GPU node).
- `--time=02:30:00` — 2h30m wall clock. Bump if running the full 300-epoch schedule;
  a single 2.5h slot is enough for a smoke run or ~50–80 epochs, not the full training.
  When the wall clock hits zero the kernel is killed mid-step, so:
  - Ultralytics' default periodic save to `weights/last.pt` is your friend.
  - In the next `srun` slot, resume with
    `YOLO("runs/defect_detection/<RUN>/weights/last.pt").train(resume=True, ...)`
    — do **not** start a fresh run, that resets the epoch counter and learning-rate schedule.
  - After the final epoch, copy `weights/best.pt` and `weights/last.pt` to a scratch path
    and `git add`/`git commit`/`git push` them so the local checkout has the latest weights.
- `--gres=gpu:1` — one **A100 80GB**. Per the HPC's "How to request for a GPU" doc, `milan-gpu`
  is the only partition with A100s, so the GPU is fixed (not "whatever the scheduler hands you").
  This means:
  - Keep `device='0'` (the allocated GPU appears as index 0 inside the job).
  - `batch=16` is the original A100 value and fits comfortably; `batch=-1` (Ultralytics
    AutoBatch) also lands on 16 for A100.
  - `amp=False` is the original setting; `amp=True` is also fine on A100 and slightly faster.
    Either works.
- `--cpus-per-task=4` — set Ultralytics `workers<=4` to match; more will be ignored or oversubscribe.
- `--mem=48G` — host RAM, separate from VRAM.
- `--pty bash -i` — interactive shell. For long unattended runs, use `sbatch` with a job
  script instead so the run survives ssh disconnects.

Once inside the allocation: activate the env, `cd` to the repo, fix paths (gotcha #1),
then launch the notebook (e.g. `jupyter nbconvert --to notebook --execute Train2-jason.ipynb`)
or run an equivalent Python script. Bump the run name per gotcha — next new run is `rtdetr_l_v5`.

### Where outputs land
- Training: `runs/defect_detection/<RUN>/{args.yaml, results.csv, results.png, weights/,
  confusion_matrix*.png, Box*_curve.png, …}`
- Predictions on a source: `runs/defect_detection/<RUN>_predictions/*.jpg`
- Test split eval: `runs/defect_detection/<RUN>_test/`
- Val split predictions JSON: `runs/defect_detection/<RUN>_val/predictions.json`
- EDA plots: `runs/defect_detection/{class_distribution.png, sample_annotations_<split>.png, …}`

Each notebook writes into a uniquely-named `<RUN>` subfolder (e.g. `rtdetr_l_v4`).
**Run-name convention**: bump the suffix per training — `rtdetr_l_v2`, `rtdetr_l_v3_frozen`,
`rtdetr_l_v4` already exist, so the next new run is `rtdetr_l_v5`, then `v6`, etc. v3 is reserved
for the frozen-backbone experiment.

### Iterative training (active-learning pattern) — `Train2-jason.ipynb` only
The latest notebook runs a 6-iteration loop that grows the training set cumulatively
and tracks the best iter. Train schedule (cell 10): `ITER_TRAIN_SIZES = [400, 450, 500, 550, 600, 621]`,
50 epochs per iter, `SEED=42`. Stop condition: per-iter val mAP50 ≥ 0.85 (writes a `STOP`
sentinel at the project root). Initial weights for `n>0` are loaded from the previous
iter's `last.pt` (NOT `resume=True`, which Ultralytics can't reliably re-target across
run-dirs). Per-iter artifacts land at:
- `runs/defect_detection/rtdetr_l_v5_iter{0..5}/` — same contents as a normal run dir
- `runs/defect_detection/rtdetr_l_v5_iter{N}_test_iou.csv` — per-test-image IoU (cols: image, conf, pred_xyxy, gt_xyxy, iou)
- `runs/defect_detection/rtdetr_l_v5_iter{N}_summary.json` — per-iter mAP50/precision/recall/per_class_ap50/time_h
- `runs/defect_detection/rtdetr_l_v5_iter{N}_samples/image{0,1}.jpg` — 2 sample predictions
- `runs/defect_detection/rtdetr_l_v5_iterlog.csv` — append-only row per completed iter
- `runs/defect_detection/iter_comparison_*.png` — comparison grids (training curves, confusion matrices, PR curves, samples, IoU distribution)

**Resumable**: on re-run, an iter is skipped if its `last.pt` AND iterlog row both exist.
If the HPC slot kills the run mid-iter, the partial `last.pt` is left on disk and the
iterlog row is not appended, so the next run retrains that iter (wasted ~50 epochs of
work, not catastrophic). The comparison report (cells 15-21) tolerates missing iters
with a "not run" placeholder in each subplot.

### Run workflows (Jupyter Lab interactive vs sbatch)
Two supported run modes, both resumable from the same on-disk state. Full commands
in `RUN_ON_HPC.md`; quick summary:

- **Jupyter Lab (`launch_jupyter.sh` + `hpc_jupyter_tunnel.ps1`)** — best for live
  visibility, debugging, the validation run of iter 0. The .ipynb updates cell-by-cell
  as you run them. Requires an SSH tunnel from Windows (chained `tooarrana1` → compute
  node like `gina9`) and a browser. Dies if the SSH session drops.
- **sbatch (`run_v5.sbatch`)** — best for unattended overnight runs. The HPC
  enforces a **2.5h wall-clock cap per submission** (each gets ~2-3 iters), so
  the script **auto-chains** itself: when the current job ends, it checks
  `runs/defect_detection/rtdetr_l_v5_iterlog.csv` and resubmits if iter count
  < 6. Capped at 5 submissions (12.5h) as a safety net. The .ipynb only updates
  at job end (atomic write); live progress is in `logs/slurm-<jobid>.out`,
  `logs/chain_count` (1-5), and `runs/defect_detection/rtdetr_l_v5_iterlog.csv`.
- **`chain_v5.sh` (alt to in-sbatch chain)** — login-node monitor that submits
  + watches jobs. Mutually exclusive with the in-sbatch chain; default is the
  in-sbatch chain. Useful if you want fine-grained control from the terminal.

To stop the chain: `touch STOP_CHAIN && scancel -u achin`. Resume with
`rm STOP_CHAIN && sbatch run_v5.sbatch`. To switch to Jupyter Lab, do the same
+ `srun` a new slot. The iter log + last.pt are the source of truth; nothing is lost.

## Layout
- `Train*.ipynb`            — main entry points
- `dataset/dataset.yaml`    — class config (3 classes, fix paths)
- `dataset/{train,val,test}/{images,labels}/` — YOLO-format data
- `rtdetr-l.pt`             — RT-DETR-L pretrained weights (root, committed)
- `runs/defect_detection/`  — training & eval artifacts (committed)
- `run_v5.sbatch`           — SLURM batch script with auto-chain (5x 2.5h submissions max)
- `chain_v5.sh`             — alt: login-node monitor for fine-grained job control
- `launch_jupyter.sh`       — helper to start Jupyter Lab inside an srun session
- `hpc_jupyter_tunnel.ps1`  — Windows PowerShell helper for chained SSH tunnel
- `STOP_CHAIN`              — touch to stop auto-chain at next cell boundary (rm to resume)
- `RUN_ON_HPC.md`           — quick-reference card for both run workflows
- `PRE_RUN_AUDIT_REPORT.md` — pre-run audit findings (1 bug found and fixed)
- `.venv/`, `jenny/`        — gitignored personal scratch

## Conventions
- `class_id` in label `.txt` is YOLO 0-indexed: 0=crack, 1=pothole, 2=wall_peeling. Format: `cls cx cy w h` (normalized xywh).
- Use `rtdetr_l_v{N}` (or `rtdetr_l_v{N}_{tag}`) as the run name; v2, v3_frozen, and v4 already exist.
- Don't bump Python past 3.12 carelessly — notebook outputs were produced on 3.9 and some
  matplotlib/torch warnings may shift.
