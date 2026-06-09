# AGENTS.md

## What this is
COS40007 group project (Swinburne "AI for Engineering"): trains an RT-DETR-L defect detector
on a YOLO-format dataset of 3 classes — `crack`, `pothole`, `wall_peeling`. Framework: Ultralytics 8.4.x.

## Gotchas that bite every new session

1. **Hardcoded Linux paths.** `dataset/dataset.yaml`, the first cell of every `Train*.ipynb`,
   `scripts/run_v5.sbatch`, and `scripts/launch_jupyter.sh` all contain
   `/home/achin/COS40007-Group/...`. Nothing resolves on this Windows box. Edit before running.
2. **`device='0'` assumes a CUDA GPU.** Original training used an A100 80GB (torch 2.8.0+cu128,
   Python 3.9, ultralytics 8.4.56). The HPC's `milan-gpu` partition always gives A100 80GB.
   On a non-CUDA local machine, set `device='cpu'`.
3. **Three notebooks, not one.**
   - `Train.ipynb` — first version, target `rtdetr_l_v2`
   - `Train1.ipynb` — `rtdetr_l_v3_frozen` experiment (`freeze=21`, `lr0=5e-4`, no aug)
   - `Train2-jason.ipynb` — **latest**, RUN=`rtdetr_l_v5`, iterative-training loop
   Use `Train2-jason.ipynb` for new runs unless continuing a specific experiment.
4. **`batch=8`, `amp=True` in the current notebook.** Older AGENTS.md versions said batch=16
   and amp=False — those were v2/v4 settings. The v5 notebook overrides them in the iter loop
   (cell 12). Both values are fine on A100; batch=16 also works if you change it.
5. **No `requirements.txt`.** Deps: `pip install ultralytics matplotlib pillow`
   (ultralytics pulls torch automatically).
6. **Pretrained weights `rtdetr-l.pt`** live at the repo root (committed) and are passed as `model=...`.
7. **Background images** (filenames `road-…` / `wall-…`) have empty `.txt` labels; they are not
   extra classes. `nc: 3` in `dataset/dataset.yaml` is correct. Don't add `road` / `wall` to `names`.
   Note: `wall_peeling-…` is a different prefix and is the defect class; leave the prefix convention alone.
8. **Already-trained checkpoints** in `runs/defect_detection/rtdetr_l_v{2,4,5_iter0..4}/weights/best.pt` —
   use them for inference without retraining.
9. **`runs/` is committed** (the project's 2nd commit is literally "runs"). Don't `git rm` it.
   `.gitignore` covers `.ipynb_checkpoints/`, `.venv/`, `jenny/`, `scripts/`, `docs/` — but
   `scripts/` and `docs/` are tracked from before the rule was added.
10. **Notebook outputs are base64 PNG evidence of results** (training curves, confusion matrices,
    prediction samples, etc.). Do not clear cell outputs to "clean up" — they are the report.
11. **Lmod vs venv conflict.** The HPC exposes Python via Lmod (`ml python/3.10.8`). The project
    venv is at `.venv/` (Python 3.9). `launch_jupyter.sh` and `run_v5.sbatch` both
    `module unload python/3.10.8` before `source .venv/bin/activate`. If you do both manually,
    the order matters: `ml` first, then `source`. Cleanest to NOT add `ml python/3.10.8` to
    `~/.bashrc` so the venv's interpreter always wins.
12. **Same-file nbconvert deadlock risk.** `jupyter nbconvert --to notebook --execute --output X`
    where X is the input file can stall. `run_v5.sbatch` writes to `/tmp/X.executed.ipynb` then
    `mv`-s back. Use this pattern for any unattended run.
13. **`dataset.yaml` vs notebook class name mismatch.** `dataset/dataset.yaml` line 9 says
    `potholes` (with 's'), but the notebook and `dataset_iter*.yaml` files use `pothole`
    (without 's'). Training uses the iter yamls, so this doesn't break anything — but don't
    "fix" one without the other. The canonical class list is `['crack', 'pothole', 'wall_peeling']`.
14. **Current state: STOP reached.** Iter 4 hit mAP50=0.8702 (≥ 0.85 threshold), wrote the
    `STOP` sentinel, and iter 5 was never run. The iterlog has 5 rows (iters 0–4). To re-run
    the full 6-iter loop, delete `runs/defect_detection/STOP` and the iterlog + per-iter dirs.

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
4. Run all cells top-to-bottom. 50 epochs × 6 iters @ `imgsz=640`, `batch=8` is hours on an
   A100, much longer on CPU.

### SSH into HPC
Training and inference both happen on the Swinburne HPC. This box is **not** on the
cluster — you must SSH in first, then request a GPU allocation with `srun`:

```bash
ssh tooarrana1.hpc.swin.edu.au
# (full form: ssh achin@tooarrana1.hpc.swin.edu.au)
```

Auth is by **password**, not SSH key. Once on the login node, clone or pull the repo there
(the local Windows copy and the HPC copy will diverge — commit/push from the HPC
checkout when training is done so the local box has the latest `runs/` and weights).

### Retrain on HPC (interactive)
```bash
srun --partition=milan-gpu --time=02:30:00 --gres=gpu:1 --cpus-per-task=4 --mem=48G --pty bash -i
cd ~/COS40007-Group
source .venv/bin/activate
jupyter nbconvert --to notebook --execute Train2-jason.ipynb
```

A 2.5h slot covers ~2-3 iters. The iter loop is resumable — completed iters auto-skip.
Resume with the same `jupyter nbconvert` command. Do **not** start a fresh run.

### Retrain on HPC (unattended sbatch)
```bash
sbatch scripts/run_v5.sbatch
```

Auto-chains up to 5 submissions (12.5h). See `docs/RUN_ON_HPC.md` for full details
(monitoring, stopping the chain, switching between workflows, etc.).

### Where outputs land
- Training: `runs/defect_detection/<RUN>/{args.yaml, results.csv, results.png, weights/, ...}`
- Predictions: `runs/defect_detection/<RUN>_predictions/*.jpg`
- Test eval: `runs/defect_detection/<RUN>_test/`
- Val predictions JSON: `runs/defect_detection/<RUN>_val/predictions.json`
- EDA plots: `runs/defect_detection/{class_distribution.png, sample_annotations_*.png, ...}`

**Run-name convention**: bump the suffix — `rtdetr_l_v2`, `rtdetr_l_v3_frozen`, `rtdetr_l_v4`
already exist, so the next new run is `rtdetr_l_v5`, then `v6`, etc.

### Iterative training (active-learning pattern) — `Train2-jason.ipynb` only
6-iteration loop that grows the training set cumulatively. Train schedule (cell 12):
`ITER_TRAIN_SIZES = [400, 450, 500, 550, 600, 621]`, 50 epochs per iter, `SEED=42`.
Stop condition: per-iter val mAP50 ≥ 0.85 (writes a `STOP` sentinel at the project root).

Per-iter artifacts:
- `runs/defect_detection/rtdetr_l_v5_iter{0..4}/` — weights, plots, confusion matrices
- `runs/defect_detection/rtdetr_l_v5_iter{N}_test_iou.csv` — per-test-image IoU
- `runs/defect_detection/rtdetr_l_v5_iter{N}_summary.json` — mAP50/precision/recall/per-class AP50/time
- `runs/defect_detection/rtdetr_l_v5_iterlog.csv` — append-only row per completed iter

**Resumable**: an iter is skipped if its `last.pt` AND iterlog row both exist.
Mid-kill leaves a partial `last.pt` with no iterlog row → next run retrains that iter.
Initial weights for `n>0` are loaded from the previous iter's `last.pt` (NOT `resume=True`).

## Layout
- `Train*.ipynb`            — main entry points
- `dataset/dataset.yaml`    — class config (3 classes, fix paths; has `potholes` typo)
- `dataset/dataset_iter*.yaml` — per-iter yamls generated by the notebook (use `pothole`)
- `dataset/splits/`         — train/val/test split lists (`.txt`, one image path per line)
- `dataset/{train,val,test}/{images,labels}/` — YOLO-format data
- `rtdetr-l.pt`             — RT-DETR-L pretrained weights (root, committed)
- `runs/defect_detection/`  — training & eval artifacts (committed)
- `scripts/run_v5.sbatch`   — SLURM batch script with auto-chain (5× 2.5h max)
- `scripts/chain_v5.sh`     — alt: login-node monitor for fine-grained job control
- `scripts/launch_jupyter.sh` — helper to start Jupyter Lab inside an srun session
- `scripts/hpc_jupyter_tunnel.ps1` — Windows PowerShell helper for chained SSH tunnel
- `docs/RUN_ON_HPC.md`      — quick-reference card for both run workflows
- `docs/PRE_RUN_AUDIT_REPORT.md` — pre-run audit findings (1 bug found and fixed)
- `.venv/`, `jenny/`        — gitignored personal scratch

## Conventions
- `class_id` in label `.txt` is YOLO 0-indexed: 0=crack, 1=pothole, 2=wall_peeling.
  Format: `cls cx cy w h` (normalized xywh).
- Use `rtdetr_l_v{N}` (or `rtdetr_l_v{N}_{tag}`) as the run name; v2, v3_frozen, and v4
  already exist.
- Don't bump Python past 3.12 carelessly — notebook outputs were produced on 3.9.
- Detailed HPC run workflows (interactive vs sbatch, tunnel setup, chain management)
  are in `docs/RUN_ON_HPC.md` — don't duplicate them here.
