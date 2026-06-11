# AGENTS.md

## What this is
COS40007 Design Project (Swinburne "AI for Engineering", Theme 2: Structural Defect
Detection). Two-model pipeline trained on a YOLO-format dataset of 3 classes —
`crack`, `pothole`, `wall_peeling` — over 780 images (train 546 / val 117 / test 117):

1. **RT-DETR-L** (`runs/defect_detection/rtdetr_l_v7_iter3/` — peak mAP50 ≈ 0.61) — driven
   by the 6-iteration active-learning loop in `Train3.ipynb`.
2. **YOLO26s** ablation (`jenny/runs/detect/road_damage/unfrozen/`) — three runs
   (`frozen_backbone`, `partial_freeze_neck`, `unfrozen`) for side-by-side demo
   comparison.

The Streamlit demo `app.py` loads both and lets users pick / upload images. Framework:
Ultralytics 8.4.x. Deployed to Streamlit Cloud as `infrastructure-defect-vision`.

## Gotchas that bite every new session

1. **Hardcoded Linux paths.** `dataset/dataset.yaml`, the first cell of every
   `Train*.ipynb`, the root-level `run_*.sbatch` files, and `scripts/launch_jupyter.sh`
   all contain `/home/achin/COS40007-Group/...`. Nothing resolves on this Windows box —
   edit before running on a non-HPC machine.
2. **`device='0'` assumes a CUDA GPU.** Original training used an A100 80GB
   (torch 2.8.0+cu128, Python 3.9, ultralytics 8.4.56). The HPC's `milan-gpu` partition
   always gives A100 80GB. On a non-CUDA local machine, set `device='cpu'`.
3. **Four notebooks, three generations.**
   - `Train.ipynb` — v1, target `rtdetr_l_v2`
   - `Train1.ipynb` — `rtdetr_l_v3_frozen` experiment
   - `Train2-jason.ipynb` — `rtdetr_l_v5` (STOPPED, dirs deleted; do not re-use)
   - `Train3.ipynb` — **current**, RUN=`rtdetr_l_v7`, active-learning loop
   - `Train copy.ipynb` — earlier `rtdetr_l_v4` run (last.pt only; no best.pt)
   Use `Train3.ipynb` for new RT-DETR runs.
4. **YOLO notebooks live in `jenny/`.** `jenny/aipro.ipynb` is the training notebook;
   output weights land in `jenny/runs/detect/road_damage/{frozen_backbone,partial_freeze_neck,unfrozen}/`.
5. **Pretrained weights** are committed: `rtdetr-l.pt` at repo root (66 MB) and
   `jenny/yolo26s.pt` (20 MB). Passed as `model=...` when fine-tuning.
6. **No `requirements.txt` for training.** The repo's `requirements.txt` is for the
   **demo only** (`streamlit`, `ultralytics`, `pandas`, `pillow`, `numpy`). Training
   notebooks just need `ultralytics` — it pulls torch automatically.
7. **Background images** (filenames `road-…` / `wall-…`) have empty `.txt` labels; they
   are not extra classes. `nc: 3` in `dataset/dataset.yaml` is correct. Do **not** add
   `road` / `wall` to `names`. Note: `wall_peeling-…` is a different prefix and is the
   defect class — leave the prefix convention alone.
8. **`dataset.yaml` vs notebook class name mismatch.** `dataset/dataset.yaml` line 9
   says `potholes` (with 's'), and `jenny/dataset.yaml` copies the same typo. The
   notebooks and per-iter yamls use `pothole` (without 's'). Training uses the iter
   yamls, so this doesn't break anything — but don't "fix" one without the other.
   The canonical class list is `['crack', 'pothole', 'wall_peeling']`.
9. **Notebook outputs are base64 PNG evidence of results** (training curves, confusion
   matrices, prediction samples, etc.). Do not clear cell outputs to "clean up" — they
   are the report.
10. **Lmod vs venv conflict.** The HPC exposes Python via Lmod (`ml python/3.10.8`).
    The project venv is at `.venv/` (Python 3.9). All `run_*.sbatch` and `scripts/*.sh`
    `module load` (or `unload`) Python first, then `source .venv/bin/activate`. If you
    do both manually: `ml` first, then `source`. Cleanest to NOT add `ml python/3.10.8`
    to `~/.bashrc` so the venv's interpreter always wins.
11. **Same-file nbconvert deadlock risk.** `jupyter nbconvert --to notebook --execute
    --output X` where X is the input file can stall. The current `run_train3.sbatch`
    writes to `/tmp/Train3.${JOB_ID}.executed.ipynb` then `mv`-s back. Use this pattern
    for any unattended run.
12. **`.gitignore` caveats.** `runs/` is committed (the project's 2nd commit is literally
    "runs"). Don't `git rm` it. `.gitignore` covers `.ipynb_checkpoints/`, `.venv/`,
    `__pycache__/`, `logs/`, `runs/defect_detection/rtdetr_l_v4/weights/best.pt`
    (intentionally excluded to keep the repo small — see gotcha #15).
13. **App model path resolution.** `app.py` uses `_resolve_path([...])` to pick the
    first existing candidate from a list. The first candidate for RTDETR is
    `runs/defect_detection/rtdetr_l_v7_iter3/weights/best.pt` — keep it. The fallback
    `rtdetr_l_v5_iter4/weights/best.pt` is **dead code** (v5 dirs deleted) but harmless
    because the first candidate always exists. Don't add a new "first" candidate without
    updating the order so the best model wins.
14. **Demo images are mid-rename.** `demo_images/` has 5 old files deleted and 6 new
    ones staged but uncommitted. Don't `git restore demo_images/` without checking —
    the new set is intentional.
15. **`rtdetr_l_v4` weights are partial.** `runs/defect_detection/rtdetr_l_v4/weights/`
    contains only `last.pt` (66 MB) — `best.pt` is gitignored. v4 is kept for plots
    and results.csv only; do not try to load it via `RTDETR(.../v4/weights/best.pt)`.
16. **Current state: v7 STOP not reached.** v7's 6-iter loop completed (iters 0-5) but
    the peak (iter 3, mAP50=0.6133) was below the 0.85 threshold, so no `STOP` sentinel
    was written. `rtdetr_l_v7_iterlog.csv` is the source of truth. To re-run, delete
    the iterlog + per-iter dirs.

## How to run

### Smoke-test inference (RT-DETR)
```python
from ultralytics import RTDETR
m = RTDETR("runs/defect_detection/rtdetr_l_v7_iter3/weights/best.pt")
m.predict(source="dataset/test/images", conf=0.25, save=True,
          project="runs/defect_detection", name="infer_smoke")
```

### Smoke-test inference (YOLO)
```python
from ultralytics import YOLO
m = YOLO("jenny/runs/detect/road_damage/unfrozen/weights/best.pt")
m.predict(source="dataset/test/images", conf=0.25, save=True,
          project="jenny/runs/detect", name="infer_smoke")
```

### Run the Streamlit demo (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`. Loads both models via the cached `_resolve_path`
helper. See `DEMO_README.md` for screenshots, troubleshooting, and HF Space deploy
instructions.

### Retrain RT-DETR (from a notebook)
1. Open `Train3.ipynb` (or `jenny/aipro.ipynb` for YOLO).
2. Edit the **first cell** paths to point at the local repo root.
3. Set `device=` and (if needed) `batch=` on the training call.
4. Run all cells top-to-bottom. 50 epochs × 6 iters @ `imgsz=640`, `batch=8` is hours
   on an A100, much longer on CPU.

### SSH into HPC
Training and inference both happen on the Swinburne HPC. This box is **not** on the
cluster — you must SSH in first, then request a GPU allocation with `srun`:

```bash
ssh tooarrana1.hpc.swin.edu.au
# (full form: ssh achin@tooarrana1.hpc.swin.edu.au)
```

Auth is by **password**, not SSH key. Once on the login node, clone or pull the repo
there (the local Windows copy and the HPC copy will diverge — commit/push from the
HPC checkout when training is done so the local box has the latest `runs/` and
weights). **Git push hangs** from this Windows box's git; use `hf upload` for
remote sync (see deployment below).

### Retrain RT-DETR on HPC (unattended sbatch)
```bash
sbatch run_train3.sbatch     # v7 / Train3.ipynb (current default)
# or
sbatch run_300epoch.sbatch   # v4 / Train copy.ipynb (legacy; has the SIGTERM-120s hook)
```

`run_train3.sbatch` is a single 4-hour job — no auto-chain (unlike the old
`run_v5.sbatch`, which is gone). The iter loop is resumable: completed iters
auto-skip. Re-submit to resume after a kill.

### Where outputs land
- **RT-DETR training:** `runs/defect_detection/rtdetr_l_v{N}_iter{K}/{args.yaml, results.csv, results.png, weights/, ...}`
- **RT-DETR predictions:** `runs/defect_detection/rtdetr_l_v{N}_iter{K}_predictions/*.jpg`
- **RT-DETR test eval:** `runs/defect_detection/rtdetr_l_v{N}_iter{K}_test/`
- **RT-DETR val JSON:** `runs/defect_detection/rtdetr_l_v{N}_iter{K}_val/predictions.json`
- **RT-DETR iterlog:** `runs/defect_detection/rtdetr_l_v{N}_iterlog.csv` (one row per completed iter)
- **RT-DETR iter summary:** `runs/defect_detection/rtdetr_l_v{N}_iter{K}_summary.json`
- **YOLO training:** `jenny/runs/detect/road_damage/{frozen_backbone,partial_freeze_neck,unfrozen}/...`
- **EDA plots:** `runs/defect_detection/{class_distribution.png, iter_comparison_*.png, ...}`

**Run-name convention**: bump the suffix — `rtdetr_l_v2`, `rtdetr_l_v3_frozen`,
`rtdetr_l_v4` (last.pt only), `rtdetr_l_v5` (deleted), `rtdetr_l_v6`,
`rtdetr_l_v7` (current) already exist. Next new run is `rtdetr_l_v8`.

### Iterative training (active-learning pattern) — `Train3.ipynb` only
6-iteration loop that grows the training set cumulatively. Train schedule (cell 12):
`ITER_TRAIN_SIZES = [400, 428, 456, 484, 512, 541]`, 50 epochs per iter, `SEED=42`.
Stop condition: per-iter val mAP50 ≥ 0.85 (writes a `STOP` sentinel at the project
root). **v7 never hit the threshold** — iter 3 was the peak at 0.6133.

**Resumable**: an iter is skipped if its `last.pt` AND iterlog row both exist. Mid-kill
leaves a partial `last.pt` with no iterlog row → next run retrains that iter. Initial
weights for `n>0` are loaded from the previous iter's `last.pt` (NOT `resume=True`).

### Deploy the Streamlit demo to Streamlit Cloud
The demo is hosted on **Streamlit Cloud** as `infrastructure-defect-vision` (not Hugging
Face). Streamlit Cloud auto-pulls from the configured GitHub branch on push and rebuilds
the venv from `requirements.txt`.

```bash
# from this Windows box, just commit & push; Streamlit Cloud does the rest
git add requirements.txt
git commit -m "fix: pin opencv-python-headless for Streamlit Cloud"
git push origin main
```

The repo root is the deployable unit. **Critical:** the app expects `runs/` and
`jenny/runs/` paths exactly as they are in the repo (relative to repo root). Don't flatten
to `models/`. If a deploy breaks, the most common causes are:
1. A missing or renamed weights file in `runs/defect_detection/rtdetr_l_v7_iter3/weights/`
   or `jenny/runs/detect/road_damage/unfrozen/weights/`.
2. A new transitive dep (e.g. `opencv-python` full build) that needs a system library
   not present in the Streamlit Cloud image — fix by pinning `opencv-python-headless`
   in `requirements.txt`.

## Layout
- `Train*.ipynb`             — RT-DETR training notebooks (use `Train3.ipynb`)
- `jenny/aipro.ipynb`        — YOLO26s training notebook
- `app.py`                   — Streamlit demo (RT-DETR vs YOLO side-by-side)
- `requirements.txt`         — **demo only** deps (`streamlit`, `ultralytics`, `pandas`, `pillow`, `numpy`)
- `DEMO_README.md`           — how to install, run, and deploy the demo
- `dataset/dataset.yaml`     — class config (3 classes, hardcoded paths, `potholes` typo)
- `dataset/dataset_iter*.yaml` — per-iter yamls generated by `Train3.ipynb` (use `pothole`)
- `dataset/splits/iter{0..5}_train.txt` — cumulative train paths per iter
- `dataset/{train,val,test}/{images,labels}/` — YOLO-format data (546 / 117 / 117)
- `demo_images/`             — hero/curated images shown in the Streamlit demo
- `default.png`              — fallback image when nothing else is loaded
- `rtdetr-l.pt`              — RT-DETR-L pretrained weights (root, 66 MB, committed)
- `jenny/yolo26s.pt`         — YOLO26s pretrained weights (20 MB, committed)
- `runs/defect_detection/`   — RT-DETR training & eval artifacts (committed; v4 best.pt gitignored)
- `jenny/runs/detect/road_damage/` — YOLO training & eval artifacts
- `run_train3.sbatch`        — SLURM batch script for `Train3.ipynb` (v7, current)
- `run_300epoch.sbatch`      — SLURM batch script for `Train copy.ipynb` (v4, legacy)
- `scripts/chain_v5.sh`      — v5-only login-node monitor (legacy; v5 dirs deleted)
- `scripts/launch_jupyter.sh` — older helper to start Jupyter inside an srun session
- `scripts/notebook.sh`      — newer generic Jupyter launcher (v7-compatible)
- `scripts/hpc_jupyter_tunnel.ps1` — Windows helper for chained SSH tunnel
- `docs/RUN_ON_HPC.md`       — quick-reference card for both run workflows (still v5-flavoured; out of date)
- `docs/PRE_RUN_AUDIT_REPORT.md` — pre-run audit findings
- `.venv/`, `__pycache__/`, `logs/` — gitignored personal scratch

## Conventions
- `class_id` in label `.txt` is YOLO 0-indexed: 0=crack, 1=pothole, 2=wall_peeling.
  Format: `cls cx cy w h` (normalized xywh).
- Use `rtdetr_l_v{N}` (or `rtdetr_l_v{N}_{tag}`) as the run name; v2, v3_frozen, v4,
  v5 (deleted), v6, v7 already exist.
- For YOLO ablations, use `frozen_backbone` / `partial_freeze_neck` / `unfrozen` as
  the variant name. `unfrozen` is the chosen one for the demo.
- Don't bump Python past 3.12 carelessly — notebook outputs were produced on 3.9.
- When changing `app.py` model candidates, **first = best** so `_resolve_path` picks it.
- Detailed HPC run workflows (interactive vs sbatch, tunnel setup, chain management)
  are in `docs/RUN_ON_HPC.md` — don't duplicate them here (and don't trust that file's
  v5 references either; it predates the v7 era).
