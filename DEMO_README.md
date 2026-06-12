# How to Run the AI Structural Defect Detection Demo

**COS40007 Design Project — Theme 2: Smart Campus Infrastructure Monitoring**

---

## 1. Prerequisites

- Python 3.9+ installed
- GPU recommended (for faster inference) but CPU works fine
- ~500 MB disk space for models + dependencies

---

## 2. Installation

### Step 1: Clone the Project
```bash
git clone <GITHUB_REPO_URL>
cd COS40007-Group
```

### Step 2: Create and Activate a Virtual Environment

**On Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux / HPC:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Run the Demo

```bash
python -m streamlit run app.py
```

This opens a browser at `http://localhost:8501` with the dashboard interface.

---

## 4. Using the Dashboard

### Step 1: Select Analysis Protocol (Sidebar)

- **Single Image Audit**: Run in-depth visual and quantitative reports on individual images
- **Batch Folder Scan**: Process bulk directories with consensus checks and aggregate statistics

### Step 2: Choose Image Source (Sidebar)

- **Upload Image File**: Drag & drop your own infrastructure photo
- **Curated Infrastructure Gallery**: Click any curated sample to load it into the audit engine
- **Academic Test Dataset**: Pick from the 117-image test set partition

### Step 3: Configure Neural Ensemble Parameters (Sidebar)

- **Confidence Gate** (default: 0.25): Minimum confidence to show a detection. Lower values increase recall.
- **Ensemble Fusion View**: Choose from 6 modes:
  - *Compare Side-by-Side*: RT-DETR-L and YOLO26m rendered independently
  - *Union (High Recall Fused)*: All detections from both models, deduplicated
  - *Intersection (High Precision Fused)*: Only defects both models agree on
  - *Weighted Average Fused*: Confidence-weighted coordinate averaging for overlapping boxes
  - *RT-DETR-L (Transformer Only)*: Single-model view
  - *YOLO26m (CNN Only)*: Single-model view
- **Overlap Matching IoU** (default: 0.35): IoU threshold for declaring consensus between models

### Step 4: View Results

- **KPI Dashboard**: Structural Health Index (SHI), active detection count, average latency, consensus overlaps
- **Urgency Status Banner**: Color-coded safety assessment (Safe / Watch / Alert / Critical)
- **Annotated Visualization**: High-definition defect overlay with class-colored bounding boxes
- **Interactive ROI Gallery**: Cropped defect regions with morphological sub-classification
- **Detailed Diagnostic Table**: Full detection metadata including source model and consensus status
- **Model Architectural Specs**: Side-by-side comparison of RT-DETR-L and YOLO26m parameters

### Step 5: Export Results

- **Annotated PNG (Fused Model)**: Single overlay image
- **Side-by-Side Composite Image**: RT-DETR-L and YOLO26m rendered side-by-side
- **Diagnostic Table CSV**: All detections with coordinates and metadata
- **Diagnostic Meta JSON**: Full structured export including SHI and urgency class
- **Field Audit Report (Markdown)**: Printable inspection document with executive summary, defect roster, and recommended actions

---

## 5. Structural Health Index (SHI)

The SHI is a computed metric from 0 to 100 that integrates:

- **Class priority weights**: `crack` (15.0), `pothole` (25.0), `wall_peeling` (8.0)
- **Pixel area fraction**: Bounding box area relative to total image size
- **Detection confidence**: Weighted by model certainty

| SHI Range | Status | Meaning |
|-----------|--------|---------|
| >= 90 | Safe / Nominal | No urgent issues. Schedule annual maintenance. |
| 75 - 89 | Watch | Minor deterioration. Monitor quarterly. |
| 55 - 74 | Alert / Warning | Moderate defects. Schedule repairs within 6 months. |
| < 55 | Critical Safety Hazard | Severe damage. Immediate intervention required. |

---

## 6. Neural Ensemble Fusion Engine

The fusion engine combines detections from RT-DETR-L and YOLO26m using spatial IoU matching:

| Mode | Strategy | Use Case |
|------|----------|----------|
| Union (High Recall) | Keep all, deduplicate overlaps by confidence | Safety-critical inspections |
| Intersection (High Precision) | Keep only mutual matches | Minimize false positives |
| Weighted Average | Average overlapping box coordinates | Precise boundary estimation |

---

## 7. What the Models Detect

| Class | Description | Colour | Engineering Weight |
|-------|-------------|--------|--------------------|
| crack | Linear fractures in concrete/asphalt | Red | 15.0 |
| pothole | Circular/irregular depressions | Green | 25.0 |
| wall_peeling | Surface delamination/peeling paint | Blue | 8.0 |

---

## 8. Model Details

| Model | Type | mAP50 | mAP50-95 | Precision | Recall | Parameters |
|-------|------|-------|----------|-----------|--------|------------|
| RT-DETR-L | Transformer (encoder-decoder) | 0.613 | 0.315 | 0.696 | 0.661 | 32M |
| YOLO26m | CNN (single-stage) | 0.763 | — | — | — | 11M |

*RT-DETR-L metrics are from the peak iteration (iter 3) of a 6-iter active-learning loop. YOLO26m metrics are from the frozen-backbone run (277 epochs).*

---

## 9. Dashboard Tabs

### Real-time Diagnostic Audit
Primary tab for image upload, inference, SHI calculation, annotated visualization, ROI crop gallery, and export.

### Engineering Methodology & Trade-offs
Documents the architecture differences between RT-DETR-L and YOLO26m, dataset profile (780 images, 3 classes), training loops (active-learning for RT-DETR-L, ablation study for YOLO26m), and fusion logic.

### Empirical Training Evidence
Displays iterlog tables, YOLO ablation comparisons, training curves, PR curves, confusion matrices, and class distribution charts for both model families.

---

## 10. Batch Folder Scan

- Upload multiple images for bulk evaluation
- Progress bar tracks inference across all images
- Aggregate dashboard shows: images processed, mean SHI, critical hazard frames, consensus matching rate
- Bar charts for defect class breakdown and model latency comparison
- Per-image inspector with side-by-side views and ROI crops
- ZIP export package containing: batch summary CSV, metadata JSON, and side-by-side composite JPGs

---

## 11. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: streamlit` | Run `pip install streamlit` |
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| Model file not found | Ensure `runs/defect_detection/rtdetr_l_v7_iter3/weights/best.pt` and `jenny/runs/detect/road_damage/frozen_backbone/weights/best.pt` exist |
| Slow inference on CPU | Wait ~2-3 seconds per image; GPU recommended for real-time |
| `Address already in use` | Run `python -m streamlit run app.py --server.port 8502` |
| `streamlit: command not found` | Run `python -m streamlit run app.py` instead |
| Deprecated `use_container_width` warning | Cosmetic only; the dashboard handles this gracefully |

---

## 12. Project Structure (relevant files)

```
COS40007-Group/
├── app.py                              ← Main dashboard (run this)
├── app_backup.py                       ← Backup of original app.py
├── requirements.txt                    ← Dependencies
├── demo_images/                        ← Curated gallery images
├── dataset/test/images/                ← Academic test set (117 images)
├── runs/defect_detection/
│   ├── rtdetr_l_v7_iter3/weights/best.pt  ← RT-DETR-L model (peak iter)
│   └── rtdetr_l_v7_iterlog.csv            ← Iteration training log
├── jenny/yolo26m.pt                    ← YOLO26m pretrained weights
└── jenny/runs/detect/road_damage/
    └── frozen_backbone/weights/best.pt ← YOLO26m model (best ablation)
```

---

*Prepared for COS40007 Design Project Submission, June 2026*
