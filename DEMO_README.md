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

This opens a browser at `http://localhost:8501` with the demo interface.

---

## 4. Using the Demo

### Step 1: View or Upload an Image
- **Upload Image**: Click **"Upload Image"** to analyze your own photo
- **Select Sample**: Pick a sample demo image from the dropdown to swap cases
- **AI Demo Gallery**: Choose from the curated gallery in the sidebar for quick comparison

### Step 2: Adjust Parameters (optional)
- **Confidence Threshold** (sidebar): Minimum confidence to show a detection (default: 0.25). Lower values increase recall but may produce more false positives.

### Step 3: View Results
- **Side-by-side comparison**: RT-DETR-L (left) vs YOLO26s (right)
- **Detection boxes**: Coloured rectangles with defect labels and confidence scores
- **Analytics panel**: Defect counts, average confidence, confidence bands
- **Confidence bands**: High (≥ 0.70) / Medium (≥ 0.40) / Low (< 0.40) — note this is model confidence, not defect severity

### Step 4: Export Results
- **Download Annotated Image**: PNG with detection boxes drawn

---

## 5. What the Models Detect

| Class | Description | Colour |
|-------|-------------|--------|
| crack | Linear fractures in concrete/asphalt | Red |
| pothole | Circular/irregular depressions | Green |
| wall_peeling | Surface delamination/peeling paint | Blue |

---

## 6. Model Details

| Model | Type | mAP50 | mAP50-95 | Precision | Recall | Parameters |
|-------|------|-------|----------|-----------|--------|------------|
| RT-DETR-L | Transformer (encoder-decoder) | 0.613 | 0.315 | 0.696 | 0.661 | 32M |
| YOLO26s | CNN (single-stage) | 0.816 | 0.703 | 0.797 | 0.439 | 11M |

*RT-DETR-L metrics are from the peak iteration (iter 3) of a 6-iter active-learning loop. YOLO26s metrics are from the unfrozen-backbone run (300 epochs).*

---

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: streamlit` | Run `pip install streamlit` |
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| Model file not found | Ensure `runs/defect_detection/rtdetr_l_v7_iter3/weights/best.pt` and `jenny/runs/detect/road_damage/unfrozen/weights/best.pt` exist |
| Slow inference on CPU | Wait ~2-3 seconds per image; GPU recommended for real-time |
| `Address already in use` | Run `python -m streamlit run app.py --server.port 8502` |
| `streamlit: command not found` | Run `python -m streamlit run app.py` instead |

---

## 8. Project Structure (relevant files)

```
COS40007-Group/
├── app.py                          ← Main demo (run this)
├── default.png                     ← Default demo image loaded on launch
├── requirements.txt                ← Dependencies
├── dataset/test/images/            ← Sample images for testing
├── runs/defect_detection/
│   └── rtdetr_l_v7_iter3/weights/best.pt  ← RT-DETR-L model (peak iter)
├── jenny/yolo26s.pt                ← YOLO26s pretrained weights
└── jenny/runs/detect/road_damage/
    └── unfrozen/weights/best.pt    ← YOLO26s model (best ablation)
```

---

*Prepared for COS40007 Design Project Submission, June 2026*
