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
- **Instant Demo**: The app launches with a pre-loaded default image (`default.png`) containing all three defect types (crack, pothole, wall peeling) to showcase side-by-side detection immediately!
- **Upload Image**: Click **"Upload Image"** to analyze your own photo
- **Select Sample**: Or pick a different sample demo image from the dropdown to swap cases

### Step 2: Adjust Parameters (optional)
- **Confidence Threshold** (sidebar): Minimum confidence to show a detection (default: 0.25)
- **IoU Threshold** (sidebar): Non-Max Suppression overlap threshold (default: 0.50)

### Step 3: View Results
- **Side-by-side comparison**: RT-DETR-L (left) vs YOLOv8s (right)
- **Detection boxes**: Colored rectangles with defect labels and confidence scores
- **Analytics panel**: Defect counts, average confidence, severity ratings
- **Severity indicator**: 🔴 High / 🟠 Medium / 🟢 Low

### Step 4: Export Results
- **Download CSV**: Full detection table (model, class, confidence, severity, bounding box)
- **Download Annotated Image**: PNG with detection boxes drawn

---

## 5. What the Models Detect

| Class | Description | Color |
|-------|-------------|-------|
| crack | Linear fractures in concrete/asphalt | 🔴 Red |
| pothole | Circular/irregular depressions | 🟢 Green |
| wall_peeling | Surface delamination/peeling paint | 🔵 Blue |

---

## 6. Model Details

| Model | Type | mAP50 | Parameters | Inference Speed |
|-------|------|-------|------------|-----------------|
| RT-DETR-L | Transformer (encoder-decoder) | 0.87 | 32M | ~150 ms |
| YOLOv8s | CNN (single-stage) | 0.75 | 11M | ~80 ms |

---

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: streamlit` | Run `pip install streamlit` |
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| Model file not found | Ensure `runs/defect_detection/rtdetr_l_v5_iter4/weights/best.pt` exists |
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
│   └── rtdetr_l_v5_iter4/weights/best.pt  ← RT-DETR-L model
└── jenny/runs/detect/road_damage/
    └── unfrozen/weights/best.pt    ← YOLOv8s model
```

---

*Prepared for COS40007 Design Project Submission, June 2026*
