"""
COS40007 Design Project — AI Structural Defect Detection Demo
RT-DETR-L vs YOLOv26s: Side-by-Side Comparison

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO, RTDETR
import io, os

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = {0: "crack", 1: "pothole", 2: "wall_peeling"}
CLASS_COLORS = {0: (255, 50, 50), 1: (50, 200, 50), 2: (50, 100, 255)}

RTDETR_CANDIDATES = [
    os.path.join("runs", "defect_detection", "rtdetr_l_v7_iter3", "weights", "best.pt"),
    os.path.join("runs", "defect_detection", "rtdetr_l_v5_iter4", "weights", "best.pt"),
]
YOLO_CANDIDATES = [
    os.path.join("jenny", "runs", "detect", "road_damage", "unfrozen", "weights", "best.pt"),
    os.path.join("jenny", "runs", "detect", "road_damage", "frozen_backbone", "weights", "best.pt"),
]

DEMO_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "test", "images")
DEFAULT_IMAGE_PATH = os.path.join(BASE_DIR, "default.png")

# ── Page Setup ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Structural Defect Detection",
    page_icon=":building_construction:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────
def _resolve_path(candidates):
    """Return the first candidate that exists on disk, else the first."""
    for p in candidates:
        full = os.path.join(BASE_DIR, p)
        if os.path.isfile(full):
            return full
    return os.path.join(BASE_DIR, candidates[0])


# ── Model Loading (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}

    rtdetr_path = _resolve_path(RTDETR_CANDIDATES)
    try:
        models["rtdetr"] = RTDETR(rtdetr_path)
        models["rtdetr_name"] = "RT-DETR-L"
        models["rtdetr_path"] = rtdetr_path
    except Exception as e:
        models["rtdetr"] = None
        models["rtdetr_error"] = str(e)

    yolo_path = _resolve_path(YOLO_CANDIDATES)
    try:
        models["yolo"] = YOLO(yolo_path)
        models["yolo_name"] = "YOLOv26s"
        models["yolo_path"] = yolo_path
    except Exception as e:
        models["yolo"] = None
        models["yolo_error"] = str(e)

    return models


# ── Inference ───────────────────────────────────────────────────────────
def run_inference(model, image: Image.Image, conf: float, iou: float):
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    r = results[0]
    boxes = r.boxes
    detections = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        xyxy = boxes.xyxy[i].cpu().numpy().tolist()
        conf_val = float(boxes.conf[i])
        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            "bbox": xyxy,
            "confidence": conf_val,
            "severity": classify_severity(conf_val),
        })
    return {"detections": detections}


def classify_severity(conf: float) -> str:
    if conf >= 0.7:
        return "High"
    elif conf >= 0.4:
        return "Medium"
    return "Low"


# ── Drawing ─────────────────────────────────────────────────────────────
def draw_boxes_on_pil(image: Image.Image, detections: list) -> Image.Image:
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id = det["class_id"]
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = f"{det['class_name']} {det['confidence']:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label above the box, or inside if near the top
        tw = len(label) * 8 + 10
        if y1 >= 24:
            draw.rectangle([x1, y1 - 22, x1 + tw, y1], fill=color)
            draw.text((x1 + 4, y1 - 20), label, fill="white")
        else:
            draw.rectangle([x1, y1 + 2, x1 + tw, y1 + 24], fill=color)
            draw.text((x1 + 4, y1 + 4), label, fill="white")

    return draw_img


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    conf = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )
    iou = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="RT-DETR-L is NMS-free and ignores this. For YOLOv26s, only filters duplicate overlapping boxes of the same class — detection count won't change if boxes don't overlap.",
    )

    st.divider()
    st.markdown("**Model Paths:**")
    models = load_models()

    if models.get("rtdetr"):
        st.success(f"RT-DETR-L loaded")
    else:
        st.error("RT-DETR-L not found")
        st.caption(models.get("rtdetr_error", ""))

    if models.get("yolo"):
        st.success(f"YOLOv26s loaded")
    else:
        st.error("YOLOv26s not found")
        st.caption(models.get("yolo_error", ""))

    st.divider()
    st.caption("COS40007 Design Project — Theme 2")


# ── Image Upload ────────────────────────────────────────────────────────
if "demo_sel" not in st.session_state:
    st.session_state["demo_sel"] = "(none)"
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0


def _on_upload():
    st.session_state["demo_sel"] = "(none)"


def _on_demo_change():
    st.session_state["uploader_key"] += 1


col_upload, col_demo = st.columns(2)

uploaded_file = col_upload.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    key=f"uploader_{st.session_state['uploader_key']}",
    on_change=_on_upload,
)

demo_files = []
if os.path.isdir(DEMO_IMAGES_DIR):
    demo_files = sorted(
        f for f in os.listdir(DEMO_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )[:12]

if demo_files:
    selected_demo = col_demo.selectbox(
        "Or pick a demo image:",
        ["(none)"] + demo_files,
        key="demo_sel",
        on_change=_on_demo_change,
    )
    if selected_demo != "(none)":
        demo_path = os.path.join(DEMO_IMAGES_DIR, selected_demo)
        with open(demo_path, "rb") as f:
            uploaded_file = io.BytesIO(f.read())
else:
    col_demo.info("Demo images not available — upload your own.")

if uploaded_file is None and os.path.isfile(DEFAULT_IMAGE_PATH):
    with open(DEFAULT_IMAGE_PATH, "rb") as f:
        uploaded_file = io.BytesIO(f.read())


# ── Detection Pipeline ─────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")
    st.subheader("Uploaded Image")
    st.image(image, caption="Input image", use_container_width=True)

    col1, col2 = st.columns(2)
    results = {}

    for col, model_key, label in [
        (col1, "rtdetr", "RT-DETR-L"),
        (col2, "yolo", "YOLOv26s"),
    ]:
        model = models.get(model_key)
        if model is None:
            col.error(f"{label} model not loaded")
            continue

        with col.spinner(f"Running {label} inference..."):
            t0 = __import__("time").time()
            result = run_inference(model, image, conf, iou)
            elapsed = __import__("time").time() - t0
            result["time_ms"] = elapsed * 1000
            results[model_key] = result

    if results:
        st.markdown("---")
        st.subheader("Detection Results — Side-by-Side Comparison")

        col_r, col_y = st.columns(2)

        for col, model_key, label in [
            (col_r, "rtdetr", "RT-DETR-L"),
            (col_y, "yolo", "YOLOv26s"),
        ]:
            if model_key not in results:
                continue
            r = results[model_key]
            det_img = draw_boxes_on_pil(image, r["detections"])

            with col:
                st.markdown(f"### {label}")
                st.image(det_img, caption=f"{label} detections", use_container_width=True)
                st.caption(f"Inference: {r['time_ms']:.0f} ms | Detections: {len(r['detections'])}")
                buf = io.BytesIO()
                det_img.save(buf, format="PNG")
                st.download_button(
                    label=f"Download {label} Image",
                    data=buf.getvalue(),
                    file_name=f"{model_key}_annotated.png",
                    mime="image/png",
                    key=f"dl_{model_key}_comparison",
                    use_container_width=True,
                )

        st.markdown("---")
        st.subheader("Detection Analytics")

        cols = st.columns(2)
        for col, model_key, label in [
            (cols[0], "rtdetr", "RT-DETR-L"),
            (cols[1], "yolo", "YOLOv26s"),
        ]:
            if model_key not in results:
                continue
            r = results[model_key]
            dets = r["detections"]

            with col:
                st.markdown(f"**{label}**")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total", len(dets))
                m2.metric("Crack", sum(1 for d in dets if d["class_id"] == 0))
                m3.metric("Pothole", sum(1 for d in dets if d["class_id"] == 1))
                m4.metric("Wall Peel", sum(1 for d in dets if d["class_id"] == 2))

                if dets:
                    avg_conf = np.mean([d["confidence"] for d in dets])
                    high = sum(1 for d in dets if d["severity"] == "High")
                    med = sum(1 for d in dets if d["severity"] == "Medium")
                    low = sum(1 for d in dets if d["severity"] == "Low")

                    st.metric("Avg Confidence", f"{avg_conf:.2f}")
                    sev1, sev2, sev3 = st.columns(3)
                    sev1.metric("High", high)
                    sev2.metric("Medium", med)
                    sev3.metric("Low", low)

                    if high > 0:
                        st.error("HIGH severity defects detected — immediate attention recommended")
                    elif med > 0:
                        st.warning("MEDIUM severity — monitor and schedule maintenance")
                    else:
                        st.success("LOW severity — routine monitoring sufficient")
                else:
                    st.info("No defects detected at current threshold.")

        st.markdown("---")
        st.subheader("Detailed Detections")

        for model_key, label in [("rtdetr", "RT-DETR-L"), ("yolo", "YOLOv26s")]:
            if model_key not in results:
                continue
            dets = results[model_key]["detections"]

            if dets:
                df = pd.DataFrame(dets)
                df["bbox_str"] = df["bbox"].apply(
                    lambda b: f"({b[0]:.0f}, {b[1]:.0f}) -> ({b[2]:.0f}, {b[3]:.0f})"
                )
                df["confidence_%"] = (df["confidence"] * 100).round(1)
                display_df = df[["class_name", "confidence_%", "severity", "bbox_str"]].copy()
                display_df.columns = ["Defect", "Confidence %", "Severity", "Bounding Box"]
                st.markdown(f"**{label}**")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"{label}: No detections at conf >= {conf}")

else:
    st.info("Upload an image or select a demo image to start detection.")
