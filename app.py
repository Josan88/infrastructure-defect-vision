"""
COS40007 Design Project — AI Structural Defect Detection Demo
RT-DETR-L vs YOLO26s: Side-by-Side Comparison

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO, RTDETR
import io, os, time, zipfile

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = {0: "crack", 1: "pothole", 2: "wall_peeling"}
CLASS_COLORS = {0: (255, 50, 50), 1: (50, 200, 50), 2: (50, 100, 255)}
CLASS_LEGEND = [
    ((255, 50, 50), "crack"),
    ((50, 200, 50), "pothole"),
    ((50, 100, 255), "wall_peeling"),
]

RTDETR_CANDIDATES = [
    os.path.join("runs", "defect_detection", "rtdetr_l_v7_iter3", "weights", "best.pt"),
    os.path.join("runs", "defect_detection", "rtdetr_l_v5_iter4", "weights", "best.pt"),
]
YOLO_CANDIDATES = [
    os.path.join("jenny", "runs", "detect", "road_damage", "unfrozen", "weights", "best.pt"),
]

DEMO_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "test", "images")
AI_DEMO_IMAGES_DIR = os.path.join(BASE_DIR, "demo_images")
DEFAULT_IMAGE_PATH = os.path.join(BASE_DIR, "default.png")

ARTIFACT_BASE = os.path.join("runs", "defect_detection")
EVIDENCE_IMAGES = [
    ("iter_comparison_training_curves.png", "Active-Learning Training Curves (v7)"),
    ("iter_comparison_pr_curves.png", "Precision-Recall Curves (v7)"),
    ("iter_comparison_confusion_matrices.png", "Confusion Matrices Across Iterations"),
    ("class_distribution.png", "Training Set Class Distribution"),
    ("iter_comparison_iou_distribution.png", "IoU Distribution Across Iterations"),
    ("iter_comparison_samples.png", "Sample Predictions Across Iterations"),
]

YOLO_EVIDENCE = [
    ("frozen_backbone", "Frozen Backbone"),
    ("partial_freeze_neck", "Partial Freeze Neck"),
    ("unfrozen", "Unfrozen"),
]

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


# ── Session State Init ──────────────────────────────────────────────────
if "demo_sel" not in st.session_state:
    st.session_state["demo_sel"] = "(none)"
if "ai_demo_sel" not in st.session_state:
    st.session_state["ai_demo_sel"] = None
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "batch_uploads" not in st.session_state:
    st.session_state["batch_uploads"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []
if "source_choice" not in st.session_state:
    st.session_state["source_choice"] = "Upload"


# ── Callbacks ───────────────────────────────────────────────────────────
def _on_upload():
    st.session_state["demo_sel"] = "(none)"
    st.session_state["ai_demo_sel"] = None


def _on_demo_change():
    st.session_state["ai_demo_sel"] = None
    st.session_state["uploader_key"] += 1


def _on_ai_demo_change():
    st.session_state["demo_sel"] = "(none)"


def _on_source_change():
    st.session_state["demo_sel"] = "(none)"
    st.session_state["ai_demo_sel"] = None
    st.session_state["uploader_key"] += 1


# ── Helpers ─────────────────────────────────────────────────────────────
def _resolve_path(candidates):
    """Return the first candidate that exists on disk, else the first."""
    for p in candidates:
        full = os.path.join(BASE_DIR, p)
        if os.path.isfile(full):
            return full
    return os.path.join(BASE_DIR, candidates[0])


def _classify_confidence(conf: float) -> str:
    if conf >= 0.7:
        return "High confidence"
    elif conf >= 0.4:
        return "Medium confidence"
    return "Low confidence"


def _draw_boxes_on_pil(image: Image.Image, detections: list) -> Image.Image:
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id = det["class_id"]
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tw = len(label) * 8 + 10
        if y1 >= 24:
            draw.rectangle([x1, y1 - 22, x1 + tw, y1], fill=color)
            draw.text((x1 + 4, y1 - 20), label, fill="white")
        else:
            draw.rectangle([x1, y1 + 2, x1 + tw, y1 + 24], fill=color)
            draw.text((x1 + 4, y1 + 4), label, fill="white")
    return draw_img


def _make_composite(left_img: Image.Image, right_img: Image.Image,
                    left_label: str, right_label: str) -> Image.Image:
    w1, h1 = left_img.size
    w2, h2 = right_img.size
    max_h = max(h1, h2)
    total_w = w1 + w2 + 20
    composite = Image.new("RGB", (total_w, max_h + 40), (255, 255, 255))
    composite.paste(left_img, (0, 40))
    composite.paste(right_img, (w1 + 20, 40))
    draw = ImageDraw.Draw(composite)
    draw.text((10, 10), left_label, fill=(0, 0, 0))
    draw.text((w1 + 30, 10), right_label, fill=(0, 0, 0))
    return composite


@st.cache_data
def _read_iterlog():
    csv_path = os.path.join(BASE_DIR, ARTIFACT_BASE, "rtdetr_l_v7_iterlog.csv")
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def _read_yolo_last_row(variant: str):
    csv_path = os.path.join(BASE_DIR, "jenny", "runs", "detect", "road_damage",
                            variant, "results.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            return df.iloc[-1].to_dict()
    return None


@st.cache_data
def _read_yolo_summary():
    rows = []
    for variant, label in YOLO_EVIDENCE:
        row = _read_yolo_last_row(variant)
        if row:
            rows.append({"Variant": label, **{k: v for k, v in row.items() if k in [
                "epoch", "metrics/precision(B)", "metrics/recall(B)",
                "metrics/mAP50(B)", "metrics/mAP50-95(B)"
            ]}})
    return pd.DataFrame(rows) if rows else None


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
        models["yolo_name"] = "YOLO26s"
        models["yolo_path"] = yolo_path
    except Exception as e:
        models["yolo"] = None
        models["yolo_error"] = str(e)

    zero_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for key in ("rtdetr", "yolo"):
        if models.get(key) is not None:
            try:
                models[key].predict(source=zero_img, verbose=False)
            except Exception:
                pass

    return models


def run_inference(model, image: Image.Image, conf: float):
    results = model.predict(source=image, conf=conf, verbose=False)
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
            "severity": _classify_confidence(conf_val),
        })
    return {"detections": detections}


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

    st.divider()

    ai_demo_files = []
    if os.path.isdir(AI_DEMO_IMAGES_DIR):
        ai_demo_files = sorted(
            f for f in os.listdir(AI_DEMO_IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    demo_files = []
    if os.path.isdir(DEMO_IMAGES_DIR):
        demo_files = sorted(
            f for f in os.listdir(DEMO_IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    source_options = ["Upload"]
    if ai_demo_files:
        source_options.append("Curated Gallery")
    if demo_files:
        source_options.append("Test Set")

    source_choice = st.radio(
        "Image source",
        source_options,
        horizontal=True,
        key="source_choice",
        on_change=_on_source_change,
    )

    st.divider()
    st.markdown("**Models**")
    models = load_models()

    if models.get("rtdetr"):
        st.success("RT-DETR-L loaded")
    else:
        st.error("RT-DETR-L not loaded")
        st.caption(models.get("rtdetr_error", ""))

    if models.get("yolo"):
        st.success("YOLO26s loaded")
    else:
        st.error("YOLO26s not loaded")
        st.caption(models.get("yolo_error", ""))

    st.divider()
    st.caption("**Provenance:**")
    if models.get("rtdetr_path"):
        rtdetr_size = os.path.getsize(models["rtdetr_path"]) / (1024 * 1024)
        st.caption(f"RT-DETR-L: {rtdetr_size:.1f} MB — v7 iter3 (mAP50 0.613)")
    if models.get("yolo_path"):
        yolo_size = os.path.getsize(models["yolo_path"]) / (1024 * 1024)
        st.caption(f"YOLO26s: {yolo_size:.1f} MB — unfrozen (mAP50 0.816)")

    st.divider()
    st.caption("COS40007 Design Project — Theme 2")


st.title("Structural Defect Detection")
st.markdown(
    "Real-time side-by-side comparison of **RT-DETR-L** (transformer) and "
    "**YOLO26s** (CNN) on infrastructure defect images."
)


# ── Tabs ────────────────────────────────────────────────────────────────
tab_detect, tab_method, tab_evidence = st.tabs(["Detect", "Methodology", "Evidence"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB: Detect
# ═══════════════════════════════════════════════════════════════════════════
with tab_detect:
    both_loaded = models.get("rtdetr") is not None and models.get("yolo") is not None
    either_loaded = models.get("rtdetr") is not None or models.get("yolo") is not None

    if not either_loaded:
        st.error(
            "Neither model could be loaded. The demo requires at least one model. "
            "Check that the weight files exist in the expected paths."
        )
        st.stop()

    st.subheader("Source Image")

    uploaded_file = None

    if source_choice == "Upload":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key=f"uploader_{st.session_state['uploader_key']}",
            on_change=_on_upload,
            accept_multiple_files=False,
        )

    elif source_choice == "Curated Gallery":
        if not ai_demo_files:
            st.info("No curated demo images found in `demo_images/`.")
        else:
            thumbs = [
                (f, os.path.join(AI_DEMO_IMAGES_DIR, f)) for f in ai_demo_files
            ]
            N_COLS = 4
            thumb_cols = st.columns(N_COLS)
            for idx, (fname, img_path) in enumerate(thumbs):
                with thumb_cols[idx % N_COLS]:
                    st.image(img_path, width="stretch")
                    if st.button(
                        "Use this image",
                        key=f"ai_demo_btn_{idx}",
                        width="stretch",
                    ):
                        st.session_state["ai_demo_sel"] = fname
                        st.session_state["uploader_key"] += 1
                        _on_ai_demo_change()
            st.caption("Click any image to load it into the detector.")

    elif source_choice == "Test Set":
        if not demo_files:
            st.info("Demo images not available in `dataset/test/images/`.")
        else:
            selected_demo = st.selectbox(
                "Pick a test image:",
                ["(none)"] + demo_files,
                key="demo_sel",
                on_change=_on_demo_change,
            )
            if selected_demo != "(none)":
                demo_path = os.path.join(DEMO_IMAGES_DIR, selected_demo)
                with open(demo_path, "rb") as f:
                    uploaded_file = io.BytesIO(f.read())

    if (source_choice == "Curated Gallery"
            and st.session_state.get("ai_demo_sel")):
        ai_path = os.path.join(AI_DEMO_IMAGES_DIR, st.session_state["ai_demo_sel"])
        if os.path.isfile(ai_path):
            with open(ai_path, "rb") as f:
                uploaded_file = io.BytesIO(f.read())
            st.info(f"Loaded curated image: **{st.session_state['ai_demo_sel']}**")

    if uploaded_file is None and os.path.isfile(DEFAULT_IMAGE_PATH):
        with open(DEFAULT_IMAGE_PATH, "rb") as f:
            uploaded_file = io.BytesIO(f.read())

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("---")
        st.subheader("Uploaded Image")
        st.image(image, caption="Input image", width="stretch")

        # ── Inference ───────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        results = {}

        for col, model_key, label in [
            (col1, "rtdetr", "RT-DETR-L"),
            (col2, "yolo", "YOLO26s"),
        ]:
            model = models.get(model_key)
            if model is None:
                col.error(f"{label} model not loaded")
                continue
            with st.spinner(f"Running {label} inference..."):
                t0 = time.time()
                result = run_inference(model, image, conf)
                elapsed = time.time() - t0
                result["time_ms"] = elapsed * 1000
                results[model_key] = result

        if not results:
            st.warning("No model produced results. Try lowering the confidence threshold.")
        else:
            avg_ms = np.mean([r["time_ms"] for r in results.values()])
            st.caption(f"Average inference time: {avg_ms:.0f} ms")

        # ── Quantitative Comparison Panel ───────────────────────────────
        if results:
            st.markdown("---")
            st.subheader("Model Comparison")

            iterlog = _read_iterlog()
            if iterlog is not None and len(iterlog) > 0:
                peak = iterlog.loc[iterlog["mAP50"].idxmax()]
                rtdetr_row = {
                    "Model": "RT-DETR-L",
                    "Architecture": "Transformer (encoder-decoder)",
                    "mAP50": f"{peak['mAP50']:.3f}",
                    "mAP50-95": f"{peak['mAP50_95']:.3f}",
                    "Precision": f"{peak['precision']:.3f}",
                    "Recall": f"{peak['recall']:.3f}",
                    "Parameters": "32M",
                    "Best Epoch": int(peak["best_epoch"]),
                    "Peak Iter": str(peak["iter"]),
                }
            else:
                rtdetr_row = {
                    "Model": "RT-DETR-L", "Architecture": "Transformer",
                    "Parameters": "32M",
                }

            yolo_df = _read_yolo_summary()
            if yolo_df is not None and len(yolo_df) > 0:
                unfrozen = yolo_df[yolo_df["Variant"] == "Unfrozen"]
                if len(unfrozen) > 0:
                    u = unfrozen.iloc[0]
                    yolo_row = {
                        "Model": "YOLO26s",
                        "Architecture": "CNN (single-stage)",
                        "mAP50": f"{u.get('metrics/mAP50(B)', 0):.3f}",
                        "mAP50-95": f"{u.get('metrics/mAP50-95(B)', 0):.3f}",
                        "Precision": f"{u.get('metrics/precision(B)', 0):.3f}",
                        "Recall": f"{u.get('metrics/recall(B)', 0):.3f}",
                        "Parameters": "11M",
                        "Best Epoch": int(u.get("epoch", 0)),
                        "Peak Iter": "-",
                    }
                else:
                    yolo_row = {"Model": "YOLO26s", "Architecture": "CNN", "Parameters": "11M"}
            else:
                yolo_row = {"Model": "YOLO26s", "Architecture": "CNN", "Parameters": "11M"}

            comp_df = pd.DataFrame([rtdetr_row, yolo_row])
            st.dataframe(comp_df, width="stretch", hide_index=True)

            if "time_ms" in results.get("rtdetr", {}) or "time_ms" in results.get("yolo", {}):
                ms_data = []
                for mk, lbl in [("rtdetr", "RT-DETR-L"), ("yolo", "YOLO26s")]:
                    if mk in results:
                        ms_data.append({"Model": lbl, "Inference (ms)": f"{results[mk]['time_ms']:.0f}"})
                if ms_data:
                    ms_df = pd.DataFrame(ms_data)
                    st.caption("Measured inference on this image:")
                    st.dataframe(ms_df, width="stretch", hide_index=True)

            st.info(
                "Confidence threshold does not affect mAP — it only filters which detections are shown. "
                "Lower threshold = more detections (higher recall, lower precision). "
                "The peak mAP50 shown above is the best val-set result from training."
            )

        # ── Side-by-Side Annotated Images ───────────────────────────────
        if results:
            st.markdown("---")
            st.subheader("Annotated Images")

            col_r, col_y = st.columns(2)
            for col, model_key, label in [
                (col_r, "rtdetr", "RT-DETR-L"),
                (col_y, "yolo", "YOLO26s"),
            ]:
                if model_key not in results:
                    continue
                r = results[model_key]
                det_img = _draw_boxes_on_pil(image, r["detections"])

                with col:
                    st.markdown(f"### {label}")
                    st.image(det_img, caption=f"{label} detections",
                             width="stretch")
                    st.caption(
                        f"Inference: {r['time_ms']:.0f} ms | "
                        f"Detections: {len(r['detections'])}"
                    )

        # ── Class Legend (always visible) ────────────────────────────────
        if results:
            legend_html = "  ".join(
                f'<span style="display:inline-block;width:14px;height:14px;'
                f'background:rgb({c[0]},{c[1]},{c[2]});border-radius:3px;'
                f'margin-right:4px;vertical-align:middle;"></span>{n}'
                for c, n in CLASS_LEGEND
            )
            st.markdown(f"**Class legend:** {legend_html}", unsafe_allow_html=True)

        # ── Detection Analytics ──────────────────────────────────────────
        if results:
            st.markdown("---")
            st.subheader("Detection Analytics")

            analytics_cols = st.columns(2)
            for col, model_key, label in [
                (analytics_cols[0], "rtdetr", "RT-DETR-L"),
                (analytics_cols[1], "yolo", "YOLO26s"),
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
                        high = sum(1 for d in dets if d["severity"] == "High confidence")
                        med = sum(1 for d in dets if d["severity"] == "Medium confidence")
                        low = sum(1 for d in dets if d["severity"] == "Low confidence")

                        st.metric("Avg Confidence", f"{avg_conf:.2f}")
                        s1, s2, s3 = st.columns(3)
                        s1.metric("High (\u2265 0.70)", high)
                        s2.metric("Medium (\u2265 0.40)", med)
                        s3.metric("Low (< 0.40)", low)

                        if high > 0:
                            st.info(
                                "Detections with confidence \u2265 0.70 — "
                                "recommended for prioritised review"
                            )
                        elif med > 0:
                            st.info(
                                "Detections with confidence \u2265 0.40 — "
                                "moderate certainty; review recommended"
                            )
                        else:
                            st.info(
                                "All detections below 0.40 confidence — "
                                "consider lowering threshold"
                            )
                    else:
                        st.info("No defects detected at current threshold.")

        # ── Detailed Detections ──────────────────────────────────────────
        if results:
            st.markdown("---")
            st.subheader("Detailed Detections")

            for model_key, label in [("rtdetr", "RT-DETR-L"), ("yolo", "YOLO26s")]:
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
                    display_df.columns = ["Defect", "Confidence %", "Confidence Band", "Bounding Box"]
                    st.markdown(f"**{label}**")
                    st.dataframe(display_df, width="stretch", hide_index=True)
                else:
                    st.info(f"{label}: No detections at conf >= {conf}")

        # ── Export Menu ──────────────────────────────────────────────────
        if results:
            st.markdown("---")
            st.subheader("Export Results")

            export_format = st.selectbox(
                "Export format:",
                [
                    "Annotated PNG (RT-DETR-L)",
                    "Annotated PNG (YOLO26s)",
                    "Side-by-side composite PNG",
                    "Detections CSV",
                    "Detections JSON",
                ],
                key="export_format",
            )

            if export_format.startswith("Annotated PNG"):
                model_key = "rtdetr" if "RT-DETR" in export_format else "yolo"
                label = "RT-DETR-L" if model_key == "rtdetr" else "YOLO26s"
                if model_key in results:
                    det_img = _draw_boxes_on_pil(image, results[model_key]["detections"])
                    buf = io.BytesIO()
                    det_img.save(buf, format="PNG")
                    st.download_button(
                        label=f"Download {label} Annotated Image",
                        data=buf.getvalue(),
                        file_name=f"{model_key}_annotated.png",
                        mime="image/png",
                        key="dl_single",
                        width="stretch",
                    )

            elif export_format == "Side-by-side composite PNG":
                imgs = {}
                for mk, lbl in [("rtdetr", "RT-DETR-L"), ("yolo", "YOLO26s")]:
                    if mk in results:
                        imgs[mk] = (_draw_boxes_on_pil(image, results[mk]["detections"]), lbl)
                if len(imgs) == 2:
                    composite = _make_composite(
                        imgs["rtdetr"][0], imgs["yolo"][0],
                        f"RT-DETR-L ({results['rtdetr']['time_ms']:.0f} ms)",
                        f"YOLO26s ({results['yolo']['time_ms']:.0f} ms)",
                    )
                    buf = io.BytesIO()
                    composite.save(buf, format="PNG")
                    st.download_button(
                        label="Download Side-by-Side Composite",
                        data=buf.getvalue(),
                        file_name="comparison_composite.png",
                        mime="image/png",
                        key="dl_composite",
                        width="stretch",
                    )
                else:
                    st.warning("Both models must produce results for composite export.")

            elif export_format == "Detections CSV":
                all_rows = []
                for mk, lbl in [("rtdetr", "RT-DETR-L"), ("yolo", "YOLO26s")]:
                    if mk in results:
                        for d in results[mk]["detections"]:
                            all_rows.append({
                                "model": lbl,
                                "class": d["class_name"],
                                "confidence": round(d["confidence"], 4),
                                "confidence_band": d["severity"],
                                "bbox_x1": round(d["bbox"][0], 1),
                                "bbox_y1": round(d["bbox"][1], 1),
                                "bbox_x2": round(d["bbox"][2], 1),
                                "bbox_y2": round(d["bbox"][3], 1),
                            })
                if all_rows:
                    csv_df = pd.DataFrame(all_rows)
                    csv_buf = io.BytesIO()
                    csv_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="Download Detections CSV",
                        data=csv_buf.getvalue(),
                        file_name="detections.csv",
                        mime="text/csv",
                        key="dl_csv",
                        width="stretch",
                    )

            elif export_format == "Detections JSON":
                all_data = {}
                for mk, lbl in [("rtdetr", "RT-DETR-L"), ("yolo", "YOLO26s")]:
                    if mk in results:
                        all_data[mk] = {
                            "model": lbl,
                            "inference_ms": round(results[mk]["time_ms"], 1),
                            "detections": [
                                {
                                    "class": d["class_name"],
                                    "confidence": round(d["confidence"], 4),
                                    "confidence_band": d["severity"],
                                    "bbox": [round(v, 1) for v in d["bbox"]],
                                }
                                for d in results[mk]["detections"]
                            ],
                        }
                if all_data:
                    json_buf = io.BytesIO()
                    pd.io.json.dumps = None
                    import json
                    json_buf.write(json.dumps(all_data, indent=2).encode())
                    st.download_button(
                        label="Download Detections JSON",
                        data=json_buf.getvalue(),
                        file_name="detections.json",
                        mime="application/json",
                        key="dl_json",
                        width="stretch",
                    )

        # ── Detection History ────────────────────────────────────────────
        if results:
            st.session_state["history"].append({
                "n_dets": {mk: len(results[mk]["detections"]) for mk in results},
            })
            if len(st.session_state["history"]) > 10:
                st.session_state["history"] = st.session_state["history"][-10:]

        # ── Batch Export ─────────────────────────────────────────────────
        if len(st.session_state["batch_uploads"]) > 1:
            st.markdown("---")
            st.subheader("Batch Export")
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for mk in ("rtdetr", "yolo"):
                    if mk in results:
                        det_img = _draw_boxes_on_pil(image, results[mk]["detections"])
                        img_buf = io.BytesIO()
                        det_img.save(img_buf, format="PNG")
                        zf.writestr(f"{mk}_annotated.png", img_buf.getvalue())
                if "rtdetr" in results and "yolo" in results:
                    composite = _make_composite(
                        _draw_boxes_on_pil(image, results["rtdetr"]["detections"]),
                        _draw_boxes_on_pil(image, results["yolo"]["detections"]),
                        "RT-DETR-L", "YOLO26s",
                    )
                    comp_buf = io.BytesIO()
                    composite.save(comp_buf, format="PNG")
                    zf.writestr("composite.png", comp_buf.getvalue())
            st.download_button(
                label="Download All Results (ZIP)",
                data=zip_buf.getvalue(),
                file_name="detection_results.zip",
                mime="application/zip",
                key="dl_zip",
                width="stretch",
            )

    else:
        st.info("Upload an image or select a demo image to start detection.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB: Methodology
# ═══════════════════════════════════════════════════════════════════════════
with tab_method:
    st.header("System Methodology")

    st.subheader("Problem Context")
    st.markdown(
        "This system detects structural defects in civil infrastructure using "
        "deep learning object detection. The demo compares two fundamentally "
        "different architectures — a transformer-based encoder-decoder (RT-DETR-L) "
        "and a single-stage CNN (YOLO26s) — to evaluate the trade-off between "
        "detection accuracy and inference speed."
    )

    st.subheader("Dataset")
    st.markdown(
        "A custom YOLO-format dataset of **780 images** (train 546 / val 117 / test 117) "
        "with 3 defect classes: **crack** (linear fractures), **pothole** (surface depressions), "
        "and **wall_peeling** (surface delamination). All images are resized to 640×640 for "
        "training and inference."
    )

    st.subheader("Models")
    st.markdown(
        "| Model | Architecture | Params | Training | Peak mAP50 |\n"
        "|-------|-------------|--------|----------|------------|\n"
        "| **RT-DETR-L** | Transformer encoder-decoder | 32M | 6-iter active learning, 50 epochs/iter | 0.613 (iter 3) |\n"
        "| **YOLO26s** | CNN single-stage | 11M | 300 epochs, unfrozen backbone | 0.816 |"
    )

    st.subheader("RT-DETR-L Active-Learning Loop")
    st.markdown(
        "The RT-DETR-L training uses a 6-iteration active-learning loop where each "
        "iteration adds 28 new training images and resumes from the previous iteration's "
        "checkpoint. Performance peaked at **iteration 3** (484 images, mAP50 = 0.613) "
        "and plateaued thereafter — a common pattern with transformer-based detectors "
        "on small datasets where additional data introduces label noise rather than "
        "useful signal."
    )

    st.subheader("YOLO26s Ablation Study")
    st.markdown(
        "Three freeze-strategy variants were trained to evaluate transfer learning:"
    )
    st.markdown(
        "- **Frozen Backbone**: Pre-trained backbone weights locked; only the detection head is trained.\n"
        "- **Partial Freeze Neck**: Neck (feature pyramid) layers partially frozen.\n"
        "- **Unfrozen**: All layers trainable; best overall mAP50 (0.816) but lower recall."
    )
    st.markdown(
        "The unfrozen variant was selected for the demo as it achieves the highest "
        "mAP50 among the three."
    )

    st.subheader("Confidence Threshold")
    st.markdown(
        "The sidebar slider controls the minimum detection confidence. Lower thresholds "
        "increase recall (catch more defects) but also increase false positives. "
        "The default of **0.25** is a reasonable starting point; adjust based on the "
        "use case."
    )

    st.info(
        "**Note on confidence vs. severity:** The confidence bands (High/Medium/Low) shown "
        "in the analytics panel reflect the model's confidence in each detection, not the "
        "actual severity of the defect. In production, defect severity would be assessed "
        "using additional geometric and environmental features."
    )

    st.subheader("Architecture Overview")
    st.markdown(
        """
        <svg width="720" height="380" viewBox="0 0 720 380" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" fill="#666"/>
            </marker>
          </defs>
          <rect x="10" y="155" width="110" height="50" rx="8" fill="#e8f0fe" stroke="#4285f4" stroke-width="1.5"/>
          <text x="65" y="177" text-anchor="middle" font-size="11" fill="#333">Image Input</text>
          <text x="65" y="192" text-anchor="middle" font-size="10" fill="#666">(upload / demo)</text>

          <line x1="120" y1="180" x2="165" y2="180" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

          <rect x="170" y="155" width="110" height="50" rx="8" fill="#fef3e0" stroke="#f9a825" stroke-width="1.5"/>
          <text x="225" y="177" text-anchor="middle" font-size="11" fill="#333">Preprocess</text>
          <text x="225" y="192" text-anchor="middle" font-size="10" fill="#666">resize 640x640</text>

          <line x1="280" y1="170" x2="320" y2="100" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
          <line x1="280" y1="190" x2="320" y2="260" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

          <rect x="325" y="65" width="130" height="55" rx="8" fill="#e8f5e9" stroke="#43a047" stroke-width="1.5"/>
          <text x="390" y="87" text-anchor="middle" font-size="11" fill="#333">RT-DETR-L</text>
          <text x="390" y="102" text-anchor="middle" font-size="10" fill="#666">Transformer | 32M</text>

          <rect x="325" y="235" width="130" height="55" rx="8" fill="#e3f2fd" stroke="#1e88e5" stroke-width="1.5"/>
          <text x="390" y="257" text-anchor="middle" font-size="11" fill="#333">YOLO26s</text>
          <text x="390" y="272" text-anchor="middle" font-size="10" fill="#666">CNN | 11M</text>

          <line x1="455" y1="93" x2="510" y2="140" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
          <line x1="455" y1="263" x2="510" y2="215" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

          <rect x="515" y="155" width="100" height="50" rx="8" fill="#fce4ec" stroke="#e53935" stroke-width="1.5"/>
          <text x="565" y="177" text-anchor="middle" font-size="11" fill="#333">Detections</text>
          <text x="565" y="192" text-anchor="middle" font-size="10" fill="#666">bbox + class</text>

          <line x1="615" y1="180" x2="655" y2="180" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

          <rect x="660" y="155" width="55" height="50" rx="8" fill="#f3e5f5" stroke="#8e24aa" stroke-width="1.5"/>
          <text x="687" y="177" text-anchor="middle" font-size="11" fill="#333">Analytics</text>
          <text x="687" y="192" text-anchor="middle" font-size="10" fill="#666">& export</text>

          <text x="390" y="345" text-anchor="middle" font-size="10" fill="#999">
            Trained on 546 images (YOLO-format) | Tested on 117 images | Active-learning loop (RT-DETR-L)
          </text>
        </svg>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB: Evidence
# ═══════════════════════════════════════════════════════════════════════════
with tab_evidence:
    st.header("Training & Evaluation Evidence")

    st.subheader("RT-DETR-L Active-Learning Iterlog")
    iterlog = _read_iterlog()
    if iterlog is not None:
        st.dataframe(
            iterlog.style.highlight_max(
                subset=["mAP50", "mAP50_95", "precision", "recall"],
                color="#c8e6c9",
            ),
            width="stretch",
            hide_index=True,
        )
        peak = iterlog.loc[iterlog["mAP50"].idxmax()]
        st.info(
            f"Peak mAP50 = **{peak['mAP50']:.4f}** at **iter {int(peak['iter'])}** "
            f"({int(peak['cum_train_imgs'])} training images, best epoch {int(peak['best_epoch'])})."
        )
    else:
        st.warning("Iterlog not found at `runs/defect_detection/rtdetr_l_v7_iterlog.csv`.")

    st.subheader("YOLO26s Ablation Comparison")
    yolo_df = _read_yolo_summary()
    if yolo_df is not None and len(yolo_df) > 0:
        st.dataframe(
            yolo_df.style.highlight_max(
                subset=[c for c in yolo_df.columns if c.startswith("metrics/")],
                color="#c8e6c9",
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("YOLO results CSVs not found.")

    st.subheader("RT-DETR-L Training Artefacts")
    for fname, caption in EVIDENCE_IMAGES:
        img_path = os.path.join(BASE_DIR, ARTIFACT_BASE, fname)
        if os.path.isfile(img_path):
            with st.expander(caption, expanded=False):
                st.image(img_path, width="stretch", caption=caption)
        else:
            st.caption(f"Artefact not found: {fname}")

    st.subheader("YOLO26s Ablation Artefacts")
    for variant, label in YOLO_EVIDENCE:
        with st.expander(label, expanded=False):
            for artefact in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
                img_path = os.path.join(
                    BASE_DIR, "jenny", "runs", "detect", "road_damage",
                    variant, artefact,
                )
                if os.path.isfile(img_path):
                    st.image(img_path, width="stretch",
                             caption=f"{label} — {artefact.replace('_', ' ').replace('.png', '')}")

    st.subheader("Test-Set Confusion Matrices (RT-DETR-L)")
    for iter_idx in range(6):
        cm_path = os.path.join(
            BASE_DIR, ARTIFACT_BASE,
            f"rtdetr_l_v7_iter{iter_idx}_test",
            "confusion_matrix_normalized.png",
        )
        if os.path.isfile(cm_path):
            with st.expander(f"Iteration {iter_idx} confusion matrix", expanded=False):
                st.image(cm_path, width="stretch",
                         caption=f"RT-DETR-L iter {iter_idx} — normalised confusion matrix")
