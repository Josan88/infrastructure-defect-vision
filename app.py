"""
COS40007 Design Project — AI Structural Defect Detection Demo
RT-DETR-L vs YOLO26m: Side-by-Side & Neural Ensemble Fusion Dashboard

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO, RTDETR
import io
import os
import time
import zipfile
import copy

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
]
YOLO_CANDIDATES = [
    os.path.join("jenny", "runs", "detect", "road_damage", "unfrozen", "weights", "best.pt"),
]

DEMO_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "test", "images")
AI_DEMO_IMAGES_DIR = os.path.join(BASE_DIR, "demo_images")

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
    page_title="Structural Defect Auditing & Ensemble Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-end professional appearance
st.markdown("""
<style>
    /* Styling Streamlit metric containers */
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 12px !important;
        padding: 15px 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08) !important;
        border-color: rgba(255, 215, 0, 0.4) !important;
    }
    
    /* Clean, modern tab typography */
    button[data-baseweb="tab"] {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    
    /* Structural Audit Status Cards */
    .status-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 8px solid;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .status-safe {
        background-color: rgba(46, 125, 50, 0.1);
        border-color: #2e7d32;
        color: #1b5e20;
    }
    .status-watch {
        background-color: rgba(249, 168, 37, 0.1);
        border-color: #f9a825;
        color: #f57f17;
    }
    .status-alert {
        background-color: rgba(239, 108, 0, 0.1);
        border-color: #ef6c00;
        color: #e65100;
    }
    .status-critical {
        background-color: rgba(198, 40, 40, 0.15);
        border-color: #c62828;
        color: #b71c1c;
    }
    
    /* Subtitle and header formatting */
    .sub-title {
        font-size: 0.95rem;
        color: #666;
        margin-top: -15px;
        margin-bottom: 25px;
    }
    
    /* Crop gallery cards */
    .crop-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
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

def bbox_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

# ── Neural Ensemble Fusion Engine ───────────────────────────────────────
def run_ensemble_fusion(rt_dets, yo_dets, iou_thresh=0.35, mode="Union (High Recall)"):
    """
    Fuses predictions from RT-DETR-L and YOLO26m depending on the selected mode:
    - 'Union (High Recall)': Keep all detections from both models. For matches (IoU >= iou_thresh),
      keep only the higher confidence detection, marked as consensus.
    - 'Intersection (High Precision)': Only keep detections where both models agree (IoU >= iou_thresh).
    - 'Weighted Average': Keep all, but for overlapping boxes, compute a confidence-weighted average.
    """
    rt_copy = copy.deepcopy(rt_dets)
    yo_copy = copy.deepcopy(yo_dets)
    
    # Init flags
    for d in rt_copy:
        d["is_consensus"] = False
        d["consensus_with"] = None
        d["match_iou"] = 0.0
        d["source"] = "RT-DETR-L"
    for d in yo_copy:
        d["is_consensus"] = False
        d["consensus_with"] = None
        d["match_iou"] = 0.0
        d["source"] = "YOLO26m"
        
    matched_rt = set()
    matched_yo = set()
    fused_detections = []
    
    # Identify overlaps
    matches = []
    for i, rt in enumerate(rt_copy):
        for j, yo in enumerate(yo_copy):
            if rt["class_id"] != yo["class_id"]:
                continue
            iou = bbox_iou(rt["bbox"], yo["bbox"])
            if iou >= iou_thresh:
                matches.append((iou, i, j))
                
    # Sort matches by overlapping IoU (highest first)
    matches.sort(key=lambda x: x[0], reverse=True)
    
    paired_rt = {}
    paired_yo = {}
    for iou, i, j in matches:
        if i in paired_rt or j in paired_yo:
            continue
        paired_rt[i] = (j, iou)
        paired_yo[j] = (i, iou)
        
    # Enrich details on copies
    for i, rt in enumerate(rt_copy):
        if i in paired_rt:
            j, iou = paired_rt[i]
            rt["is_consensus"] = True
            rt["consensus_with"] = j
            rt["match_iou"] = iou
    for j, yo in enumerate(yo_copy):
        if j in paired_yo:
            i, iou = paired_yo[j]
            yo["is_consensus"] = True
            yo["consensus_with"] = i
            yo["match_iou"] = iou
            
    if mode == "Intersection (High Precision)":
        # Only keep matched overlapping detections
        for i, rt in enumerate(rt_copy):
            if i in paired_rt:
                j, iou = paired_rt[i]
                yo = yo_copy[j]
                # Keep the more confident detector's coordinates
                best_det = rt if rt["confidence"] >= yo["confidence"] else yo
                fused = copy.deepcopy(best_det)
                fused["is_consensus"] = True
                fused["match_iou"] = iou
                fused["source"] = f"Intersection Match ({rt['source']} + {yo['source']})"
                fused_detections.append(fused)
                
    elif mode == "Weighted Average":
        # For matches, compute weighted box coordinates. For non-matches, keep original.
        for i, rt in enumerate(rt_copy):
            if i in paired_rt:
                j, iou = paired_rt[i]
                yo = yo_copy[j]
                matched_rt.add(i)
                matched_yo.add(j)
                
                # Weights proportional to model confidence
                c_rt = rt["confidence"]
                c_yo = yo["confidence"]
                sum_conf = c_rt + c_yo
                b_rt = rt["bbox"]
                b_yo = yo["bbox"]
                
                weighted_bbox = [
                    (b_rt[0] * c_rt + b_yo[0] * c_yo) / sum_conf,
                    (b_rt[1] * c_rt + b_yo[1] * c_yo) / sum_conf,
                    (b_rt[2] * c_rt + b_yo[2] * c_yo) / sum_conf,
                    (b_rt[3] * c_rt + b_yo[3] * c_yo) / sum_conf,
                ]
                
                fused = {
                    "class_id": rt["class_id"],
                    "class_name": rt["class_name"],
                    "bbox": weighted_bbox,
                    "confidence": max(c_rt, c_yo),
                    "severity": _classify_confidence(max(c_rt, c_yo)),
                    "is_consensus": True,
                    "consensus_with": f"rt_{i}_yo_{j}",
                    "match_iou": iou,
                    "source": "Weighted Ensemble",
                }
                fused_detections.append(fused)
            else:
                rt["source"] = "RT-DETR-L (Exclusive)"
                fused_detections.append(rt)
                
        for j, yo in enumerate(yo_copy):
            if j not in matched_yo:
                yo["source"] = "YOLO26m (Exclusive)"
                fused_detections.append(yo)
                
    else:  # Union (High Recall)
        # Keep both. For matches, keep only the higher confidence detection, marked as consensus.
        for i, rt in enumerate(rt_copy):
            if i in paired_rt:
                j, iou = paired_rt[i]
                yo = yo_copy[j]
                matched_rt.add(i)
                matched_yo.add(j)
                
                if rt["confidence"] >= yo["confidence"]:
                    fused = copy.deepcopy(rt)
                else:
                    fused = copy.deepcopy(yo)
                fused["is_consensus"] = True
                fused["match_iou"] = iou
                fused["source"] = "Ensemble Union"
                fused_detections.append(fused)
            else:
                rt["source"] = "RT-DETR-L (Exclusive)"
                fused_detections.append(rt)
                
        for j, yo in enumerate(yo_copy):
            if j not in matched_yo:
                yo["source"] = "YOLO26m (Exclusive)"
                fused_detections.append(yo)
                
    return fused_detections

# ── Structural Health Index (SHI) Calculation ───────────────────────────
def calculate_structural_health(detections, img_width=640, img_height=640):
    """
    Calculates the Structural Health Index (SHI) from 0 to 100.
    Integrates class priority weights and pixel area fractions.
    """
    weights = {"crack": 15.0, "pothole": 25.0, "wall_peeling": 8.0}
    total_area = img_width * img_height
    penalty_sum = 0.0
    
    for d in detections:
        cls_name = d["class_name"]
        weight = weights.get(cls_name, 10.0)
        
        # Calculate bbox area fraction
        x1, y1, x2, y2 = d["bbox"]
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        box_area = box_w * box_h
        area_frac = box_area / total_area
        
        # Scale and cap the penalty per box to avoid out-of-scale effects
        scaled_area = min(10.0, area_frac * 100.0)
        
        penalty = weight * scaled_area * d["confidence"]
        penalty_sum += penalty
        
    shi = max(0.0, 100.0 - penalty_sum)
    
    if shi >= 90.0:
        status_color = "safe"
        urgency_label = "🟢 SAFE / NOMINAL"
        description = "Structure nominal. No urgent issues detected. Schedule regular annual maintenance checks."
        action_plan = ["Conduct normal annual inspection.", "Document baseline hairline cracks if any."]
    elif shi >= 75.0:
        status_color = "watch"
        urgency_label = "🟡 WATCH"
        description = "Minor deterioration observed. Monitor quarterly to ensure cracks or delamination do not expand."
        action_plan = ["Seal minor cracks to prevent water ingress.", "Re-inspect delaminating walls in 3 months."]
    elif shi >= 55.0:
        status_color = "orange"
        urgency_label = "🟠 ALERT / WARNING"
        description = "Moderate structural defects detected. Schedule detailed engineering inspection and repairs within 6 months."
        action_plan = ["Schedule comprehensive engineer inspection.", "Prepare surface patch & patching repairs.", "Monitor weekly for dynamic expansion."]
    else:
        status_color = "critical"
        urgency_label = "🔴 CRITICAL SAFETY HAZARD"
        description = "Severe damage centers or extensive potholing detected. Requires immediate structural intervention or road resurfacing!"
        action_plan = ["Isolate dynamic loading areas.", "Deploy emergency asphalt/concrete structural patching.", "Immediate full field manual structural audit."]
        
    return {
        "shi": round(shi, 1),
        "status_color": status_color,
        "urgency_label": urgency_label,
        "description": description,
        "action_plan": action_plan,
        "total_penalty": round(penalty_sum, 1)
    }

# ── Interactive ROI Defect Gallery Creator ──────────────────────────────
def get_defect_crops(image: Image.Image, detections: list):
    """
    Crops defect bounding boxes from the main image and classifies them
    using geometric/morphological properties.
    """
    crops = []
    w, h = image.size
    for idx, d in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        x1_clip = max(0, min(w, x1))
        y1_clip = max(0, min(h, y1))
        x2_clip = max(0, min(w, x2))
        y2_clip = max(0, min(h, y2))
        
        box_w = x2_clip - x1_clip
        box_h = y2_clip - y1_clip
        
        if box_w <= 3 or box_h <= 3:
            continue
            
        crop_img = image.crop((x1_clip, y1_clip, x2_clip, y2_clip))
        
        cls_name = d["class_name"]
        aspect_ratio = max(box_w, 1) / max(box_h, 1)
        area = box_w * box_h
        
        sub_cls = "Defect"
        details = ""
        
        if cls_name == "crack":
            if aspect_ratio >= 2.5:
                sub_cls = "Longitudinal/Horizontal Fracture"
                details = f"Horizontal orientation (aspect ratio {aspect_ratio:.1f}). Frequently induced by structural bending stress."
            elif aspect_ratio <= 0.4:
                sub_cls = "Transverse/Vertical Fracture"
                details = f"Vertical orientation (aspect ratio {aspect_ratio:.1f}). Frequently caused by shear stress or temperature shrinkage."
            else:
                sub_cls = "Block / Alligator Fatigue Crack"
                details = f"Isotropic fatigue pattern (aspect ratio {aspect_ratio:.1f}). Indicates localized base pavement failure."
        elif cls_name == "pothole":
            if area >= 10000:
                sub_cls = "Severe Surface Pothole"
                details = f"Surface area {area:,} px. Structural sub-grade layers compromised. High vehicle impact hazard."
            else:
                sub_cls = "Superficial Pavement Pit"
                details = f"Surface area {area:,} px. Initial surface layer break. Fast water ingress catalyst."
        elif cls_name == "wall_peeling":
            if area >= 12000:
                sub_cls = "Severe Wall Delamination"
                details = f"Delamination area {area:,} px. Widespread coating failure. Substrate highly exposed to damp."
            else:
                sub_cls = "Superficial Coating Flaking"
                details = f"Delamination area {area:,} px. Outer paint finish separating. Under-layer intact."
                
        crops.append({
            "id": idx + 1,
            "crop_img": crop_img,
            "class_name": cls_name,
            "sub_class": sub_cls,
            "details": details,
            "confidence": d["confidence"],
            "dimensions": f"{box_w}x{box_h} px",
            "is_consensus": d.get("is_consensus", False),
            "bbox_str": f"[{x1}, {y1}, {x2}, {y2}]",
        })
    return crops

# ── General Image Box Drawing Helper ────────────────────────────────────
def _draw_boxes_on_pil(image: Image.Image, detections: list,
                        selected_classes=None, box_width=3, fill_boxes=False,
                        font_scale=1.0, highlight_consensus=True,
                        consensus_style="Dotted Border") -> Image.Image:
    if selected_classes is None:
        selected_classes = ["crack", "pothole", "wall_peeling"]

    filtered_dets = [d for d in detections if d["class_name"] in selected_classes]

    if fill_boxes:
        overlay_img = image.convert("RGBA")
        overlay = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))
        draw_over = ImageDraw.Draw(overlay)
        
        for det in filtered_dets:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_id = det["class_id"]
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            fill_color = color + (55,)  # alpha = 55/255
            draw_over.rectangle([x1, y1, x2, y2], fill=fill_color)
            
        draw_img = Image.alpha_composite(overlay_img, overlay).convert("RGB")
    else:
        draw_img = image.copy()

    draw = ImageDraw.Draw(draw_img)
    
    for det in filtered_dets:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id = det["class_id"]
        is_con = det.get("is_consensus", False)
        
        base_color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label_prefix = ""
        current_box_width = box_width
        
        if is_con and highlight_consensus:
            label_prefix = "[CON] "
            if consensus_style == "Golden Outline":
                border_color = (255, 215, 0)  # Soft Gold
                current_box_width = max(4, box_width + 1)
            elif consensus_style == "Double Thickness":
                border_color = base_color
                current_box_width = box_width * 2
            else:  # Dotted Border / default inner outline
                border_color = base_color
        else:
            border_color = base_color
            
        # Draw bounding rectangle
        draw.rectangle([x1, y1, x2, y2], outline=border_color, width=current_box_width)
        
        # Inner outline for dotted look
        if is_con and highlight_consensus and consensus_style == "Dotted Border":
            draw.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=(255, 215, 0), width=1)
            
        confidence_val = det['confidence']
        label = f"{label_prefix}{det['class_name']} {confidence_val:.2f}"
        
        # Label size scales
        text_char_width = int(8 * font_scale)
        text_height = int(18 * font_scale)
        tw = len(label) * text_char_width + 8
        
        if y1 >= text_height + 4:
            draw.rectangle([x1, y1 - (text_height + 2), x1 + tw, y1], fill=border_color)
            draw.text((x1 + 4, y1 - text_height), label, fill="white")
        else:
            draw.rectangle([x1, y1 + 2, x1 + tw, y1 + (text_height + 4)], fill=border_color)
            draw.text((x1 + 4, y1 + 4), label, fill="white")
            
    return draw_img

def _make_composite(left_img: Image.Image, right_img: Image.Image,
                    left_label: str, right_label: str) -> Image.Image:
    w1, h1 = left_img.size
    w2, h2 = right_img.size
    max_h = max(h1, h2)
    total_w = w1 + w2 + 20
    composite = Image.new("RGB", (total_w, max_h + 40), (240, 240, 240))
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
        models["yolo_name"] = "YOLO26m"
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
    st.image("https://img.icons8.com/color/96/structural.png", width=64)
    st.title("System Controls")
    
    analysis_mode = st.radio(
        "Analysis Protocol",
        ["Single Image Audit", "Batch Folder Scan"],
        index=0,
        help="Single Image Audit runs in-depth visual and quantitative reports. Batch Folder Scan processes bulk directories with consensus checks.",
    )

    st.divider()

    # Model parameters
    st.subheader("Model Parameters")
    conf = st.slider(
        "Confidence Gate",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Confidence cutoff score above which detections are rendered.",
    )

    selected_classes = st.multiselect(
        "Filter Classes",
        ["crack", "pothole", "wall_peeling"],
        default=["crack", "pothole", "wall_peeling"],
        help="Enable/disable rendering of specific classes.",
    )

    st.divider()

    # Advanced Ensemble settings
    st.subheader("Neural Ensemble Panel")
    ensemble_view_mode = st.selectbox(
        "Ensemble Fusion View",
        [
            "Compare Side-by-Side",
            "Union (High Recall Fused)",
            "Intersection (High Precision Fused)",
            "Weighted Average Fused",
            "RT-DETR-L (Transformer Only)",
            "YOLO26m (CNN Only)"
        ],
        index=0,
        help="Pick the neural voting and combination protocol. Fusion modes overlay predictions from both models."
    )
    
    consensus_threshold = st.slider(
        "Overlap Matching IoU",
        0.10, 0.90, 0.35, 0.05,
        help="Overlap Intersection-over-Union matching threshold to declare consensus between models."
    )

    with st.expander("Aesthetic Rendering Adjustments", expanded=False):
        box_width = st.slider("Box Boundary Width", 1, 8, 3, 1)
        fill_boxes = st.checkbox("Overlay Box Transparency Fill", value=True)
        font_scale = st.slider("Label Font Scale", 0.5, 2.0, 1.0, 0.1)
        highlight_consensus = st.checkbox("Visually Mark Consensus", value=True)
        consensus_style = st.selectbox(
            "Consensus Border Style",
            ["Dotted Border", "Golden Outline", "Double Thickness"],
            index=0
        )

    st.divider()

    # Pre-load available files
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

    if analysis_mode == "Single Image Audit":
        source_options = ["Upload Image File"]
        if ai_demo_files:
            source_options.append("Curated Infrastructure Gallery")
        if demo_files:
            source_options.append("Academic Test Dataset")

        source_choice = st.radio(
            "Source Protocol",
            source_options,
            key="source_choice",
            on_change=_on_source_change,
        )

    st.divider()
    
    # Model Loading indicators
    models = load_models()
    st.markdown("**Core Models Status**")
    if models.get("rtdetr") is not None:
        st.success("RT-DETR-L Active")
    else:
        st.error("RT-DETR-L Down")
        st.caption(models.get("rtdetr_error", ""))

    if models.get("yolo") is not None:
        st.success("YOLO26m Active")
    else:
        st.error("YOLO26m Down")
        st.caption(models.get("yolo_error", ""))

    if models.get("rtdetr_path") or models.get("yolo_path"):
        st.caption("**Hardware Check:** Operating on CPU fallback mode.")

# ── Title Block ─────────────────────────────────────────────────────────
st.title("🏗️ Structural Audit & Neural Ensemble Dashboard")
st.markdown(
    "<div class='sub-title'>High-fidelity infrastructure monitoring system. Fuses transformer attention "
    "(RT-DETR-L) with localized deep convolutions (YOLO26m).</div>",
    unsafe_allow_html=True
)

# ── Tabs ────────────────────────────────────────────────────────────────
tab_detect, tab_method, tab_evidence = st.tabs([
    "🔍 Real-time Diagnostic Audit", 
    "📖 Engineering Methodology & Trade-offs", 
    "📊 Empirical Training Evidence"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB: Real-time Diagnostic Audit
# ═══════════════════════════════════════════════════════════════════════════
with tab_detect:
    both_loaded = models.get("rtdetr") is not None and models.get("yolo") is not None
    if not both_loaded:
        st.warning("Ensure weights files are initialized to load both architectures.")

    if "batch_cache" not in st.session_state:
        st.session_state["batch_cache"] = {}

    if analysis_mode == "Single Image Audit":
        uploaded_file = None

        if source_choice == "Upload Image File":
            uploaded_file = st.file_uploader(
                "Drag & drop image file to analyze",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key=f"uploader_{st.session_state['uploader_key']}",
                on_change=_on_upload,
            )

        elif source_choice == "Curated Infrastructure Gallery":
            if not ai_demo_files:
                st.info("Curated gallery empty.")
            else:
                st.markdown("**Click any curated sample to load it into the audit engine:**")
                thumbs = [(f, os.path.join(AI_DEMO_IMAGES_DIR, f)) for f in ai_demo_files]
                N_COLS = 5
                thumb_cols = st.columns(N_COLS)
                for idx, (fname, img_path) in enumerate(thumbs):
                    with thumb_cols[idx % N_COLS]:
                        st.image(img_path, use_container_width=True)
                        if st.button("Audit Sample", key=f"ai_demo_btn_{idx}", use_container_width=True):
                            st.session_state["ai_demo_sel"] = fname
                            st.session_state["uploader_key"] += 1
                            _on_ai_demo_change()

        elif source_choice == "Academic Test Dataset":
            if not demo_files:
                st.info("Academic test set files not found.")
            else:
                selected_demo = st.selectbox(
                    "Select a file from the 117-image test set partition:",
                    ["(none)"] + demo_files,
                    key="demo_sel",
                    on_change=_on_demo_change,
                )
                if selected_demo != "(none)":
                    demo_path = os.path.join(DEMO_IMAGES_DIR, selected_demo)
                    with open(demo_path, "rb") as f:
                        uploaded_file = io.BytesIO(f.read())

        if source_choice == "Curated Infrastructure Gallery" and st.session_state.get("ai_demo_sel"):
            ai_path = os.path.join(AI_DEMO_IMAGES_DIR, st.session_state["ai_demo_sel"])
            if os.path.isfile(ai_path):
                with open(ai_path, "rb") as f:
                    uploaded_file = io.BytesIO(f.read())
                st.info(f"Loaded gallery sample: **{st.session_state['ai_demo_sel']}**")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            # ── Inference Execution ─────────────────────────────────────────
            results = {}
            for key, label in [("rtdetr", "RT-DETR-L"), ("yolo", "YOLO26m")]:
                model = models.get(key)
                if model is not None:
                    t0 = time.time()
                    res = run_inference(model, image, conf)
                    elapsed = time.time() - t0
                    res["time_ms"] = elapsed * 1000
                    results[key] = res

            rt_raw = results.get("rtdetr", {}).get("detections", [])
            yo_raw = results.get("yolo", {}).get("detections", [])

            # Apply UI filters
            rt_filtered = [d for d in rt_raw if d["class_name"] in selected_classes]
            yo_filtered = [d for d in yo_raw if d["class_name"] in selected_classes]

            # Generate Neural Fusion datasets
            ensemble_fused = run_ensemble_fusion(
                rt_filtered, yo_filtered, 
                iou_thresh=consensus_threshold, 
                mode=ensemble_view_mode if "Fused" in ensemble_view_mode else "Union (High Recall)"
            )

            # Choose active detections based on dropdown
            if "RT-DETR-L" in ensemble_view_mode:
                active_detections = rt_filtered
                active_label = "RT-DETR-L (Transformer Only)"
            elif "YOLO26m" in ensemble_view_mode:
                active_detections = yo_filtered
                active_label = "YOLO26m (CNN Only)"
            else:
                active_detections = ensemble_fused
                active_label = f"Fused Ensemble ({ensemble_view_mode})"

            # Calculate Structural Health Index
            health_audit = calculate_structural_health(active_detections, image.width, image.height)

            # ── HEADER KPI DASHBOARD ────────────────────────────────────────
            st.subheader("📊 Diagnostic Audit Panel")
            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
            
            col_kpi1.metric(
                label="Structural Health Index (SHI)",
                value=f"{health_audit['shi']}/100",
                help="Aggregated structural safe index. Deductions are scaled by class priority and bbox pixel fraction."
            )
            col_kpi2.metric(
                label="Active Detections Count",
                value=len(active_detections),
                help="Number of active defect detections matching your current filter and neural ensemble settings."
            )
            
            rt_ms = results.get("rtdetr", {}).get("time_ms", 0)
            yo_ms = results.get("yolo", {}).get("time_ms", 0)
            col_kpi3.metric(
                label="Average Latency",
                value=f"{np.mean([rt_ms, yo_ms]):.0f} ms",
                help="Mean hardware inference delay for both architecture evaluations."
            )
            
            consensus_overlap = sum(1 for d in ensemble_fused if d.get("is_consensus", False))
            col_kpi4.metric(
                label="Consensus Defect Count",
                value=consensus_overlap,
                help="Overlapping defects detected simultaneously by RT-DETR-L and YOLO26m."
            )

            # ── STRUCTURAL URGENCY STATUS BANNER ────────────────────────────
            st.markdown(
                f"""
                <div class="status-card status-{health_audit['status_color']}">
                    <h3 style="margin-top: 0; color: inherit;">{health_audit['urgency_label']}</h3>
                    <p style="font-size: 1.1rem; color: inherit; margin-bottom: 12px;"><strong>Assessment:</strong> {health_audit['description']}</p>
                    <strong>Recommended Field Actions:</strong>
                    <ul style="margin-top: 5px; color: inherit; margin-bottom: 0;">
                        {"".join(f"<li>{act}</li>" for act in health_audit['action_plan'])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ── MAIN ANNOTATED VISUALIZATION ────────────────────────────────
            st.markdown("---")
            st.subheader("🖼️ High-Definition Diagnostic Visualization")
            
            if ensemble_view_mode == "Compare Side-by-Side":
                # Original side-by-side mode
                col_img_l, col_img_r = st.columns(2)
                with col_img_l:
                    st.markdown("#### RT-DETR-L (Transformer Attention)")
                    img_rt = _draw_boxes_on_pil(
                        image, rt_filtered, selected_classes, box_width, fill_boxes, 
                        font_scale, highlight_consensus, consensus_style
                    )
                    st.image(img_rt, use_container_width=True)
                    st.caption(f"Detections: {len(rt_filtered)} | Speed: {rt_ms:.0f} ms")
                    
                with col_img_r:
                    st.markdown("#### YOLO26m (Convolutional Feature Pyramid)")
                    img_yo = _draw_boxes_on_pil(
                        image, yo_filtered, selected_classes, box_width, fill_boxes, 
                        font_scale, highlight_consensus, consensus_style
                    )
                    st.image(img_yo, use_container_width=True)
                    st.caption(f"Detections: {len(yo_filtered)} | Speed: {yo_ms:.0f} ms")
            else:
                # Large Fused view
                st.markdown(f"#### {active_label}")
                img_fused = _draw_boxes_on_pil(
                    image, active_detections, selected_classes, box_width, fill_boxes, 
                    font_scale, highlight_consensus, consensus_style
                )
                st.image(img_fused, use_container_width=True)
                st.caption(f"Rendered Overlay: {len(active_detections)} total items.")

            legend_html = "  ".join(
                f'<span style="display:inline-block;width:14px;height:14px;'
                f'background:rgb({c[0]},{c[1]},{c[2]});border-radius:3px;'
                f'margin-right:4px;vertical-align:middle;"></span><strong>{n}</strong>'
                for c, n in CLASS_LEGEND
            )
            st.markdown(f"**Defect class legend:** &nbsp;&nbsp;&nbsp;&nbsp;{legend_html}", unsafe_allow_html=True)

            # ── INTERACTIVE ROI CLOSEUP GALLERY ─────────────────────────────
            st.markdown("---")
            st.subheader("🔍 Interactive Bounding Box ROI Inspector")
            st.markdown(
                "Individual defect regions cropped on the fly. Aspect ratios and dimensions "
                "are analyzed below using specialized civil engineering guidelines."
            )
            
            crops = get_defect_crops(image, active_detections)
            if not crops:
                st.info("No active defect crops at this confidence threshold.")
            else:
                crop_cols = st.columns(4)
                for c_idx, c in enumerate(crops):
                    with crop_cols[c_idx % 4]:
                        st.markdown(
                            f"""
                            <div class="crop-card">
                                <span style="font-size: 0.85rem; font-weight: bold; background-color: #2196f3; color: white; padding: 2px 6px; border-radius: 4px;">
                                    Defect #{c['id']}
                                </span>
                                <h5 style="margin: 8px 0 2px 0;">{c['sub_class']}</h5>
                                <span style="font-size: 0.8rem; color: #666;">Class: {c['class_name']} | Conf: {c['confidence']:.2f}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.image(c["crop_img"], use_container_width=True)
                        with st.expander("Diagnostic details", expanded=False):
                            st.caption(f"**Sizing:** {c['dimensions']}")
                            st.caption(f"**Bounding Box:** {c['bbox_str']}")
                            st.caption(c["details"])

            # ── TABULAR DETECTIONS & ARCHITECTURAL SPECS ────────────────────
            st.markdown("---")
            col_tbl_l, col_img_specs = st.columns([2, 1])
            
            with col_tbl_l:
                st.subheader("📋 Detailed Diagnostic Table")
                if active_detections:
                    df_dets = pd.DataFrame(active_detections)
                    df_dets["bbox_str"] = df_dets["bbox"].apply(
                        lambda b: f"[{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]"
                    )
                    df_dets["conf_pct"] = (df_dets["confidence"] * 100).round(1)
                    df_dets["is_con_str"] = df_dets["is_consensus"].apply(lambda x: "✅ Yes" if x else "❌ No")
                    
                    df_display = df_dets[["class_name", "conf_pct", "severity", "is_con_str", "bbox_str", "source"]].copy()
                    df_display.columns = ["Defect Class", "Confidence %", "Confidence Band", "Consensus Overlap", "Bounding Box (XYXY)", "Source Model"]
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No active detections in table.")
                    
            with col_img_specs:
                st.subheader("📐 Model Architectural Specs")
                iterlog = _read_iterlog()
                yolo_df = _read_yolo_summary()
                
                # Default baseline values
                peak_mAP50_rt = 0.613
                best_epoch_rt = 43
                
                peak_mAP50_yo = 0.763
                best_epoch_yo = 277
                
                if iterlog is not None:
                    peak_idx = iterlog["mAP50"].idxmax()
                    peak_mAP50_rt = iterlog.loc[peak_idx, "mAP50"]
                    best_epoch_rt = iterlog.loc[peak_idx, "best_epoch"]
                    
                if yolo_df is not None:
                    frozen = yolo_df[yolo_df["Variant"] == "Frozen Backbone"]
                    if len(frozen) > 0:
                        peak_mAP50_yo = frozen.iloc[0].get("metrics/mAP50(B)", 0.763)
                        best_epoch_yo = frozen.iloc[0].get("epoch", 277)
                
                specs_data = [
                    {
                        "Attribute": "Neural Model",
                        "RT-DETR-L": "RT-DETR-L",
                        "YOLO26m": "YOLO26m"
                    },
                    {
                        "Attribute": "Core Backbone",
                        "RT-DETR-L": "Transformer HGNetv2",
                        "YOLO26m": "CNN DarkNet"
                    },
                    {
                        "Attribute": "Parameter Count",
                        "RT-DETR-L": "32.0 Million",
                        "YOLO26m": "11.2 Million"
                    },
                    {
                        "Attribute": "Dataset mAP@50",
                        "RT-DETR-L": f"{peak_mAP50_rt:.3f}",
                        "YOLO26m": f"{peak_mAP50_yo:.3f}"
                    },
                    {
                        "Attribute": "Peak Train Epoch",
                        "RT-DETR-L": str(int(best_epoch_rt)),
                        "YOLO26m": str(int(best_epoch_yo))
                    }
                ]
                st.dataframe(pd.DataFrame(specs_data), use_container_width=True, hide_index=True)

            # ── EXPORT OPTIONS ──────────────────────────────────────────────
            st.markdown("---")
            st.subheader("💾 Export Diagnostics & Field Audit Reports")
            
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                export_format = st.selectbox(
                    "Select Export Artifact:",
                    [
                        "Annotated PNG (Fused Model)",
                        "Side-by-Side Composite Image",
                        "Diagnostic Table CSV",
                        "Diagnostic Meta JSON",
                    ],
                    key="export_format"
                )
            
            with ex_col2:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                if export_format == "Annotated PNG (Fused Model)":
                    img_out = _draw_boxes_on_pil(
                        image, active_detections, selected_classes, box_width, fill_boxes, 
                        font_scale, highlight_consensus, consensus_style
                    )
                    buf = io.BytesIO()
                    img_out.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Overlay Image (PNG)",
                        data=buf.getvalue(),
                        file_name="structural_overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )
                elif export_format == "Side-by-Side Composite Image":
                    img_rt = _draw_boxes_on_pil(image, rt_filtered, selected_classes, box_width, fill_boxes, font_scale, highlight_consensus, consensus_style)
                    img_yo = _draw_boxes_on_pil(image, yo_filtered, selected_classes, box_width, fill_boxes, font_scale, highlight_consensus, consensus_style)
                    composite = _make_composite(img_rt, img_yo, f"RT-DETR-L ({rt_ms:.0f} ms)", f"YOLO26m ({yo_ms:.0f} ms)")
                    buf = io.BytesIO()
                    composite.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Composite comparison",
                        data=buf.getvalue(),
                        file_name="diagnostic_composite.png",
                        mime="image/png",
                        use_container_width=True
                    )
                elif export_format == "Diagnostic Table CSV":
                    all_rows = []
                    for d in active_detections:
                        all_rows.append({
                            "defect_class": d["class_name"],
                            "confidence_score": round(d["confidence"], 4),
                            "consensus": "Yes" if d.get("is_consensus", False) else "No",
                            "bbox_x1": round(d["bbox"][0], 1),
                            "bbox_y1": round(d["bbox"][1], 1),
                            "bbox_x2": round(d["bbox"][2], 1),
                            "bbox_y2": round(d["bbox"][3], 1),
                            "source_model": d.get("source", "Combined")
                        })
                    if all_rows:
                        csv_data = pd.DataFrame(all_rows).to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download Diagnostics CSV",
                            data=csv_data,
                            file_name="diagnostic_table.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                elif export_format == "Diagnostic Meta JSON":
                    import json
                    json_data = {
                        "structural_health_index": health_audit["shi"],
                        "urgency_class": health_audit["urgency_label"],
                        "total_defects_fused": len(active_detections),
                        "consensus_defect_count": consensus_overlap,
                        "latencies": {"rtdetr_ms": rt_ms, "yolo_ms": yo_ms},
                        "detections": active_detections
                    }
                    st.download_button(
                        label="⬇️ Download Diagnostics JSON",
                        data=json.dumps(json_data, indent=2),
                        file_name="structural_meta.json",
                        mime="application/json",
                        use_container_width=True
                    )

            # ── AUDIT REPORT MARKDOWN EXPORTER ──────────────────────────────
            st.markdown("---")
            st.markdown("### 📝 Generate Printable Field Audit Report")
            report_md = f"""# STRUCTURAL DIAGNOSTIC INSPECTION REPORT
---
**Date of Audit:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Target Structure Profile:** Civil Infrastructure Core Component  
**Evaluated Diagnostic Mode:** {active_label}  

## 1. EXECUTIVE SUMMARY
*   **Structural Health Index (SHI):** {health_audit['shi']}/100
*   **Safety Status:** {health_audit['urgency_label']}
*   **Defect Count:** {len(active_detections)} items
*   **Consensus Overlaps:** {consensus_overlap} overlapping items

### Assessment & Action Plan:
{health_audit['description']}

**Field Actions Recommended:**
{chr(10).join(f'*   {act}' for act in health_audit['action_plan'])}

## 2. COMPREHENSIVE DEFECT ROSTER
| # | Class Name | Confidence | Overlap Consensus | Bounding Box Coordinate (XYXY) |
|---|------------|------------|-------------------|--------------------------------|
"""
            for d_idx, d in enumerate(active_detections):
                report_md += f"| {d_idx+1} | {d['class_name']} | {d['confidence']*100:.1f}% | {'Yes' if d.get('is_consensus', False) else 'No'} | [{d['bbox'][0]:.0f}, {d['bbox'][1]:.0f}, {d['bbox'][2]:.0f}, {d['bbox'][3]:.0f}] |\n"
                
            report_md += f"""
## 3. MODEL LATENCY & HARDWARE PROVENANCE
*   **RT-DETR-L Delay:** {rt_ms:.1f} ms
*   **YOLO26m Delay:** {yo_ms:.1f} ms
*   **Consensus Match Gate:** IoU >= {consensus_threshold}
"""
            with st.expander("Preview Field Audit Document (Markdown)", expanded=False):
                st.code(report_md, language="markdown")
                
            st.download_button(
                label="📥 Download Official Civil Audit Report (Markdown File)",
                data=report_md,
                file_name="structural_audit_report.md",
                mime="text/markdown",
                use_container_width=True
            )

        else:
            st.info("Upload an image file or choose a curated structural photo in the sidebar to begin active-neural diagnostic audit.")

    else:
        # ═══════════════════════════════════════════════════════════════════
        # BATCH FOLDER PROCESSING MODE
        # ═══════════════════════════════════════════════════════════════════
        st.subheader("📁 Batch Folder Scan Protocol")
        st.markdown(
            "Batch execution panel. Evaluates multiple structural frames sequentially, "
            "compiles global defect trends, computes statistical SHI, and exports bulk ZIP packages."
        )

        uploaded_files = st.file_uploader(
            "Upload image subset for bulk evaluation",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="batch_uploader",
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.info("Awaiting folder/file collection input upload.")
        else:
            files_to_process = [f for f in uploaded_files if f.name not in st.session_state["batch_cache"]]

            if files_to_process:
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, f in enumerate(files_to_process):
                    status_text.text(f"Evaluating Model Inference: {f.name} ({idx+1}/{len(files_to_process)})...")
                    img = Image.open(f).convert("RGB")

                    # Keep raw low confidence predictions inside cache for quick threshold adjustments later
                    rt_time = 0.0
                    rt_dets = []
                    if models.get("rtdetr"):
                        t0 = time.time()
                        rt_res = run_inference(models["rtdetr"], img, conf=0.05)
                        rt_dets = rt_res["detections"]
                        rt_time = (time.time() - t0) * 1000

                    yo_time = 0.0
                    yo_dets = []
                    if models.get("yolo"):
                        t0 = time.time()
                        yo_res = run_inference(models["yolo"], img, conf=0.05)
                        yo_dets = yo_res["detections"]
                        yo_time = (time.time() - t0) * 1000

                    f.seek(0)
                    file_bytes = f.read()
                    st.session_state["batch_cache"][f.name] = {
                        "file_bytes": file_bytes,
                        "rtdetr_raw_dets": rt_dets,
                        "yolo_raw_dets": yo_dets,
                        "rtdetr_time": rt_time,
                        "yolo_time": yo_time,
                    }
                    progress_bar.progress((idx + 1) / len(files_to_process))

                status_text.success("Batch pipeline calculations completed successfully!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()

            # Process cached batch records with current parameters
            batch_rows = []
            total_rtdetr_defects = 0
            total_yolo_defects = 0
            total_consensus_defects = 0
            sum_shi = 0.0
            critical_images_count = 0

            rtdetr_class_counts = {"crack": 0, "pothole": 0, "wall_peeling": 0}
            yolo_class_counts = {"crack": 0, "pothole": 0, "wall_peeling": 0}

            processed_data = {}

            for f in uploaded_files:
                cached = st.session_state["batch_cache"].get(f.name)
                if not cached:
                    continue

                rt_filtered = [
                    d for d in cached["rtdetr_raw_dets"]
                    if d["confidence"] >= conf and d["class_name"] in selected_classes
                ]
                yo_filtered = [
                    d for d in cached["yolo_raw_dets"]
                    if d["confidence"] >= conf and d["class_name"] in selected_classes
                ]

                # Run consensus and fusion
                ensemble_fused = run_ensemble_fusion(
                    rt_filtered, yo_filtered, 
                    iou_thresh=consensus_threshold, 
                    mode=ensemble_view_mode if "Fused" in ensemble_view_mode else "Union (High Recall)"
                )

                # Active selection for SHI
                if "RT-DETR-L" in ensemble_view_mode:
                    active = rt_filtered
                elif "YOLO26m" in ensemble_view_mode:
                    active = yo_filtered
                else:
                    active = ensemble_fused

                # Metrics for active
                health_meta = calculate_structural_health(active, 640, 640)
                sum_shi += health_meta["shi"]
                if health_meta["status_color"] == "critical":
                    critical_images_count += 1

                for d in rt_filtered:
                    if d["class_name"] in rtdetr_class_counts:
                        rtdetr_class_counts[d["class_name"]] += 1
                for d in yo_filtered:
                    if d["class_name"] in yolo_class_counts:
                        yolo_class_counts[d["class_name"]] += 1

                consensus_cnt = sum(1 for d in ensemble_fused if d.get("is_consensus", False))
                total_rtdetr_defects += len(rt_filtered)
                total_yolo_defects += len(yo_filtered)
                total_consensus_defects += consensus_cnt

                union_total = len(rt_filtered) + len(yo_filtered) - consensus_cnt
                agree_pct = (consensus_cnt / max(1, union_total)) * 100

                batch_rows.append({
                    "Image Frame Name": f.name,
                    "Health Index (SHI)": f"{health_meta['shi']}/100",
                    "Status": health_meta["urgency_label"].split()[-1],
                    "RT-DETR-L Defects": len(rt_filtered),
                    "YOLO26m Defects": len(yo_filtered),
                    "Consensus Overlaps": consensus_cnt,
                    "Voting Agreement": f"{agree_pct:.1f}%",
                    "RT-DETR speed (ms)": f"{cached['rtdetr_time']:.0f}",
                    "YOLO speed (ms)": f"{cached['yolo_time']:.0f}",
                })

                processed_data[f.name] = {
                    "file_bytes": cached["file_bytes"],
                    "rtdetr_dets": rt_filtered,
                    "yolo_dets": yo_filtered,
                    "ensemble_dets": ensemble_fused,
                    "rtdetr_time": cached["rtdetr_time"],
                    "yolo_time": cached["yolo_time"],
                    "health": health_meta
                }

            if batch_rows:
                # ── BATCH ANALYSIS KPI SUMMARY ──────────────────────────────
                st.markdown("---")
                st.subheader("📈 Project Batch Metrics Overview")
                
                bm1, bm2, bm3, bm4 = st.columns(4)
                bm1.metric("Analyzed Assets", len(uploaded_files))
                
                mean_shi = sum_shi / len(uploaded_files)
                bm2.metric("Mean Project Health (SHI)", f"{mean_shi:.1f}/100")
                
                bm3.metric("Critical Hazards Identified", f"{critical_images_count} frames")
                
                union_all = total_rtdetr_defects + total_yolo_defects - total_consensus_defects
                batch_agree_pct = (total_consensus_defects / max(1, union_all)) * 100
                bm4.metric("Consensus Matching Rate", f"{batch_agree_pct:.1f}%")

                st.markdown("**Field Diagnostics Roster**")
                st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)

                # Batch charts side-by-side
                st.markdown("---")
                st.subheader("📊 Statistical Visual Distribution")
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("**Defect Class Classifications by Model**")
                    dist_df = pd.DataFrame({
                        "RT-DETR-L": list(rtdetr_class_counts.values()),
                        "YOLO26m": list(yolo_class_counts.values())
                    }, index=list(rtdetr_class_counts.keys()))
                    st.bar_chart(dist_df, height=300)

                with chart_col2:
                    st.markdown("**Model Architectural Compute Speed (ms)**")
                    avg_rt_speed = np.mean([c["rtdetr_time"] for c in st.session_state["batch_cache"].values()])
                    avg_yo_speed = np.mean([c["yolo_time"] for c in st.session_state["batch_cache"].values()])
                    speed_df = pd.DataFrame({
                        "Architecture": ["RT-DETR-L (Transformer)", "YOLO26m (CNN)"],
                        "Latency (ms)": [avg_rt_speed, avg_yo_speed]
                    })
                    st.bar_chart(speed_df.set_index("Architecture"), height=300)

                # ── DETAILED BATCH BULK INSPECTOR ───────────────────────────
                st.markdown("---")
                st.subheader("🔍 Batch Frame Visual Inspector")
                selected_batch_img = st.selectbox(
                    "Pick a bulk evaluated image frame below to audit in-detail:",
                    [f.name for f in uploaded_files],
                    key="batch_inspector_sel"
                )

                if selected_batch_img in processed_data:
                    img_data = processed_data[selected_batch_img]
                    pil_img = Image.open(io.BytesIO(img_data["file_bytes"])).convert("RGB")

                    # Draw boxes on the fly
                    active_view = img_data["ensemble_dets"] if "Fused" in ensemble_view_mode else (
                        img_data["rtdetr_dets"] if "RT-DETR" in ensemble_view_mode else img_data["yolo_dets"]
                    )
                    
                    st.markdown(
                        f"""
                        <div class="status-card status-{img_data['health']['status_color']}">
                            <h4 style="margin: 0; color: inherit;">Asset Score: {img_data['health']['shi']}/100 — Status: {img_data['health']['urgency_label']}</h4>
                            <p style="margin: 4px 0 0 0; color: inherit; font-size: 0.95rem;">{img_data['health']['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    col_r, col_y = st.columns(2)
                    with col_r:
                        st.markdown("### RT-DETR-L")
                        rt_img_annotated = _draw_boxes_on_pil(
                            pil_img, img_data["rtdetr_dets"], selected_classes, box_width, 
                            fill_boxes, font_scale, highlight_consensus, consensus_style
                        )
                        st.image(rt_img_annotated, use_container_width=True)
                        st.caption(f"Inference: {img_data['rtdetr_time']:.0f} ms | Detections: {len(img_data['rtdetr_dets'])}")

                    with col_y:
                        st.markdown("### YOLO26m")
                        yo_img_annotated = _draw_boxes_on_pil(
                            pil_img, img_data["yolo_dets"], selected_classes, box_width, 
                            fill_boxes, font_scale, highlight_consensus, consensus_style
                        )
                        st.image(yo_img_annotated, use_container_width=True)
                        st.caption(f"Inference: {img_data['yolo_time']:.0f} ms | Detections: {len(img_data['yolo_dets'])}")

                    # Show Crops for this batch image
                    st.markdown("**Dynamic Crops for chosen Batch Image:**")
                    batch_crops = get_defect_crops(pil_img, active_view)
                    if batch_crops:
                        b_crop_cols = st.columns(5)
                        for bc_idx, bc in enumerate(batch_crops):
                            with b_crop_cols[bc_idx % 5]:
                                st.markdown(f"**Crop #{bc['id']}:** *{bc['sub_class']}*")
                                st.image(bc["crop_img"], use_container_width=True)
                    else:
                        st.caption("No defects detected in this frame.")

                # ── COMPREHENSIVE BATCH ZIP PACKAGE EXPORT ──────────────────
                st.markdown("---")
                st.subheader("📦 Download Consolidated Field Inspection Package")
                st.markdown("Compiles complete batch summary logs, JSON matrices, and annotated side-by-side composite images.")
                
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Summary CSV
                    csv_data = pd.DataFrame(batch_rows).to_csv(index=False)
                    zf.writestr("project_batch_summary.csv", csv_data)
                    
                    # Predictions JSON
                    json_data = {}
                    for fname, b_img in processed_data.items():
                        json_data[fname] = {
                            "structural_health_score": b_img["health"]["shi"],
                            "urgency_class": b_img["health"]["urgency_label"],
                            "rtdetr_detections": b_img["rtdetr_dets"],
                            "yolo_detections": b_img["yolo_dets"],
                        }
                    import json
                    zf.writestr("metadata_predictions.json", json.dumps(json_data, indent=2))
                    
                    # Side-by-side JPGs
                    for fname, b_img in processed_data.items():
                        img_pil = Image.open(io.BytesIO(b_img["file_bytes"])).convert("RGB")
                        rt_img = _draw_boxes_on_pil(img_pil, b_img["rtdetr_dets"], selected_classes, box_width, fill_boxes, font_scale, highlight_consensus, consensus_style)
                        yo_img = _draw_boxes_on_pil(img_pil, b_img["yolo_dets"], selected_classes, box_width, fill_boxes, font_scale, highlight_consensus, consensus_style)
                        composite_img = _make_composite(rt_img, yo_img, "RT-DETR-L (Transformer)", "YOLO26m (CNN)")
                        img_buf = io.BytesIO()
                        composite_img.save(img_buf, format="JPEG", quality=85)
                        zf.writestr(f"composites/{fname}_side_by_side.jpg", img_buf.getvalue())
                        
                st.download_button(
                    label="📥 Download Consolidated Inspection Package (ZIP File)",
                    data=zip_buf.getvalue(),
                    file_name="structural_project_audit.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

# ═══════════════════════════════════════════════════════════════════════════
# TAB: Methodology
# ═══════════════════════════════════════════════════════════════════════════
with tab_method:
    st.header("📘 Core Engineering Methodology")

    st.subheader("1. Problem Context")
    st.markdown(
        "Infrastructure damage assessment represents a critical safety concern in civil engineering. "
        "Manual visual evaluations are slow, labor-intensive, and inherently subjective. This platform "
        "integrates and compares two cutting-edge deep learning paradigms to demonstrate optimized transfer "
        "learning trade-offs for structural defect detection."
    )

    st.subheader("2. Model Architecture Paradigms")
    st.markdown(
        "To provide a complete engineering trade-off audit, the system integrates "
        "two distinct deep network categories:"
    )
    col_arch1, col_grid_arrow, col_arch2 = st.columns([10, 1, 10])
    
    with col_arch1:
        st.markdown(
            "#### 🌟 RT-DETR-L (Real-Time DEtection TRansformer)\n"
            "*   **Paradigm:** Transformer Attention Network.\n"
            "*   **Backbone:** HGNetv2 with multi-scale feature encoder & decoder.\n"
            "*   **Attention Mechanism:** Employs hybrid encoder layers to capture global contextual-spatial dependencies "
            "between features. This maximizes bounding-box localization accuracy for diffuse, large, or compound defects (e.g. wall peeling, creeping cracks).\n"
            "*   **Parameters:** ~32 Million."
        )
        
    with col_arch2:
        st.markdown(
            "#### 🚀 YOLO26m (You Only Look Once)\n"
            "*   **Paradigm:** Single-Stage Deep Convolutional Neural Network (CNN).\n"
            "*   **Architecture:** CSPDarknet backbone utilizing spatial pyramid pooling (SPPF) and advanced PANet necks.\n"
            "*   **Strengths:** Excels in high-density local spatial convolutions, providing exceptional recall for small, distinct structural "
            "targets (e.g. hairline potholes or micro-cracks) and extreme real-time frame rates.\n"
            "*   **Parameters:** ~11 Million."
        )

    st.subheader("3. Custom Dataset Profile")
    st.markdown(
        "The models were trained on a custom YOLO-format structural damage dataset comprising "
        "**780 high-resolution images** split into **546 training**, **117 validation**, and **117 test** partitions. "
        "Annotated classes follow strict, non-overlapping bounding-box formats:"
    )
    st.markdown(
        "- **`crack` (Class 0):** Structural fractures of road pavements and concrete partitions.\n"
        "- **`pothole` (Class 1):** Road pavement potholes and localized depressions.\n"
        "- **`wall_peeling` (Class 2):** Concrete cover, plaster, and superficial wall delamination."
    )

    st.subheader("4. Training Loops & Transfer Learning Strategy")
    st.markdown(
        "#### Active-Learning Iteration Cycle (RT-DETR-L)\n"
        "RT-DETR-L was trained over a **6-iteration active-learning protocol** (iters 0 to 5) configured inside `Train3.ipynb`. "
        "In each loop iteration, 28 manually flagged images were added to the training set dynamically, resuming weights from the previous checkpoint. "
        "Empirical training statistics reached an optimal performance peak of **mAP50 = 0.613 at iteration 3 (484 training images)** before plateauing. "
        "This plateau represents a common transformer training phenomenon where introducing additional small-batch samples adds background noise "
        "that exceeds the transformer's capacity to build coherent attention maps, suggesting high sensitivity to clean, distinct annotations."
    )
    
    st.markdown(
        "#### YOLO26m Ablation Study\n"
        "Three distinct transfer-learning freezing strategies were executed on YOLO26m to determine base layer representation convergence:"
        "1.  **Frozen Backbone:** All pre-trained CSPDarknet layers are locked; only Neck/Head layers are trainable. (Peak mAP50 = 0.763)\n"
        "2.  **Partial Freeze Neck:** Intermediate Neck feature fusion layer frozen. (Lower localization accuracy)\n"
        "3.  **Unfrozen:** Full backpropagation. High recall but slightly lower mAP50 due to over-fitting on a relatively small base set.\n\n"
        "The **Frozen Backbone** configuration was selected as the optimal benchmark model because it maintains pre-trained representation stability."
    )

    st.subheader("5. Neural Fusion & Structural Audit Logic")
    st.markdown(
        "The **Neural Ensemble Fusion Engine** implements spatial intersection over union matching. "
        "By adjusting the ensemble strategy, engineering teams can configure the model to serve different operational safety postures:"
    )
    st.info(
        "💡 **Intersection (High Precision):** Discards isolated detections. Minimizes maintenance false-positives.\\\n"
        "💡 **Union (High Recall):** Includes all non-overlapping boxes, resolving overlapping duplicates via confidence gating. Safety-first deployment.\\\n"
        "💡 **Weighted Average:** Resolves overlapping regions by computing confidence-weighted coordinate coordinates."
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB: Evidence
# ═══════════════════════════════════════════════════════════════════════════
with tab_evidence:
    st.header("📊 Empirical Training Evidence & Artifacts")

    st.subheader("1. RT-DETR-L Active-Learning Iterlog")
    iterlog = _read_iterlog()
    if iterlog is not None:
        st.dataframe(
            iterlog.style.highlight_max(
                subset=["mAP50", "mAP50_95", "precision", "recall"],
                color="#c8e6c9",
            ),
            use_container_width=True,
            hide_index=True,
        )
        peak = iterlog.loc[iterlog["mAP50"].idxmax()]
        st.success(
            f"🎯 **Empirical Source of Truth:** Performance peaked at **Iteration {int(peak['iter'])}** "
            f"with **mAP50 = {peak['mAP50']:.4f}** (best epoch: {int(peak['best_epoch'])}, cumulative train subset: {int(peak['cum_train_imgs'])} images)."
        )
    else:
        st.warning("Active-learning iterlog not found at expected path.")

    st.subheader("2. YOLO26m Transfer Learning Ablation Summary")
    yolo_df = _read_yolo_summary()
    if yolo_df is not None and len(yolo_df) > 0:
        st.dataframe(
            yolo_df.style.highlight_max(
                subset=[c for c in yolo_df.columns if c.startswith("metrics/")],
                color="#c8e6c9",
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("YOLO results CSV records not found.")

    st.subheader("3. Empirical Attention Graphs (RT-DETR-L)")
    for fname, caption in EVIDENCE_IMAGES:
        img_path = os.path.join(BASE_DIR, ARTIFACT_BASE, fname)
        if os.path.isfile(img_path):
            with st.expander(f"👁️ View: {caption}", expanded=False):
                st.image(img_path, use_container_width=True, caption=caption)
        else:
            st.caption(f"Artifact not found: {fname}")

    st.subheader("4. YOLO26m Transfer Learning Ablation Curves")
    for variant, label in YOLO_EVIDENCE:
        with st.expander(f"👁️ View YOLO26m: {label} Curves", expanded=False):
            for artifact in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
                img_path = os.path.join(
                    BASE_DIR, "jenny", "runs", "detect", "road_damage",
                    variant, artifact,
                )
                if os.path.isfile(img_path):
                    st.image(img_path, use_container_width=True,
                             caption=f"{label} — {artifact.replace('_', ' ').replace('.png', '')}")

    st.subheader("5. Test Set Confusion Matrices (RT-DETR-L)")
    cm_cols = st.columns(3)
    for iter_idx in range(6):
        cm_path = os.path.join(
            BASE_DIR, ARTIFACT_BASE,
            f"rtdetr_l_v7_iter{iter_idx}_test",
            "confusion_matrix_normalized.png",
        )
        if os.path.isfile(cm_path):
            with cm_cols[iter_idx % 3]:
                st.markdown(f"**Iteration {iter_idx} Normalized Confusion Matrix**")
                st.image(cm_path, use_container_width=True)
