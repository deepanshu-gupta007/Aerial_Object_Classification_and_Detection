import os
import io
import warnings
import numpy as np
import streamlit as st
import cv2
import json
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Configuration
MODELS_DIR = Path("./models")
ASSETS_DIR = Path("./assets")
YOLO_RUN_DIR = Path("./runs/detect/models/yolov8_runs/birds_drones")

IMG_SIZE     = (224, 224)
CLASSES      = ["Bird", "Drone"]
CLASS_COLORS = {"Bird": "#4CAF50", "Drone": "#F44336"}
CLASS_EMOJIS = {"Bird": "🦅",      "Drone": "🚁"}
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_COLORS = {
    "Custom_CNN":     "#b57aff",
    "ResNet50":       "#38b2ff",
    "MobileNetV2":    "#4af09a",
    "EfficientNetB0": "#ffcc44",
    "YOLOv8s":        "#ff6b6b",
}
MODEL_PARAMS = {
    "Custom_CNN":     "673,601",
    "ResNet50":       "541,569",
    "MobileNetV2":    "344,961",
    "EfficientNetB0": "344,961",
}

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-LOAD RESULT FILES ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _load_yolo_final_metrics() -> dict | None:
    """
    Parse the last row of YOLO's results.csv to get final-epoch metrics.
    Returns a dict matching the BENCHMARK column schema, or None.
    """
    results_csv = YOLO_RUN_DIR / "results.csv"
    training_time_csv = YOLO_RUN_DIR / "training_time.csv"
    if not results_csv.exists() or not training_time_csv.exists():
        return None
    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()          # strip whitespace
        last = df.iloc[-1]

        # Column names differ slightly across Ultralytics versions
        def _get(candidates, fallback=None):
            for c in candidates:
                if c in last.index:
                    return float(last[c])
            return fallback

        precision = _get(["metrics/precision(B)", "metrics/precision"])
        recall    = _get(["metrics/recall(B)",    "metrics/recall"])
        map50     = _get(["metrics/mAP50(B)",     "metrics/mAP_0.5"])
        f1        = (2 * precision * recall / (precision + recall + 1e-9)
                     if precision is not None and recall is not None else None)

        tt_df = pd.read_csv(training_time_csv)
        tt_df.columns = tt_df.columns.str.strip()
        train_time_sec = float(tt_df.columns[1])
        train_time_min = round(train_time_sec / 60, 2)

        return {
            "Model":       "YOLOv8s",
            "Accuracy":    map50 * 100 if map50 is not None else None,
            "Precision":   precision,
            "Recall":      recall,
            "F1":          f1,
            "Loss":        None,
            "Z-Score":     None,
            "Time (min)":  train_time_min,
        }
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame:
    """
    Load model comparison metrics from the CSV saved during training.
    Falls back to an empty DataFrame if the file is missing.

    Expected file: assets/model_comparison_results.csv
    Expected columns: model, accuracy_pct, precision, recall, f1, loss,
                      z_score, training_time_min
    """
    csv_path = ASSETS_DIR / "model_comparison_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # ── Normalise column names to what the app expects ──────────────────
        rename_map = {
            "model":            "Model",
            "accuracy_pct":     "Accuracy",   # stored as 0-100 float
            "precision":        "Precision",
            "recall":           "Recall",
            "f1":               "F1",
            "loss":             "Loss",
            "z_score":          "Z-Score",
            "training_time_min":"Time (min)",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # ── Append YOLO row from its own results file ────────────────────────
        yolo_row = _load_yolo_final_metrics()
        if yolo_row:
            df = pd.concat([df, pd.DataFrame([yolo_row])], ignore_index=True)

        # ── Sort best → worst ────────────────────────────────────────────────
        df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        return df

    # ── Graceful fallback: return empty skeleton ─────────────────────────────
    st.warning("⚠️  `assets/model_comparison_results.csv` not found. "
               "Run the training notebook first.")
    return pd.DataFrame(columns=["Model","Accuracy","Precision","Recall",
                                  "F1","Loss","Z-Score","Time (min)"])


@st.cache_data(show_spinner=False)
def load_yolo_history() -> dict:
    """
    Load YOLO epoch-by-epoch metrics from results.csv generated by Ultralytics.

    File: runs/detect/models/yolov8_runs/birds_drones/results.csv
    Returns dict with keys: epochs, mAP50, mAP5095, box_loss, cls_loss, dfl_loss
    """
    results_csv = YOLO_RUN_DIR / "results.csv"
    if not results_csv.exists():
        st.warning("⚠️  YOLO `results.csv` not found. Run YOLO training first.")
        return {}

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    def _col(candidates):
        for c in candidates:
            if c in df.columns:
                return df[c].tolist()
        return []

    return {
        "epochs":   list(range(1, len(df) + 1)),
        "mAP50":    _col(["metrics/mAP50(B)",     "metrics/mAP_0.5"]),
        "mAP5095":  _col(["metrics/mAP50-95(B)",  "metrics/mAP_0.5:0.95"]),
        "box_loss": _col(["train/box_loss",        "train/box_om"]),
        "cls_loss": _col(["train/cls_loss",        "train/cls_om"]),
        "dfl_loss": _col(["train/dfl_loss",        "train/dfl_om"]),
    }


# Load all data at startup
BENCHMARK    = load_benchmark()
YOLO_HISTORY = load_yolo_history()

# Derived KPIs (safe with empty data)
def _best_row() -> pd.Series | None:
    if BENCHMARK.empty:
        return None
    classif = BENCHMARK[BENCHMARK["Model"] != "YOLOv8s"]
    if classif.empty:
        return None
    return classif.iloc[0]   # already sorted best→worst

def _yolo_row() -> pd.Series | None:
    if BENCHMARK.empty:
        return None
    yolo = BENCHMARK[BENCHMARK["Model"] == "YOLOv8s"]
    return yolo.iloc[0] if not yolo.empty else None

def _fmt(val, decimals=2, suffix="", na="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return f"{val:.{decimals}f}{suffix}"

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Page Config
st.set_page_config(
    page_title="Aerial Object Classifier & Detector",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #080c18; }
  [data-testid="stSidebar"]          { background: #0d1220; border-right: 1px solid rgba(255,255,255,0.07); }
  [data-testid="stHeader"]           { background: transparent; }
  h1,h2,h3,h4 { color: #e2e8f0 !important; }
  p, li, label { color: #8fa8c0; }

  .hero-wrap {background: linear-gradient(135deg, rgba(56,178,255,0.07) 0%, rgba(181,122,255,0.05) 100%);border: 1px solid rgba(56,178,255,0.18); border-radius: 16px;padding: 2rem 2.5rem 1.5rem; margin-bottom: 1.5rem; text-align: center;}
  .hero-title {font-size: 2.2rem; font-weight: 800;background: linear-gradient(135deg, #38b2ff 0%, #b57aff 100%);-webkit-background-clip: text; -webkit-text-fill-color: transparent;margin-bottom: 0.3rem;}
  .hero-sub  { color: #5a7090; font-size: 0.95rem; margin-bottom: 1rem; }
  .hero-tags { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
  .tag { font-size: 11px; font-weight: 600; padding: 3px 11px; border-radius: 20px; border: 1px solid; display: inline-block; }
  .tag-blue   { background: rgba(56,178,255,0.1);  border-color: rgba(56,178,255,0.3);   color: #6dcfff; }
  .tag-green  { background: rgba(74,240,154,0.1);  border-color: rgba(74,240,154,0.3);   color: #5af5a8; }
  .tag-amber  { background: rgba(255,204,68,0.1);  border-color: rgba(255,204,68,0.3);   color: #ffd055; }
  .tag-purple { background: rgba(181,122,255,0.1); border-color: rgba(181,122,255,0.3);  color: #c594ff; }
  .tag-gray   { background: rgba(255,255,255,0.05);border-color: rgba(255,255,255,0.12); color: #8fa8c0; }

  .kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 1.25rem; }
  .kpi { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem 1.2rem; }
  .kpi-label { font-size: 10px; color: #4a6080; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 5px; }
  .kpi-value { font-size: 1.6rem; font-weight: 800; font-family: 'Courier New', monospace; line-height: 1; }
  .kpi-sub   { font-size: 10px; color: #4a6080; margin-top: 4px; }
  .kv-blue   { color: #38b2ff; } .kv-green { color: #4af09a; }
  .kv-amber  { color: #ffcc44; } .kv-purple{ color: #b57aff; }
  
  .model-card-live {background: rgba(56,178,255,0.04);border-radius: 12px;border: 0.5px solid rgba(0,0,0,0.12);padding: 1rem 1.25rem;display: flex;flex-direction: column;gap: 10px;width: 100%;box-sizing: border-box;}
  .model-card-live.loaded { border-left: 3px solid #1D9E75; }
  .model-card-live.missing { border-left: 3px solid #E24B4A; }
  .model-card-live.best { border-color: rgba(56,178,255,0.35); background: rgba(56,178,255,0.04); }
  .card-header { display: flex; align-items: center; gap: 10px; }
  .card-icon {width: 36px; height: 36px;border-radius: 8px;display: flex; align-items: center; justify-content: center;flex-shrink: 0; font-size: 16px;}
  .card-icon.ok  { background: #46923c; }
  .card-icon.warn { background: #ece75f; }
  .card-title { font-size: 11px; font-weight: 600; color: #e2e8f0; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }
  .card-name  { font-size: 15px; font-weight: 600; color: #3a6080; margin: 0; line-height: 1.3; }
  
  .model-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 1.2rem 1.1rem 1rem; margin-bottom: 10px; position: relative;overflow: hidden;}
  .model-card.best { border-color: rgba(56,178,255,0.35); background: rgba(56,178,255,0.04); }
  .model-name  { font-size: 13px; font-weight: 700; color: #e2e8f0; margin-bottom: 2px; }
  .model-sub   { font-size: 10px; color: #4a6080; margin-bottom: 10px; }
  .model-metric{ display: flex; justify-content: space-between; margin-bottom: 4px; }
  .mkey { font-size: 10px; color: #5a7090; }
  .mval { font-size: 11px; font-weight: 700; font-family: 'Courier New', monospace; color: #94b4d4; }
  .mc-acc-row  { display: flex; align-items: baseline; gap: 5px; margin-bottom: 8px; }
  .mc-acc-num  { font-size: 1.55rem; font-weight: 800; font-family: 'Courier New', monospace; line-height: 1; }
  .mc-acc-lbl  { font-size: 10px; color: #4a6080; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }
  
  .bar-bg { width: 100%; height: 4px; background: rgba(255,255,255,0.07); border-radius: 2px; margin-bottom: 10px; }
  .bar-fg { height: 4px; border-radius: 2px; transition: width 0.6s ease; }
  
  .pred-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; }
  .pred-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
  .pred-label { font-size: 2rem; font-weight: 800; margin-bottom: 0.25rem; }
  .pred-conf  { font-size: 1rem; color: #5a7090; }

  .upload-zone { background: rgba(56,178,255,0.03); border: 2px dashed rgba(56,178,255,0.2); border-radius: 16px; padding: 3rem 2rem; text-align: center; }
  .upload-icon  { font-size: 3.5rem; margin-bottom: 1rem; }
  .upload-title { font-size: 1.2rem; font-weight: 600; color: #94b4d4; margin-bottom: 0.5rem; }
  .upload-sub   { font-size: 0.85rem; color: #4a6080; }

  .pipeline { display: flex; flex-wrap: wrap; align-items: center; gap: 0; row-gap: 5px; }
  .pipe-box { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 5px; padding: 4px 11px; font-size: 10.5px; color: #7a9abc; white-space: nowrap; }
  .pipe-arr { font-size: 9px; color: #2a4060; padding: 0 4px; }

  .sidebar-logo       { text-align: center; padding: 1rem 0 0.5rem; margin-bottom: 0.75rem; }
  .sidebar-logo-icon  { font-size: 2.5rem; }
  .sidebar-logo-title { font-size: 1rem; font-weight: 700; color: #e2e8f0; }
  .sidebar-logo-sub   { font-size: 12px; color: #4a6080; }
  div.stButton > button {width: 20rem;text-align: center;padding: 0.6rem 1rem;border-radius: 8px;border: 0.5px solid rgba(0,0,0,0.12);background: linear-gradient(135deg, rgba(56,178,255,0.07) 0%, rgba(181,122,255,0.05) 100%);font-size: 14px;font-weight: 400;transition: background 0.15s;}
  div.stButton > button:hover {background: rgba(0,0,0,0.05);border-color: rgba(0,0,0,0.2);}
  div.stButton > button:focus {background: rgba(29,158,117,0.1);border-color: #1D9E75;color: #085041;font-weight: 500;}
  [data-testid="stSidebar"] .stButton {display: flex;justify-content: center;padding: 0 1.5rem;}
  .pill-ok   { background: rgba(74,240,154,0.12); border: 1px solid rgba(74,240,154,0.25);  color: #4af09a; font-size: 14px; font-weight: 600; padding:0.2rem 0.6rem; border-radius: 10px; display: inline-block; }
  .pill-warn { background: rgba(255,204,68,0.12);  border: 1px solid rgba(255,204,68,0.25);  color: #ffcc44; font-size: 14px; font-weight: 600; padding:0.2rem 0.6rem; border-radius: 10px; display: inline-block; }
  .js-plotly-plot .plotly { background: transparent !important; }

  /* ── Data source badge ── */
  .src-badge {display: inline-block; font-size: 9px; font-weight: 700; letter-spacing: 0.08em;text-transform: uppercase; padding: 2px 8px; border-radius: 6px;background: rgba(74,240,154,0.1); border: 1px solid rgba(74,240,154,0.25);color: #4af09a; margin-left: 8px; vertical-align: middle;}
</style>
""", unsafe_allow_html=True)

# Plotly dark theme defaults
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#7a8fab", family="'Courier New', monospace"),
    margin=dict(t=40, b=30, l=10, r=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.07)", borderwidth=1),
)

# Model Loaders
@st.cache_resource(show_spinner="Loading classification model...")
def load_classifier():
    scripted = MODELS_DIR / "best_classification_model_scripted.pt"
    if scripted.exists():
        m = torch.jit.load(str(scripted), map_location=DEVICE)
        m.eval(); return m
    ckpt = MODELS_DIR / "best_classification_model.pt"
    if ckpt.exists():
        m = torch.load(str(ckpt), map_location=DEVICE)
        return m
    return None

@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_yolo():
    p = MODELS_DIR / "yolov8_best.pt"
    return YOLO(str(p)) if p.exists() else None

# Inference helpers
def classify_image(model, image: Image.Image):
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    label      = "Drone" if prob > 0.5 else "Bird"
    confidence = prob if label == "Drone" else 1.0 - prob
    return label, confidence, prob

def detect_objects(model, image: Image.Image, conf=0.25, iou=0.45):
    img_bgr   = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results   = model(img_bgr, conf=conf, iou=iou, verbose=False)[0]
    annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
    dets = []
    if results.boxes:
        for box in results.boxes:
            dets.append({"class": CLASSES[int(box.cls[0])],
                         "confidence": float(box.conf[0]),
                         "box": box.xyxy[0].tolist()})
    return Image.fromarray(annotated), dets

def make_gauge(confidence, label):
    color = CLASS_COLORS.get(label, "#9C27B0")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=confidence * 100,
        title={"text": f"{CLASS_EMOJIS.get(label,'')} {label}", "font": {"size": 15, "color": "#e2e8f0"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#4a6080"},
               "bar": {"color": color},
               "bgcolor": "rgba(0,0,0,0)",
               "steps": [{"range": [0,  50], "color": "rgba(255,80,80,0.1)"},
                          {"range": [50, 80], "color": "rgba(255,204,68,0.1)"},
                          {"range": [80,100], "color": "rgba(74,240,154,0.1)"}],
               "bordercolor": "rgba(255,255,255,0.08)"},
        number={"suffix": "%", "font": {"size": 26, "color": "#e2e8f0"}}
    ))
    fig.update_layout(**{**PLOT_LAYOUT, "height": 240, "margin": dict(t=40, b=0, l=20, r=20)})
    return fig

def make_prob_bar(raw_prob):
    bird_p, drone_p = (1.0 - raw_prob) * 100, raw_prob * 100
    fig = go.Figure(go.Bar(
        x=["Bird 🦅", "Drone 🚁"],
        y=[bird_p, drone_p],
        marker_color=["#4af09a", "#ff6b6b"],
        text=[f"{bird_p:.1f}%", f"{drone_p:.1f}%"],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig.update_layout(**{**PLOT_LAYOUT,
        "title": {"text": "Class Probability", "font": {"size": 13, "color": "#7a8fab"}},
        "height": 240, "yaxis": {**PLOT_LAYOUT["yaxis"], "range": [0, 115]},
        "showlegend": False,
    })
    return fig

def benchmark_bar():
    df = BENCHMARK.dropna(subset=["Accuracy"])
    colors = [MODEL_COLORS.get(m, "#8fa8c0") for m in df["Model"]]
    fig = go.Figure(go.Bar(
        x=df["Model"], y=df["Accuracy"],
        marker_color=colors,
        text=[f"{v:.2f}%" for v in df["Accuracy"]],
        textposition="outside", textfont=dict(color="#e2e8f0"),
    ))
    min_acc = df["Accuracy"].min() if not df.empty else 60
    fig.update_layout(**{**PLOT_LAYOUT,
        "height": 320,
        "yaxis": {**PLOT_LAYOUT["yaxis"], "range": [max(0, min_acc - 10), 105], "title": "Accuracy (%)"},
        "showlegend": False,
    })
    return fig

def prf_grouped():
    df = BENCHMARK.dropna(subset=["Precision"])
    fig = go.Figure()
    for metric, color in [("Precision","#38b2ff"), ("Recall","#4af09a"), ("F1","#ffcc44")]:
        fig.add_trace(go.Bar(name=metric, x=df["Model"], y=df[metric],
                             marker_color=color, opacity=0.85))
    fig.update_layout(**{**PLOT_LAYOUT,
        "height": 300, "barmode": "group",
        "yaxis": {**PLOT_LAYOUT["yaxis"], "range": [0, 1.1]},
    })
    return fig

def scatter_time_acc():
    fig = go.Figure()
    for _, row in BENCHMARK.iterrows():
        if pd.isna(row.get("Accuracy")) or pd.isna(row.get("Time (min)")):
            continue
        fig.add_trace(go.Scatter(
            x=[row["Time (min)"]], y=[row["Accuracy"]],
            mode="markers+text",
            marker=dict(size=14, color=MODEL_COLORS.get(row["Model"], "#8fa8c0")),
            text=[row["Model"]], textposition="top center",
            textfont=dict(size=10, color="#8fa8c0"),
            name=row["Model"], showlegend=False,
        ))
    min_acc = BENCHMARK["Accuracy"].dropna().min() if not BENCHMARK.empty else 60
    fig.update_layout(**{**PLOT_LAYOUT,
        "height": 300,
        "xaxis": {**PLOT_LAYOUT["xaxis"], "title": "Training Time (min)"},
        "yaxis": {**PLOT_LAYOUT["yaxis"], "title": "Accuracy (%)", "range": [max(0, min_acc - 10), 105]},
    })
    return fig

def yolo_map_chart():
    if not YOLO_HISTORY or not YOLO_HISTORY.get("epochs"):
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=YOLO_HISTORY["epochs"], y=YOLO_HISTORY["mAP50"],
        name="mAP50", line=dict(color="#ffcc44", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,204,68,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=YOLO_HISTORY["epochs"], y=YOLO_HISTORY["mAP5095"],
        name="mAP50-95", line=dict(color="#ff8c5a", width=2, dash="dash"),
    ))
    fig.update_layout(**{**PLOT_LAYOUT,
        "height": 280,
        "xaxis": {**PLOT_LAYOUT["xaxis"], "title": "Epoch"},
        "yaxis": {**PLOT_LAYOUT["yaxis"], "title": "mAP", "range": [0, 1.0]},
    })
    return fig

def yolo_loss_chart():
    if not YOLO_HISTORY or not YOLO_HISTORY.get("epochs"):
        return go.Figure()
    fig = go.Figure()
    for key, name, color in [
        ("box_loss","Box Loss","#38b2ff"),
        ("cls_loss","Cls Loss","#4af09a"),
        ("dfl_loss","DFL Loss","#ff6b6b"),
    ]:
        if YOLO_HISTORY.get(key):
            fig.add_trace(go.Scatter(
                x=YOLO_HISTORY["epochs"], y=YOLO_HISTORY[key],
                name=name, line=dict(color=color, width=2),
            ))
    fig.update_layout(**{**PLOT_LAYOUT,
        "height": 280,
        "xaxis": {**PLOT_LAYOUT["xaxis"], "title": "Epoch"},
        "yaxis": {**PLOT_LAYOUT["yaxis"], "title": "Loss"},
    })
    return fig

# Sidebar
with st.sidebar:
    dev = "GPU 🚀" if torch.cuda.is_available() else "CPU 💻"
    st.markdown(f"""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">🛸</div>
      <div class="sidebar-logo-title">Aerial Classifier</div>
      <div class="sidebar-logo-sub">Bird · Drone · Detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; margin: 4px 0;">
      <span style="font-size: 14px; font-weight: 600; color: var(--text-color, #888);">Device: </span>
      <code style="font-size: 14px; font-weight: 600;">{dev}</code>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    nav_items = [
        ("🏠", "Project Overview"),
        ("📊", "Training Results"),
        ("🔍", "Live Inference"),
    ]

    if "page" not in st.session_state:
        st.session_state.page = "Project Overview"

    for icon, label in nav_items:
        if st.button(f"{icon}  {label}", key=label):
            st.session_state.page = label

    page = st.session_state.page

# Load inference models once
clf_model  = load_classifier()
yolo_model = load_yolo()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PROJECT OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:

    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-title">🛸 AERIAL OBJECT CLASSIFICATION & DETECTION</div>
      <div class="hero-sub">Custom CNN · ResNet50 · MobileNetV2 · EfficientNetB0 · YOLOv8s &nbsp;|&nbsp; By <strong>Deepanshu Gupta</strong></div>
      <div class="hero-tags">
        <span class="tag tag-blue">PyTorch 2.11.0+cu126</span>
        <span class="tag tag-blue">Transfer Learning</span>
        <span class="tag tag-green">YOLOv8s</span>
        <span class="tag tag-amber">Streamlit</span>
        <span class="tag tag-purple">RTX 4060 · CUDA</span>
        <span class="tag tag-gray">Bird vs Drone</span>
        <span class="tag tag-gray">Auto-loaded Metrics</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards — all values read from result files
    best = _best_row()
    yolo = _yolo_row()

    best_acc   = _fmt(best["Accuracy"] if best is not None else None, 2, "%", "—")
    best_name  = best["Model"] if best is not None else "—"
    yolo_map   = _fmt(yolo["Accuracy"] if yolo is not None else None, 2, "%", "—")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Best Test Accuracy</div><div class="kpi-value kv-blue">{best_acc}</div><div class="kpi-sub">{best_name}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="kpi"><div class="kpi-label">Total Images</div><div class="kpi-value kv-green">6719</div><div class="kpi-sub">Bird + Drone</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi"><div class="kpi-label">YOLO Best mAP50</div><div class="kpi-value kv-amber">{yolo_map}</div><div class="kpi-sub">From results.csv</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="kpi"><div class="kpi-label">Models Trained</div><div class="kpi-value kv-purple">5</div><div class="kpi-sub">4 classifiers + YOLOv8</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([0.8, 1])
    with left:
        st.markdown("#### 🎯 Problem Statement & Objective")
        st.markdown("""
        <p style='color:#7a8fab; font-size:18px; line-height:1.8; text-align:justify;'>
        Automatically distinguish <b style='color:#4af09a;'>Birds</b> from <b style='color:#ff6b6b;'>Drones</b>
        in aerial imagery — critical for <b style='color:#94b4d4;'>airspace security</b>,
        wildlife monitoring, and anti-drone defense systems.
        </p>
        <p style='color:#7a8fab; font-size:18px; line-height:1.8; text-align:justify;'>
        Five models across two paradigms are built, compared, and deployed in a real-time web app:
        image classification (4 models) and object detection with bounding boxes (YOLOv8s).
        </p>
        """, unsafe_allow_html=True)

    with right:
        cl, cr = st.columns([1.2, 1])
        with cl:
            st.markdown("<div style='text-align:center;'><h4>📦 Classification</h4></div>", unsafe_allow_html=True)
            st.markdown("""
            | Split | Bird | Drone | Total |
            |:-----:|:----:|:-----:|:-----:|
            | Train | 1,414 | 1,248 | 2,662 |
            | Validation | 217 | 225 | 442 |
            | Test | 121 | 94 | 215 |
            | **Total** | **1,752** | **1,567** | **3,319** |
            """)
        with cr:
            st.markdown("<div style='text-align:center;'><h4>📦 Object Detection</h4></div>", unsafe_allow_html=True)
            st.markdown("""
            | Split | Images | Labels |
            |:-----:|:------:|:------:|
            | Train | 2,728 | 2,728 |
            | Validation | 448 | 448 |
            | Test | 224 | 224 |
            | **Total** | **3,400** | **3,400** |
            """)

    st.markdown("---")
    st.markdown("#### ✅ Pipeline Status")
    statuses = [
        "Import Libraries", "Data Configuration", "EDA — Classification + YOLO",
        "Preprocessing & Augmentation", "Model Training × 5", "Training Results & Visualizations",
        "Best Models Saved", "Sample Predictions Visualized", "Streamlit App Development", "Application Launched",
    ]
    cols = st.columns(4)
    for i, s in enumerate(statuses):
        cols[i % 4].markdown(
            f'<span class="pill-ok">✓</span> &nbsp;<span style="color:#7a9abc;font-size:16px;">{s}</span>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAINING RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Training" in page:

    st.markdown("## 📊 Training Results")
    st.markdown("### 🎯 Classification Detection")
    st.markdown(
        'All 4 classification models — metrics auto-loaded from '
        '<span class="src-badge">model_comparison_results.csv</span>',
        unsafe_allow_html=True)

    # ── Model cards — values pulled from BENCHMARK & HISTORY ─────────────────
    classif_df = BENCHMARK[BENCHMARK["Model"] != "YOLOv8s"] if not BENCHMARK.empty else pd.DataFrame()
    model_order = ["ResNet50", "MobileNetV2", "EfficientNetB0", "Custom_CNN"]
    cols = st.columns(4)
    for col, name in zip(cols, model_order):
        row = classif_df[classif_df["Model"] == name]
        if not row.empty:
            acc_val  = float(row.iloc[0]["Accuracy"])
            acc_str  = f"{acc_val:.1f}%"
        else:
            acc_val, acc_str = 0.0, "—"

        color    = MODEL_COLORS.get(name, "#8fa8c0")
        is_best  = (not classif_df.empty and classif_df.iloc[0]["Model"] == name)

        with col:
            badge      = "best" if is_best else ""
            best_label = " 🏆" if is_best else ""
            bar_w      = acc_val
            sub_type   = "From Scratch" if name == "Custom_CNN" else "Transfer"
            st.markdown(f"""
            <div class="model-card {badge}">
              <div class="model-name">{name}{best_label}</div>
              <div class="model-sub">{sub_type}</div>
              <div class="mc-acc-row">
                <span class="mc-acc-num" style="color:{color};">{acc_str}</span>
                <span class="mc-acc-lbl">Test Accuracy</span>
              </div>
              <div class="bar-bg"><div class="bar-fg" style="width:{bar_w}%;background:{color};"></div></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🎯 YOLOv8s Object Detection Results")
    st.markdown(
        'Epoch metrics auto-loaded from '
        '<span class="src-badge">runs/detect/.../results.csv</span>',
        unsafe_allow_html=True)

    yolo = _yolo_row()
    if YOLO_HISTORY and YOLO_HISTORY.get("mAP50"):
        best_map50   = max(YOLO_HISTORY["mAP50"])
        best_map5095 = max(YOLO_HISTORY["mAP5095"]) if YOLO_HISTORY.get("mAP5095") else None
        best_epoch   = YOLO_HISTORY["epochs"][YOLO_HISTORY["mAP50"].index(best_map50)]
    else:
        best_map50 = best_map5095 = best_epoch = None

    yolo_prec = float(yolo["Precision"]) if yolo is not None and pd.notna(yolo.get("Precision")) else None
    yolo_rec  = float(yolo["Recall"])    if yolo is not None and pd.notna(yolo.get("Recall"))    else None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Best mAP50</div><div class="kpi-value kv-amber">{_fmt(best_map50*100 if best_map50 else None,1,"%")}</div><div class="kpi-sub">Epoch {best_epoch}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Best mAP50-95</div><div class="kpi-value kv-amber">{_fmt(best_map5095*100 if best_map5095 else None,1,"%")}</div><div class="kpi-sub">From results.csv</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Final Precision</div><div class="kpi-value kv-green">{_fmt(yolo_prec*100 if yolo_prec else None,1,"%")}</div><div class="kpi-sub">Last epoch</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Final Recall</div><div class="kpi-value kv-blue">{_fmt(yolo_rec*100 if yolo_rec else None,1,"%")}</div><div class="kpi-sub">Last epoch</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Peak Accuracy — All Models**")
        st.plotly_chart(benchmark_bar(), use_container_width=True)
    with c2:
        st.markdown("**Precision · Recall · F1 — Grouped**")
        st.plotly_chart(prf_grouped(), use_container_width=True)
        
    if YOLO_HISTORY and YOLO_HISTORY.get("epochs"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**mAP50 & mAP50-95 Progression**")
            st.plotly_chart(yolo_map_chart(), use_container_width=True)
        with c2:
            st.markdown("**Box · Class · DFL Losses**")
            st.plotly_chart(yolo_loss_chart(), use_container_width=True)

    else:
        st.info("📂 YOLO `results.csv` not found. Run YOLO training first.")
    
    st.markdown("---")

    st.markdown("## 🏆 Final Model Benchmark")
    st.markdown(
        'Metrics auto-loaded from '
        '<span class="src-badge">assets/model_comparison_results.csv</span>',
        unsafe_allow_html=True)

    if BENCHMARK.empty:
        st.warning("No benchmark data found. Run the training notebook first.")
    else:
        # Full formatted table
        st.markdown("#### 📋 Full Metric Table")
        display_df = BENCHMARK.copy()
        display_df["Accuracy"]   = display_df["Accuracy"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
        display_df["Precision"]  = display_df["Precision"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        display_df["Recall"]     = display_df["Recall"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        display_df["F1"]         = display_df["F1"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        display_df["Loss"]       = display_df["Loss"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        display_df["Z-Score"]    = display_df["Z-Score"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        display_df["Time (min)"] = display_df["Time (min)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        display_df = display_df.drop(columns=["training_time_sec", "accuracy_dec", "z_score_dec"], errors="ignore")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Inference" in page:

    st.markdown("## 🔍 Live Inference")

    def model_card(title, model_name, loaded):
        state = "loaded" if loaded else "missing"
        icon = "✓" if loaded else "!"
        icon_cls = "ok" if loaded else "warn"
        pill_cls = "ok" if loaded else "warn"
        pill_txt = "Loaded" if loaded else "Not found — run notebook first"
        return f"""
        <div class="model-card-live {state}">
            <div class="card-header">
                <div class="card-icon {icon_cls}">{icon}</div>
                <div>
                <p class="card-title">{title}</p>
                <p class="card-name">{model_name}</p>
                </div>
            </div>
        </div>
        """

    # REPLACE the s1, s2 columns block in the Live Inference page with this:

    clf_loaded_html  = '<span class="pill-ok">Loaded</span>' if clf_model  else '<span class="pill-warn">Not found</span>'
    yolo_loaded_html = '<span class="pill-ok">Loaded</span>' if yolo_model else '<span class="pill-warn">Not found</span>'
    dev_label = "GPU · CUDA" if torch.cuda.is_available() else "CPU"
    gpu_name  = "RTX 4060"   # update if needed

    s1, s2 = st.columns([1, 1.1])
    with s1:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:14px;padding:1.1rem 1.3rem;display:flex;flex-direction:column;gap:10px;">

          <div style="display:flex;align-items:center;justify-content:space-between;
                      background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                      border-radius:10px;padding:0.75rem 1rem;">
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="width:32px;height:32px;background:#1a6e4a;border-radius:7px;
                          display:flex;align-items:center;justify-content:center;
                          font-size:14px;flex-shrink:0;">✓</div>
              <div>
                <div style="font-size:18px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                            color:#d0dcea;margin-bottom:2px;">Classification Model</div>
                <div style="font-size:14px;font-weight:600;color:#4a6080;">ResNet50 classifier</div>
              </div>
            </div>
            {clf_loaded_html}
          </div>

          <div style="display:flex;align-items:center;justify-content:space-between;
                      background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                      border-radius:10px;padding:0.75rem 1rem;">
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="width:32px;height:32px;background:#1a6e4a;border-radius:7px;
                          display:flex;align-items:center;justify-content:center;
                          font-size:14px;flex-shrink:0;">✓</div>
              <div>
                <div style="font-size:18px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                            color:#d0dcea;margin-bottom:2px;">Object Detection Model</div>
                <div style="font-size:14px;font-weight:600;color:#4a6080;">YOLOv8s · Roboflow annotations</div>
              </div>
            </div>
            {yolo_loaded_html}
          </div>

          <div style="display:flex;align-items:center;justify-content:space-between;
                      padding:0.5rem 0.25rem 0.1rem;">
            <div style="display:flex;align-items:center;gap:7px;">
              <span style="width:12px;height:12px;background:#4af09a;border-radius:50%;display:inline-block;"></span>
              <span style="font-size:16px;color:#5a7090;">Running on</span>
              <span style="font-size:16px;font-weight:700;color:#d0dcea;">{dev_label}</span>
            </div>
            <span style="font-size:16px;color:#4a6080;">{gpu_name}</span>
          </div>

        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div style="background:rgba(56,178,255,0.04);border:1px solid rgba(56,178,255,0.12);
                    border-radius:14px;padding:0.6rem 1.4rem;margin-bottom:0.5rem;">
        <div style="font-size:22px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                    color:{color};margin-bottom:0.9rem;">🤖 Model Selection</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
          div[data-testid="stRadio"] label p {
            padding-left: 1rem;
            font-size: 17px !important;
            font-weight: 500 !important;
          }
        </style>
        """, unsafe_allow_html=True)

        task = st.radio(
            "Select model",
            ["Classification Model", "YoloV8 Detection Model", "Both the Models"],
            index=2,
            label_visibility="collapsed",
        )

        task_meta = {
            "Classification Model": ("🧠", "#b57aff", "Classifies entire image as Bird or Drone"),
            "YoloV8 Detection Model":    ("📦", "#ff6b6b", "Draws bounding boxes around objects"),
            "Both the Models":           ("⚡", "#38b2ff", "Runs classification + detection together"),
        }

        icon, color, desc = task_meta[task]
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid {color};
                    border-radius:8px;padding:0.4rem 0.85rem;margin-top:0.5rem;">
            <span style="font-size:16px;font-weight:700;color:{color};">{icon} {task}</span><br>
            <span style="font-size:13px;color:#4a6080;">{desc}</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload aerial image(s)",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown("""
        <div class="upload-zone">
          <div class="upload-icon">🛸</div>
          <div class="upload-title">Drop aerial images here to get started</div>
          <div class="upload-sub">Supports Bird 🦅 and Drone 🚁 · JPG · PNG · JPEG · Multi-image</div>
        </div>
        """, unsafe_allow_html=True)

    for idx, uploaded in enumerate(uploaded_files):
        st.markdown(f"### 🖼️ Image {idx+1} — `{uploaded.name}`")
        image = Image.open(uploaded).convert("RGB")

        n_cols = 1 + (task in ["Classification Model","Both the Models"] and bool(clf_model)) + (task in ["YoloV8 Detection Model", "Both the Models"] and bool(yolo_model))
        cols    = st.columns(max(n_cols, 1))
        col_idx = 0

        with cols[col_idx]:
            st.image(image, caption="Original", use_container_width=True)
        col_idx += 1

        if task in ["Classification Model","Both the Models"] and clf_model:
            with st.spinner("Classifying..."):
                label, confidence, raw_prob = classify_image(clf_model, image)
            with cols[col_idx]:
                color = CLASS_COLORS[label]; emoji = CLASS_EMOJIS[label]
                st.markdown(f"""
                <div class="pred-card">
                  <div class="pred-emoji">{emoji}</div>
                  <div class="pred-label" style="color:{color}">{label}</div>
                  <div class="pred-conf">{confidence:.1%} confidence</div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(make_gauge(confidence, label), use_container_width=True)
                st.plotly_chart(make_prob_bar(raw_prob), use_container_width=True)
            col_idx += 1

        if task in ["YoloV8 Detection Model", "Both the Models"] and yolo_model:
            with st.spinner("Detecting objects..."):
                annotated_img, dets = detect_objects(yolo_model, image, 0.5, 0.3)
            with cols[col_idx]:
                st.image(annotated_img, caption=f"YOLOv8 — {len(dets)} detection(s)", use_container_width=True)
                if dets:
                    st.markdown("**Detection Details**")
                    for j, d in enumerate(dets, 1):
                        e = CLASS_EMOJIS[d["class"]]; c = CLASS_COLORS[d["class"]]
                        st.markdown(
                            f"{j}. {e} **{d['class']}** — "
                            f"<span style='color:{c};font-weight:700;'>{d['confidence']:.1%}</span>",
                            unsafe_allow_html=True)
                else:
                    st.info("No objects detected at current threshold.")
            col_idx += 1

    st.markdown("---")
    best = _best_row(); yolo = _yolo_row()
    best_acc_str  = _fmt(best["Accuracy"] if best is not None else None, 2, "%")
    best_name_str = best["Model"] if best is not None else "—"
    yolo_map_str  = _fmt(yolo["Accuracy"] if yolo is not None else None, 1, "%")

    st.markdown(f"""
    <div style="background:rgba(56,178,255,0.04);border:1px solid rgba(56,178,255,0.15);
                border-radius:14px;padding:1.5rem 2rem;text-align:center;">
      <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem;">
        🎉 Pipeline Complete — All 10 Sections Finished!
      </div>
      <div style="font-size:14px;color:#5a7090;max-width:600px;margin:0 auto;line-height:1.7;">
        {best_name_str} achieved <b style="color:#38b2ff;">{best_acc_str} test accuracy</b>.
        YOLOv8s reached <b style="color:#ffcc44;">mAP50 = {yolo_map_str}</b>.
        All metrics auto-loaded from training result files — no manual entry required.
      </div>
    </div>
    """, unsafe_allow_html=True)