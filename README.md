# 🛸 Aerial Object Classification & Detection

<div align="center">

> ### **Data Science with AI Internship @ Labmentix** · Project 3 of the Portfolio Series

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Internship](https://img.shields.io/badge/Internship-Labmentix-0A66C2?style=flat)](https://www.labmentix.in)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)]()

</div>

---

## 🏢 About the Organisation

**[Labmentix](https://www.labmentix.in)** is an **ISO 9001:2015 certified** ed-tech and industry training organisation headquartered in **Bengaluru, Karnataka (560029)**. Founded by **Swaraj Phatangade**, Labmentix bridges the gap between academic theory and industry-ready data science practice by immersing interns in real-world, production-grade problem statements from day one.

| Detail | Info |
|---|---|
| 🏙️ **Location** | Bengaluru, Karnataka 560029 |
| 🌐 **Website** | [www.labmentix.in](https://www.labmentix.in) |
| 🏅 **Certification** | ISO 9001:2015 |
| 👤 **Founder & CEO** | Swaraj Phatangade |
| 💼 **Internship Role** | Data Science with AI Intern |
| 📅 **Duration** | 15 Mar 2026 → 15 Jun 2026 (3 Months) |
| 👨‍💻 **Intern** | Deepanshu Gupta |

---

## 📌 Project Overview

This project tackles the real-world challenge of **automatically distinguishing between birds and drones** in aerial imagery — a task critical for airspace security, wildlife monitoring, and anti-drone defence systems. Five deep learning models are built, trained, benchmarked, and deployed across two computer vision paradigms: **image classification** and **object detection**.

| Attribute | Detail |
|---|---|
| 📦 **Classification Dataset** | 3,319 images — Bird & Drone classes (Roboflow Universe, CC BY 4.0) |
| 📦 **Detection Dataset** | 3,400 YOLOv8-format annotated images with bounding boxes |
| 🧠 **Models** | Custom CNN · ResNet50 · MobileNetV2 · EfficientNetB0 · YOLOv8s |
| 🎯 **Project Type** | Real-Time Object Classification & Detection |
| 🚀 **Deployment** | Streamlit web application with live inference |
| 💻 **Hardware** | NVIDIA RTX 4060 GPU · CUDA 12.6 · PyTorch 2.2 |

---

## 🗂️ Repository Structure

```
Project_3_Aerial_Object_Classification_&_Detection/
│
├── Aerial Object Classification & Detection.docx   # Project brief & problem statement
├── Aerial_Object_Classification_Report.docx        # Full technical report
│
└── Development/
    ├── assets/                                     # Saved visualisation PNGs
    │   ├── accuracy_vs_time_all_models.png
    │   ├── all_confusion_matrices.png
    │   ├── augmentation_examples.png
    │   ├── class_distribution.png
    │   ├── Custom_CNN_history.png
    │   ├── EfficientNetB0_history.png
    │   ├── MobileNetV2_history.png
    │   ├── model_comparison_all5.png
    │   ├── model_comparison_results.csv
    │   ├── pixel_distribution.png
    │   ├── precision_recall_curves.png
    │   ├── predictions_ResNet50.png
    │   ├── ResNet50_history.png
    │   ├── sample_images.png
    │   ├── yolo_class_dist.png
    │   ├── yolo_sample_annotations.png
    │   └── yolov8_inference_samples.png
    │
    ├── classification_dataset/                     # Classification dataset
    │   ├── train/                                  # Training images (Bird / Drone)
    │       ├── birds/images*.png
    │       └── drones/images*.png
    │   ├── valid/                                  # Validation images
    │       ├── birds/images*.png
    │       └── drones/images*.png
    │   └── test/                                   # Test images
    │       ├── birds/images*.png
    │       └── drones/images*.png
    │
    ├── models/                                     # Saved model weights
    │   ├── best_classification_model_scripted.pt   # TorchScript export (production)
    │   ├── best_classification_model.pt            # Best classifier state dict
    │   ├── Custom_CNN.pt
    │   ├── EfficientNetB0.pt
    │   ├── MobileNetV2.pt
    │   ├── ResNet50.pt
    │   ├── yolov8_best.onnx                        # ONNX cross-platform export
    │   └── yolov8_best.pt                          # Best YOLOv8 weights
    │
    ├── object_detection_Dataset/                   # YOLO detection dataset
    │   ├── train/
    │       ├── images/images*.png
    │       ├── labels/images*.png
    │       └── labels.cache
    │   ├── valid/
    │       ├── images/images*.png
    │       ├── labels/images*.png
    │       └── labels.cache
    │   ├── test/
    │       ├── images/images*.png
    │       ├── labels/images*.png
    │       └── labels.cache
    │   ├── data.yaml                               # YOLO dataset configuration
    │   ├── README.dataset.txt
    │   └── README.roboflow.txt
    │
    ├── runs/detect/                                # YOLOv8 training run artifacts
    │   ├── models/yolov8_runs/birds_drones/
    │       ├── weights/                            # Epoch checkpoints (best, last, epoch0–40)
    │           ├── best.pt
    │           ├── epoch0.pt
    │           ├── epoch10.pt
    │           ├── epoch20.pt
    │           ├── epoch30.pt
    │           ├── epoch40.pt
    │           └── last.pt
    │       ├── args.yaml
    │       ├── labels.png
    │       ├── results.csv                         # Per-epoch training metrics
    │       ├── training_time.csv
    │       ├── confusion_matrix.png
    │       ├── confusion_matrix_normalized.png
    │       ├── BoxF1_curve.png · BoxP_curve.png · BoxPR_curve.png · BoxR_curve.png
    │       └── train_batch*.jpg · val_batch*_labels.jpg · val_batch*_pred.jpg
    │   └── val/
    │       ├── confusion_matrix.png
    │       ├── confusion_matrix_normalized.png
    │       ├── BoxF1_curve.png · BoxP_curve.png · BoxPR_curve.png · BoxR_curve.png
    │       └── val_batch*_labels.jpg · val_batch*_pred.jpg
    │
    ├── Development_Code.ipynb                      # Main annotated Jupyter Notebook
    ├── streamlit_app.py                            # Streamlit inference application
    ├── requirements.txt                            # Full dependency stack
    ├── yolo26n.pt                                  # YOLOv8 nano weights
    └── yolov8s.pt                                  # YOLOv8 small base weights
```

---

## 🔬 Project Pipeline

The project follows a structured **10-section deep learning pipeline**.

---

### Section 1 · 📦 Import Libraries

A unified import block consolidates the full stack for both pipelines in a single cell for reproducibility and clean environment setup.

- **Core stack** — NumPy, Pandas, Matplotlib, Seaborn, OpenCV, PIL for data manipulation and visualisation
- **PyTorch** — Full DL stack (`torch`, `torch.nn`, `torchvision`) with native CUDA GPU support via `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- **Ultralytics YOLOv8** — State-of-the-art single-stage object detection framework
- **Scikit-learn** — Evaluation metrics: confusion matrix, precision, recall, F1, PR-AUC
- **SciPy** — Z-Score computation for statistical significance benchmarking
- **Reproducibility** — Global `SEED=42` + `cudnn.deterministic=True` ensures deterministic results across all experiments

---

### Section 2 · ⚙️ Data Configuration

A single source-of-truth configuration cell controls all paths, hyperparameters, and constants — every downstream cell inherits from these variables.

| Parameter | Classification | YOLOv8 |
|---|---|---|
| Image Size | 224 × 224 px | 640 × 640 px |
| Batch Size | 32 | 16 |
| Epochs | 15 | 50 |
| Classes | `['bird', 'drone']` | `['Bird', 'Drone']` |
| Dataset Root | `./classification_dataset/` | `./object_detection_Dataset/` |
| Output Dir | `./models/` + `./assets/` | Auto-created by Ultralytics |

---

### Section 3 · 🔍 Exploratory Data Analysis (EDA)

#### Part A — Classification Dataset

| Chart | Type | Insight |
|---|---|---|
| Chart 1 — Class Distribution | Side-by-side Bar Charts | Train/valid/test class counts per split; ~11.8% more bird images than drones — minor imbalance, no rebalancing required |
| Chart 2 — Sample Image Grid | 2 × 5 Image Grid | Reveals visual complexity — lighting variation, background clutter, pose diversity across both classes |
| Chart 3 — Pixel Intensity Distribution | Dual Histogram | Grayscale intensity spread per class; overlapping distributions confirm that spatial (convolutional) features — not raw pixel intensity alone — are needed to separate classes |

**Dataset split:**

| Split | Bird | Drone | Total |
|---|---|---|---|
| Train | ~1,500 | ~1,340 | ~2,840 |
| Valid | — | — | ~264 |
| Test | 121 | 94 | 215 |

#### Part B — YOLO Detection Dataset

| Chart | Type | Insight |
|---|---|---|
| Chart 4 — YOLO Dataset Split Summary | Table | Image and label counts per split; every image has a paired `.txt` label file |
| Chart 5 — Bounding Box Class Distribution | Pie Chart | Bird vs Drone bounding box frequency across all splits; reveals class balance for detection training |
| Chart 6 — Sample Annotated Images | 2 × 4 Image Grid | Ground-truth bounding boxes overlaid (green = Bird, red = Drone); validates annotation quality before training begins |

---

### Section 4 · 🔄 Data Preprocessing & Augmentation

**Validation / Test — Deterministic only** (no augmentation, preserves unbiased evaluation):
- Resize to 224 × 224 → `ToTensor()` → ImageNet Normalize (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

**Training — Full augmentation pipeline:**

| Transform | Parameters | Rationale |
|---|---|---|
| `RandomHorizontalFlip` | p=0.5 | Left/right flight directions are symmetric |
| `RandomRotation` | ±30° | Birds and drones appear at arbitrary angles |
| `RandomAffine` | translate=0.15, shear=10, scale=0.8–1.2 | Objects at varying distances and positions |
| `ColorJitter` | brightness/contrast/saturation=0.2, hue=0.05 | Outdoor lighting variation (dawn, dusk, overcast) |
| `Normalize` | ImageNet stats | Required for transfer learning weight compatibility |

**Chart 7 — Augmentation Visualisation:** Original image alongside 9 augmented variants confirming each augmented sample is a synthetically distinct training input.

---

### Section 5 · 🧠 Model Building & Training

Five models trained across two paradigms:

#### Classification Models (PyTorch)

**Training utilities used by all 4 classifiers:**
- **EarlyStopping** — Custom class halts training when `val_loss` stagnates for 7 epochs, then restores best weights
- **ModelCheckpoint** — `torch.save()` captures `.pt` checkpoint only when `val_accuracy` improves
- **ReduceLROnPlateau** — Cuts LR by 70% (`factor=0.3`) when the model plateaus for 4 epochs; `min_lr=1e-7`
- **Loss** — `BCEWithLogitsLoss` (sigmoid + BCE fused, numerically stable for GPU)
- **Optimizer** — Adam, lr=1e-4, weight_decay=1e-5

**Model 1 — Custom CNN (from scratch)**
- 4-block architecture: Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout2d per block
- Filter progression: 3 → 32 → 64 → 128 → 256 (hierarchical feature learning)
- Fully connected head: 512 → 1 with Dropout(0.5)
- ~673,601 trainable parameters

**Models 2, 3, 4 — Transfer Learning (ImageNet pretrained)**

| Model | Backbone Frozen | Head Replaced | Parameters |
|---|---|---|---|
| ResNet50 | ✅ `model.fc` → `nn.Identity()` | Linear(2048→256→64→1) | ~25.6M (backbone) |
| MobileNetV2 | ✅ `model.classifier[1]` → `nn.Identity()` | Linear(1280→256→64→1) | ~3.4M (backbone) |
| EfficientNetB0 | ✅ `model.classifier[1]` → `nn.Identity()` | Linear(1280→256→64→1) | ~5.3M (backbone) |

All backbones frozen at start → classification head trained first → full model unfrozen for end-to-end fine-tuning.

**Model 5 — YOLOv8s (Object Detection)**
- Base: COCO-pretrained `yolov8s.pt` (small variant — best speed/accuracy balance)
- Architecture: CSPDarknet53 backbone → PANet neck → anchor-free detection head (separate classification + regression branches)
- **Training augmentation:** Mosaic (4-image stitching), MixUp (0.1), HSV jitter (hue=0.015, saturation=0.7, value=0.4), rotation=10°, scale=0.5, fliplr=0.5
- Epochs: 50 | Image size: 640 | Batch: 16 | Patience: 10
- Artifacts auto-saved to `runs/detect/models/yolov8_runs/birds_drones/`

---

### Section 6 · 📊 Training Results & Visualisations

**Evaluation metrics computed for all classification models:**
- **Accuracy (%)** — Overall correct predictions on the 215-image test set
- **Precision** — Of all Drone predictions, how many were actually drones (minimises false alarms)
- **Recall** — Of all actual drones, how many were caught (minimises missed threats)
- **F1-Score** — Harmonic mean of Precision and Recall
- **Loss** — Binary cross-entropy on test set
- **Z-Score** — Statistical significance vs. random baseline (50%); Z>2 = statistically significant result

**Final Model Results:**

| Model | Test Accuracy | Precision | Recall | F1-Score | Z-Score |
|---|---|---|---|---|---|
| **ResNet50** | **99.07%** | **1.000** | 0.979 | **0.989** | Highest |
| MobileNetV2 | 97.21% | — | — | — | — |
| EfficientNetB0 | 94.88% | — | — | — | — |
| Custom CNN | 79.07% | — | — | 0.731 | — |
| **YOLOv8s** | mAP50: **82.7%** | **89.8%** | 75.3% | — | — |

> YOLOv8s best epoch: 34 of 50 · mAP50-95: 52.1% · Box/cls/dfl losses all showed smooth downward convergence

**Charts generated in this section:**

| # | Chart | Type | Key Insight |
|---|---|---|---|
| 8 | All-model Metrics Comparison | Grouped Bar | Accuracy, Precision, Recall, F1, Loss, Z-Score side-by-side for all 5 models |
| 9–12 | Training History Curves | Dual-axis Line | Accuracy & Loss per epoch for each of the 4 classifiers; train vs. val gap shows generalisation quality |
| 13 | Confusion Matrices (All 4 Classifiers) | 2×2 Heatmap Grid | False Negatives (drone → bird) are the critical failure mode for security deployments |
| 14 | Precision-Recall Curves | Overlay Line Chart | PR-AUC comparison across all 4 models; curves closer to top-right = better at all thresholds |
| 15 | Accuracy vs Training Time | Scatter Plot | Each model plotted as a dot; top-left = ideal (high accuracy, low cost) |
| 16 | YOLO Label Distribution | Auto-generated PNG | Bounding box spatial heatmap and class frequency from Ultralytics |
| 17–21 | YOLO Training Metrics | Auto-generated PNGs | Box/cls/dfl loss curves, mAP50/mAP50-95 progression, P/R/F1/PR curves, confusion matrices, training batch mosaics, val batch predictions |

---

### Section 7 · 💾 Save Best Models

**Best Classification Model** saved in two formats:
- **`.pt` state dict** (`best_classification_model.pt`) — Standard PyTorch checkpoint with weights + class metadata + image size; portable and version-stable
- **TorchScript** (`best_classification_model_scripted.pt`) — Serialised, optimised computation graph via `torch.jit.trace()`; runs without the original Python class definition — ideal for production inference servers, edge devices, and C++ deployment via LibTorch

**Best YOLOv8 Model** saved and verified:
- **`.pt` weights** (`yolov8_best.pt`) — Copied from `runs/detect/.../weights/best.pt` and verified by reload
- **ONNX export** (`yolov8_best.onnx`) — Cross-platform format for CPU inference on edge devices (Raspberry Pi, Jetson Nano) via ONNX Runtime

---

### Section 8 · 🖼️ Sample Predictions & Visualisation

| Chart | Description |
|---|---|
| Chart 22 — Classification Predictions | 12 test images with ground-truth vs. predicted labels (green = correct, red = incorrect) and confidence scores — reveals if errors cluster around specific conditions |
| Chart 23 — YOLOv8 Detection Predictions | 8 test images with YOLO-drawn bounding boxes, class labels, and confidence scores at `conf=0.25` threshold |

---

### Section 9 · 🚀 Streamlit Application Development

A full-featured multi-page inference web app (`streamlit_app.py`) written and saved entirely from within the notebook — self-contained deployment, no separate IDE required.

**App architecture:**

| Page | Content |
|---|---|
| **Page 1 — Project Overview** | Summary dashboard: best accuracy, total images, YOLO mAP50, pipeline status checklist |
| **Page 2 — Training Results** | Auto-loaded from `MODEL_COMPARISON_RESULTS.CSV` and `results.csv`; accuracy cards, YOLO metric cards, 4 Plotly charts, full sortable model comparison table |
| **Page 3 — Live Inference** | File uploader → model mode selector → real-time prediction with confidence gauge, probability bar chart, bounding boxes, and detection detail table |

**Key app features:**

| Feature | Description |
|---|---|
| Image Upload | Drag & drop single or multiple JPG/PNG images |
| Task Selection | Classification only / Detection only / Both Models |
| Classification Output | Predicted label (Bird/Drone) + confidence gauge + probability bar |
| YOLOv8 Detection Output | Bounding boxes with class labels and confidence scores on image |
| Confidence Slider | Adjust detection threshold in real-time |
| IoU Slider | Adjust NMS IoU threshold |
| `@st.cache_resource` | Both PyTorch and YOLO models loaded once and cached across sessions — eliminates repeated disk I/O |

---

### Section 10 · ▶️ Launching the Application

```bash
streamlit run streamlit_app.py
```

The launch cell spawns Streamlit as a **non-blocking subprocess** via `subprocess.Popen` — the Jupyter kernel stays alive while the server runs in the background. Open `http://localhost:8501` to interact with the app.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| 🐍 Language | Python 3.10+ |
| 🔥 Deep Learning | PyTorch 2.2+, torchvision 0.17+ |
| 🎯 Object Detection | Ultralytics YOLOv8s |
| 🖼️ Computer Vision | OpenCV, Pillow |
| 🗃️ Data Computation | Pandas, NumPy, SciPy |
| 📊 Visualisation | Matplotlib, Seaborn, Plotly |
| 📐 Evaluation | Scikit-learn (confusion matrix, PR-AUC, F1) |
| 🌐 Deployment | Streamlit 1.32+ |
| 💾 Model Persistence | TorchScript, ONNX, Joblib |
| 💻 Environment | Jupyter Notebook, Google Colab, CUDA 12.6 |
| 🔀 Version Control | Git, GitHub |
| 📽️ Reporting | Word, Video Documentation |

---

## 📦 Key Deliverables

| Deliverable | Description |
|---|---|
| 📓 **Development_Code.ipynb** | Fully annotated 10-section Jupyter Notebook — EDA, 5 models, 23 charts, Streamlit deployment |
| 🎥 **Explanatory_Video.mp4** | Narrated video walkthrough of the entire notebook and findings |
| 📄 **Aerial_Object_Classification_Report.docx** | Full 8-chapter technical report with literature review, methodology, and results discussion |
| 🌐 **streamlit_app.py** | Production-ready 3-page inference web application |
| 📋 **requirements.txt** | Complete pinned dependency stack for both training and deployment |

---

## 👨‍💻 Author

**Deepanshu Gupta**  
Data Science with AI Intern @ Labmentix

---

*© 2026 Deepanshu Gupta · Labmentix Internship Portfolio · All Rights Reserved*
