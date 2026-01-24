# Human Anomaly Localization - Fall Detection System

Real-time fall detection system using YOLOv8s-pose and custom state machine, optimized for safety-critical environments and multi-person tracking.

## 🎯 Project Overview

This project implements an end-to-end fall detection system with:

- **AI Core**: YOLOv8s-pose for human pose estimation with optimized thresholds.
- **Smart Detection**: 4-path hybrid detection with impact verification and high-angle detection.
- **Border Integrity Check**: Disables unreliable AR detection when bbox is clipped at image edges.
- **Low-light Support**: Built-in Gamma correction and CLAHE for dark environments.
- **Multi-person Support**: Greedy matching (IoU + center distance) for tracking.
- **Web Interface**: React + TypeScript frontend, FastAPI backend with WebSocket streaming.

## 📊 Performance Metrics

| Metric      | Target | Achieved  | Status        |
| ----------- | ------ | --------- | ------------- |
| Accuracy    | ≥0.85  | **0.900** | ✅ Target Met |
| Precision   | ≥0.85  | **0.926** | ✅ Target Met |
| Recall      | ≥0.80  | **0.833** | ✅ Target Met |
| Specificity | ≥0.90  | **0.950** | ✅ Target Met |
| F1-Score    | ≥0.85  | **0.877** | ✅ Target Met |

**Confusion Matrix Summary (70 Sequences)**:

- **True Positives (TP)**: 25
- **True Negatives (TN)**: 38
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 5

## 🏗️ Project Structure

```
Human-anomaly-localization-YOLOv8s-pose/
├── urfd_fall_yolo_pose/           # AI Core - Fall Detection Engine
│   ├── configs/
│   │   └── default.yaml           # Tunable detection & tracking thresholds
│   ├── src/
│   │   └── urfd/
│   │       ├── yolo_pose.py       # YOLOv8s-pose wrapper with preprocessing
│   │       ├── features.py        # Body angle, AR, dy, border integrity check
│   │       ├── tracking.py        # Multi-person greedy tracker
│   │       ├── fallback_tracker.py# KCF/CSRT tracker for occlusion handling
│   │       ├── smoothing.py       # Temporal State Machine (NORMAL→CONFIRMED)
│   │       ├── rules.py           # 4-path fall detection with high-angle detection
│   │       ├── preprocessing.py   # Low-light enhancement (Gamma/CLAHE)
│   │       ├── overlay.py         # Visualization & skeleton drawing
│   │       ├── eval.py            # Evaluation & metrics generation
│   │       └── dataset.py         # URFD dataset loader
│   ├── scripts/
│   │   ├── eval_all.py            # Full evaluation suite (70 sequences)
│   │   ├── create_plots.py        # Generate CM & metrics visualizations
│   │   ├── create_demo_videos.py  # Export overlay demos
│   │   ├── prepare_urfd.py        # Dataset preprocessing script
│   │   └── infer_sequence.py      # Run inference on a single sequence
│   └── outputs/                   # Performance reports & visualizations
│
├── web/
│   ├── frontend/                  # React + Tailwind UI
│   │   ├── src/
│   │   │   ├── pages/
│   │   │   │   ├── LandingPage.tsx   # Dashboard overview
│   │   │   │   ├── CameraPage.tsx    # Single camera edge simulator
│   │   │   │   ├── ReceiverPage.tsx  # Multi-camera monitoring
│   │   │   │   └── MetricsPage.tsx   # Results visualization
│   │   │   └── components/           # Reusable UI primitives
│   │   └── vite.config.ts         # Vite dev server config
│   │
│   └── backend/                   # FastAPI Production Server
│       └── app/
│           ├── main.py            # Server entry point
│           ├── api/               # REST endpoints (health, upload)
│           └── ws/                # WebSocket connection management
│
├── docs/                          # Vietnamese documentation
│   ├── 1_dataset_analysis.md      # Dataset description
│   ├── 2_metrics_explanation.md   # Metrics formulas
│   ├── 3_pipeline_explanation.md  # Algorithm pipeline
│   ├── 4_source_code_explanation.md # Code documentation
│   └── 5_project_report.md        # Full project report
│
└── data/
    ├── urfd_index.csv             # Dataset index (70 sequences)
    └── raw/                       # URFD raw frames directory
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA GPU (Optional, but recommended for real-time inference)

### 2. Backend & AI Core Setup

```bash
# Install core dependencies
cd urfd_fall_yolo_pose
pip install ultralytics opencv-python pandas pyyaml matplotlib seaborn scikit-learn

# Install backend dependencies
cd ../web/backend
pip install fastapi uvicorn websockets pydantic-settings python-multipart
```

### 3. Frontend Setup

```bash
cd ../frontend
npm install
```

### 4. Running the System

**Terminal 1: FastAPI Backend**

```bash
cd web/backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 9000
```

**Terminal 2: React Frontend**

```bash
cd web/frontend
npm run dev
# Dashboard available at http://localhost:3005
```

### 5. Run Evaluation

```bash
cd urfd_fall_yolo_pose
python scripts/eval_all.py --root ../data --index ../data/urfd_index.csv --config configs/default.yaml
```

## 🧠 Core Features

### 1. Robust Pose Detection

Powered by **YOLOv8s-pose**, extracting 17 keypoints per person. We apply automatic **Low-light Enhancement** using Gamma correction (γ=1.3) and CLAHE.

### 2. 4-Path Hybrid Fall Detection Logic

- **Path 1**: Angle + AR + Impact (requires lying posture with impact velocity ≥5.0)
- **Path 2**: Height Drop detection (bbox height dropped ≥18%)
- **Path 3**: Fast Motion detection (dy ≥10.0 px/frame)
- **Path 4**: High-Angle detection (body angle ≥65° with any motion) - catches slow/frontal falls

### 3. Border Integrity Check

Disables AR-based detection when bbox touches image edges (prevents "close-to-camera" false positives).

### 4. Advanced Tracking & Occlusion

- **Greedy Tracker**: Maintains identity using IoU + center distance cost function.
- **KCF Fallback**: Kernelized Correlation Filter takes over when YOLO misses.

## 📄 License

MIT License - See LICENSE file for details.

---

**Last Updated**: January 24, 2026
