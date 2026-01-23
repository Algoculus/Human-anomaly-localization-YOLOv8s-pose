# Human Anomaly Localization - Fall Detection System

Real-time fall detection system using YOLOv8s-pose and custom state machine, optimized for safety-critical environments and multi-person tracking.

## 🎯 Project Overview

This project implements an end-to-end fall detection system with:

- **AI Core**: YOLOv8s-pose for human pose estimation with optimized thresholds.
- **Smart Detection**: Hybrid rules-based and vertical velocity (`dy`) state machine.
- **Low-light Support**: Built-in Gamma correction and CLAHE for dark environments.
- **Multi-person Support**: Hungarian matching (IoU + center distance) for tracking multiple individuals.
- **Web Interface**: React + TypeScript frontend, FastAPI backend with WebSocket streaming.

## 📊 Performance Metrics

| Metric      | Target | Achieved    | Status         |
| ----------- | ------ | ----------- | -------------- |
| Accuracy    | ≥0.85  | 0.886       | ✅ Target Met  |
| Precision   | ≥0.85  | 0.893       | ✅ Target Met  |
| **Recall**  | ≥0.85  | 0.833       | ⚠️ Near Target |
| Specificity | ≥0.85  | 0.925       | ✅ Target Met  |
| F1-Score    | ≥0.85  | 0.862       | ✅ Target Met  |

**Confusion Matrix Summary**:
- **True Positives (TP)**: 25
- **True Negatives (TN)**: 37
- **False Positives (FP)**: 3
- **False Negatives (FN)**: 5

## 🏗️ Project Structure

```
Human-anomaly-localization-YOLOv8s-pose/
├── urfd_fall_yolo_pose/           # AI Core - Fall Detection Engine
│   ├── configs/
│   │   └── default.yaml           # Tunable detection & tracking thresholds
│   ├── src/
│   │   └── urfd/
│   │       ├── yolo_pose.py       # YOLOv8s-pose wrapper
│   │       ├── features.py        # Vertical angle & aspect ratio extraction
│   │       ├── tracking.py        # Multi-person Hungarian tracker
│   │       ├── fallback_tracker.py# KCF tracker for occlusion handling
│   │       ├── smoothing.py       # Temporal State Machine (NORMAL→CONFIRMED)
│   │       ├── rules.py           # Core detection logic (Thresholds & dy)
│   │       ├── preprocessing.py   # Low-light enhancement (Gamma/CLAHE)
│   │       ├── overlay.py         # Visualization & Skeleton drawing
│   │       ├── eval.py            # Evaluation & Metrics generation
│   │       └── dataset.py         # URFD Dataset loader
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

## 🧠 Core Features

### 1. Robust Pose Detection
Powered by **YOLOv8s-pose**, extracting 17 keypoints per person. We apply automatic **Low-light Enhancement** using Gamma correction and CLAHE (Contrast Limited Adaptive Histogram Equalization) to ensure reliable detection in varied indoor lighting.

### 2. Hybrid Fall Detection Logic
Multiple logic layers combined for high precision:
- **Geometry**: Trunk angle relative to vertical and Aspect Ratio (AR) of bounding box.
- **Dynamic**: Vertical velocity (`dy`) peak detection to capture the impact phase.
- **Temporal**: State machine requiring consecutive frames or high "fall scores" to confirm a detection, reducing false positives from fast sitting or bending.

### 3. Advanced Tracking & Occlusion
- **Hungarian Tracking**: Maintains identity of multiple people even in crowded scenes.
- **KCF Fallback**: If YOLO fails to detect a person (due to occlusion or motion blur), a Kernelized Correlation Filter (KCF) takes over to track until the person is redetected.

## 📄 License
MIT License - See LICENSE file for details.

---
**Last Updated**: January 23, 2026
