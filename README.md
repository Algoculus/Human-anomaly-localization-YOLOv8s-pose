<<<<<<< HEAD

# Enhanced Fall Detection System - Installation Guide

## 📋 Bước 1: Cập nhật Code

### 1.1 Thay thế các file core

```bash
# Backup files cũ trước
mkdir backup
cp src/rules/features.py backup/
cp src/rules/scoring.py backup/
cp src/rules/state_machine.py backup/
cp src/eval/metrics.py backup/
cp scripts/run_infer.py backup/
cp scripts/run_eval.py backup/
cp scripts/run_webcam.py backup/

# Copy các file mới từ artifacts
# (Bạn đã có các file này từ Claude)
```

### 1.2 Tạo file test

Tạo file `scripts/quick_test.py` với nội dung từ artifact "Quick Test Script"

## 🧪 Bước 2: Verify Installation

```bash
# Chạy quick test
python scripts/quick_test.py
```

**Kết quả mong đợi:**

```
✓ All modules imported successfully
✓ Feature extraction works (Activity: STANDING)
✓ Scoring works (Normal: 0.15, Fall: 0.88)
✓ State machine works (Alert: Fall Sequence Detected)
✓ Metrics work (Acc: 0.92, Recall: 0.94)
✓ ALL TESTS PASSED - System is ready!
```

Nếu có lỗi, xem phần Troubleshooting ở cuối.

## 🚀 Bước 3: Chạy Inference

### 3.1 Test trên 1 sequence

```bash
# Test trên 1 fall sequence
python scripts/run_infer.py --sequence fall-01 --no-video

# Test trên 1 ADL sequence
python scripts/run_infer.py --sequence adl-01 --no-video
```

### 3.2 Test trên nhiều sequences

```bash
# Test 5 sequences đầu tiên (nhanh)
python scripts/run_infer.py --limit 5

# Chạy full dataset (chậm - có thể mất vài giờ)
python scripts/run_infer.py
```

### 3.3 Xem kết quả

```bash
# Check CSV output
cat output/results/fall/fall-01_results.csv

# Xem video (nếu generate)
# Video sẽ ở: output/videos/fall/fall-01_output.mp4
```

## 📊 Bước 4: Evaluation

### 4.1 Test set evaluation

```bash
# Chạy evaluation trên test set
python scripts/run_eval.py --test-only

# Xem kết quả
cat output/eval/test/evaluation_report.txt
cat output/eval/test/metrics_summary.json
```

### 4.2 Full evaluation (train/val/test)

```bash
python scripts/run_eval.py
```

### 4.3 Xem plots

```bash
# Plots sẽ được tạo ở:
# - output/plots/roc_curve.png
# - output/plots/pr_curve.png
# - output/plots/confusion_matrix.png
# - output/plots/evaluation_metrics.png
# - output/plots/timeline_fall-*.png
```

## 🎥 Bước 5: Real-time Webcam

```bash
# Webcam mặc định
python scripts/run_webcam.py

# Specific camera
python scripts/run_webcam.py --source 1

# Video file
python scripts/run_webcam.py --source path/to/video.mp4

# Save output
python scripts/run_webcam.py --save output/webcam_demo.mp4
```

**Phím tắt khi chạy:**

- `q`: Quit
- `r`: Reset state machine

## 📈 Kết quả Kỳ vọng

### Frame-level Metrics

```
Accuracy:              0.89-0.93
Precision:             0.90-0.95
Recall (Sensitivity):  0.88-0.95
Specificity:           0.90-0.95
F1 Score:              0.89-0.94
ROC AUC:               0.92-0.97
```

### Event-level Metrics

```
Precision:             0.92-0.97
Recall (Sensitivity):  0.90-0.98
Specificity:           0.93-0.98
Detection Delay:       0.5-1.2 seconds
False Alarm Rate:      <1.0 alarms/hour
```

### Activity Discrimination

- ✅ Falls: 90-98% detection rate
- ✅ Bending: <5% false positive rate
- ✅ Lying down intentionally: <10% false positive rate
- ✅ ADL activities: >95% correctly ignored

## ⚙️ Fine-tuning Parameters

Nếu kết quả chưa tốt, điều chỉnh trong `src/rules/scoring.py`:

```python
# Tăng Precision (giảm False Positives)
self.FAST_DROP_THRESHOLD = 0.06      # Tăng từ 0.05
self.HIGH_IMPACT_THRESHOLD = 2.7     # Tăng từ 2.5
self.LYING_ORIENTATION = 65          # Tăng từ 60

# Tăng Recall (giảm False Negatives)
self.FAST_DROP_THRESHOLD = 0.04      # Giảm từ 0.05
self.PRONE_THRESHOLD = 0.55          # Giảm từ 0.60
```

Và trong `src/rules/state_machine.py`:

```python
# Stricter (ít FP hơn)
self.FALL_ALERT_THRESHOLD = 0.72     # Tăng từ 0.65
self.PRONE_THRESHOLD = 0.65          # Tăng từ 0.60

# Sensitive (ít FN hơn)
self.FALL_ALERT_THRESHOLD = 0.60     # Giảm từ 0.65
self.t_hold_fast = int(0.3 * self.fps)  # Giảm từ 0.5s
```

## 🐛 Troubleshooting

### Lỗi: "FallStateMachine.update() missing 1 required positional argument"

**Nguyên nhân:** File cũ chưa được thay thế

**Giải pháp:**

```bash
# Xác nhận file mới đã thay thế
python -c "import inspect; from src.rules.state_machine import FallStateMachine; print(inspect.signature(FallStateMachine.update))"
# Phải thấy: (self, scores, features, context)
```

### Lỗi Import

```bash
# Cài đặt dependencies thiếu
pip install numpy pandas scikit-learn matplotlib seaborn loguru tqdm
```

### Performance thấp

1. **Recall thấp (<85%)**:
   - Giảm thresholds
   - Check xem có đủ accelerometer data không
2. **Precision thấp (nhiều FP)**:
   - Tăng thresholds
   - Check activity discrimination logic
3. **Detection delay cao (>1.5s)**:
   - Giảm `t_hold_min`
   - Enable fast confirmation path

### Memory issues

```bash
# Giảm batch size khi process
python scripts/run_infer.py --limit 10  # Process từng đợt nhỏ
```

## 📝 Best Practices

1. **Luôn test trước khi full run:**

   ```bash
   python scripts/quick_test.py
   python scripts/run_infer.py --limit 3
   ```

2. **Monitor logs:**

   ```bash
   python scripts/run_eval.py --test-only 2>&1 | tee eval.log
   ```

3. **Backup config tốt:**

   ```bash
   # Khi tìm được config tốt, save lại
   cp src/rules/scoring.py src/rules/scoring_v1.py
   ```

4. **Iterate từng bước:**
   - Fix False Positives trước
   - Sau đó tối ưu Recall
   - Cuối cùng optimize delay

## 📚 Paper Reference

Hệ thống này implement methodology từ:

> Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded platform using depth maps and wireless accelerometer. Computer Methods and Programs in Biomedicine, 117(3), 489-501.

**Key concepts implemented:**

- Multi-stage detection (Drop → Impact → Prone → Sustained)
- Sensor fusion (Vision + Accelerometer)
- Activity discrimination
- Temporal reasoning
- Event-level evaluation

## 🎯 Next Steps

1. ✅ Verify system works
2. ✅ Run evaluation on test set
3. ✅ Analyze results and tune parameters
4. ✅ Test on real-world scenarios
5. ✅ Deploy to production (if needed)

# Good luck! 🚀

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

> > > > > > > fab3677867154b462e8f8dfdd680162820c72938
