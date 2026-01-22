# Fall Detection - YOLOv8s-pose + Rule-based State Machine

Human fall detection system using YOLOv8s-pose for pose estimation and rule-based state machine for temporal classification.

## Performance Metrics

**Current Best (Baseline Configuration):**

- **Accuracy:** 85.7% (60/70 correct)
- **Precision:** 85.7% (24 TP, 4 FP)
- **Recall:** 80.0% (24/30 falls detected)
- **Specificity:** 90.0% (36/40 ADL classified correctly)
- **F1-Score:** 82.8%

**Confusion Matrix:**

```
TP: 24 (Falls detected)
TN: 36 (ADL correct)
FP: 4  (ADL misclassified as fall)
FN: 6  (Falls missed)
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install ultralytics opencv-python numpy pandas pyyaml matplotlib seaborn

# Verify YOLOv8s-pose model exists
ls yolov8s-pose.pt
```

### 2. Run Full Evaluation

```bash
# Without videos (fast: ~5-7 min)
python scripts/run_full_eval.py

# With videos (slow: ~15-20 min)
python scripts/run_full_eval.py --save-videos
```

### 3. View Results

- **Metrics:** `runs/fixed/metrics.json`
- **Plots:** `outputs/plots/` (confusion_matrix.png, metrics_bar_chart.png)
- **Videos:** `outputs/ADL/` and `outputs/FALL/`
- **Predictions:** `outputs/predictions.csv`

## Architecture

### State Machine (3 States)

```
NORMAL → CANDIDATE → FALL_CONFIRMED
  ↑___________|           |
  (recovery detection)    |
  |_______________________|
```

**State Transitions:**

- **NORMAL → CANDIDATE:** Fast downward motion (dy_peak >= 15.5) OR angle+AR conditions OR height drop
- **CANDIDATE → FALL_CONFIRMED:** Lying posture sustained for >=2 frames (angle OR AR criteria)
- **FALL_CONFIRMED → NORMAL:** Recovery detection (angle < threshold AND AR < threshold AND upward motion)
- **CANDIDATE → NORMAL:** Timeout (no confirmation within window)

### Feature Extraction (`src/urfd/features.py`)

**Primary Features:**

- `bbox`: [x1, y1, x2, y2] bounding box
- `center_y`: Vertical position (for velocity)
- `height`, `width`: Bbox dimensions
- `bbox_aspect_ratio`: width/height (lying indicator)
- `shoulder_mid`, `hip_mid`: Keypoint midpoints
- `body_angle_deg`: Torso angle vs vertical
- `dy_peak`: Max vertical velocity in recent window
- `dy_velocity`: Average velocity
- `dy_acceleration`: Velocity change (new feature)
- `dy_smoothed`: Temporal smoothing (new feature)

**Feature Validity:**

- `feature_valid=True`: All 4 keypoints (shoulders+hips) confident
- `feature_valid=False`: Missing/low-conf keypoints, fallback to bbox-only

### Detection Rules (`src/urfd/rules.py`)

**Fall Candidate Detection (3 paths):**

1. **Angle + AR:** `angle >= 22° AND AR >= 0.9`
2. **Height Drop:** Bbox height drop >= 0.1 normalized
3. **Velocity:** `dy >= 15.5 pixels/frame`

**Lying Posture Confirmation:**

- **Angle OR AR logic:** `(angle >= 40°) OR (AR >= 1.08)`
- **Relaxation:** angle-8°, AR-0.17 for balanced recall/precision

## Configuration (`configs/default.yaml`)

### Tunable Thresholds

```yaml
# Fall candidate entry
dy_peak_thres: 15.5 # Velocity threshold
cand_angle_thres: 22.0 # Angle threshold (deg)
cand_ar_thres: 0.9 # AR threshold

# Fall confirmation
confirm_angle_thres: 48.0 # Lying angle (deg)
confirm_ar_thres: 1.25 # Lying AR
min_confirm_duration_frames: 2 # Min lying duration

# State machine
cand_enter_frames: 2 # Candidate entry frames
confirm_frames: 4 # Confirmation frames
max_candidate_frames: 30 # Timeout

# Recovery detection
recovery_angle_thres: 30.0 # Standing angle
recovery_ar_thres: 0.8 # Standing AR
recovery_dy_thres: -5.0 # Upward motion
```

### YOLO Settings

```yaml
yolo_model: "yolov8s-pose.pt"
imgsz: 640
conf_thres: 0.25
iou_thres: 0.45
keypoint_conf_thres: 0.5
```

## Project Structure

```
urfd_fall_yolo_pose/
├── configs/
│   └── default.yaml          # Configuration file
├── scripts/
│   ├── run_full_eval.py      # Main evaluation script
│   ├── eval_all.py           # Evaluation logic
│   ├── create_plots.py       # Visualization
│   ├── infer_sequence.py     # Single sequence inference
│   ├── prepare_urfd.py       # Dataset preparation
│   └── auto_tune.py          # Config optimization (grid search)
├── src/urfd/
│   ├── yolo_pose.py          # YOLOv8 wrapper
│   ├── features.py           # Feature extraction
│   ├── rules.py              # Detection rules
│   ├── tracking.py           # Multi-person tracking
│   ├── smoothing.py          # Temporal smoothing
│   ├── overlay.py            # Video annotation
│   ├── eval.py               # Metrics computation
│   ├── utils.py              # Helper functions
│   └── dataset.py            # Dataset utilities
├── data/
│   └── urfd_index.csv        # Sequence index
├── outputs/
│   ├── ADL/                  # ADL videos
│   ├── FALL/                 # Fall videos
│   ├── plots/                # Metrics plots
│   ├── predictions.csv       # Sequence predictions
│   └── metrics.json          # Evaluation metrics
├── runs/fixed/               # Final results
└── README.md                 # This file
```

## Debugging

### Check False Negatives (Missed Falls)

```bash
# View FN sequences in predictions.csv
python -c "import pandas as pd; df=pd.read_csv('outputs/predictions.csv'); print(df[df['gt_label']=='FALL'][df['pred_label']=='ADL'])"

# Common FN causes:
# - Slow falls (low dy_peak)
# - Occluded keypoints (feature_valid=False)
# - Quick recovery (min_confirm_duration not met)
# - Non-standard fall postures (angle/AR not met)
```

### Check False Positives (ADL → Fall)

```bash
# View FP sequences
python -c "import pandas as pd; df=pd.read_csv('outputs/predictions.csv'); print(df[df['gt_label']=='ADL'][df['pred_label']=='FALL'])"

# Common FP causes:
# - Sitting/lying ADL (high AR)
# - Bending motions (transient angle increase)
# - Fast downward motions (dy_peak triggered)
```

### Run Single Sequence

```bash
# Debug specific sequence with verbose output
python scripts/infer_sequence.py \
  --video_path data/raw/UR_Fall_Detection_Dataset/data/adl-10-cam0-rgb.avi \
  --gt_label ADL \
  --output_dir debug_output \
  --save_video
```

## Limitations & Future Work

**Current Limitations:**

1. **6 FN sequences:** Low-motion falls with poor pose quality cannot be detected by thresholds alone
2. **Threshold-based:** Not adaptive to scene context (lighting, camera angle, person size)
3. **Single-person:** No multi-person fall detection
4. **No temporal context:** Frame-level decisions without trajectory modeling

**Possible Improvements:**

1. **ML-based classifier:** Train LSTM/Transformer on pose sequences for temporal patterns
2. **Ensemble methods:** Combine rule-based + learned features
3. **Optical flow:** Add motion analysis for occluded cases
4. **Context-aware thresholds:** Adaptive to person height, camera distance
5. **Multi-scale analysis:** Consider both fine-grained and coarse temporal windows

## Citation

Dataset: UR Fall Detection Dataset

- University of Rzeszów, Poland
- 70 sequences (40 ADL, 30 FALL)
- Single-person indoor scenarios

Model: YOLOv8s-pose (Ultralytics)

- 17-keypoint COCO format
- Real-time pose estimation

## License

Educational use only. Dataset and model licenses apply.
