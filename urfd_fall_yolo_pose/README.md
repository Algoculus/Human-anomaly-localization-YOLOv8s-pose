# URFD Fall Detection System

Rule-based fall detection using YOLOv8-pose with **Hypothesis→Verification** dual-tier architecture.

## Architecture Overview

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                    FALL DETECTION PIPELINE                      │
   └─────────────────────────────────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼────────────────────────────────┐
   │  YOLOv8-pose Detection → Multi-Person Tracking                   │
   └────────────────────────────────┬────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼────────────────────────────────┐
   │  Feature Extraction (per track)                                  │
   │  • Impact/Motion: dy_peak, dy_acc, impact_signature              │
   │  • Shape: height_drop_norm, width_increase, dAR                  │
   │  • Pose: hip_drop, angle_change, torso_compaction                │
   │  • Quality: keypoint_ratio, bbox_stability, border_penalty       │
   └────────────────────────────────┬────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼────────────────────────────────┐
   │  Dual-Tier State Machine                                         │
   │                                                                   │
   │  ┌─────────┐     Tier 1       ┌────────────┐                     │
   │  │ NORMAL  │ ─────trigger───▶ │ HYPOTHESIS │                     │
   │  └────┬────┘                  └──────┬─────┘                     │
   │       │                              │                           │
   │       │ ◀─── timeout/recovery ───────┤                           │
   │       │                              │                           │
   │       │       ┌────────────────┐     │ Tier 2                    │
   │       │ ◀───  │   VERIFYING    │ ◀───┘ (realtime mode)          │
   │       │       └───────┬────────┘                                 │
   │       │               │ confirm                                  │
   │       │               ▼                                          │
   │       │       ┌───────────────┐                                  │
   │       └────── │     FALL      │ ◀─── early decision (demo mode) │
   │  recovery     └───────────────┘                                  │
   └─────────────────────────────────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼────────────────────────────────┐
   │  Output: FallDecision                                            │
   │  • Label: FALL / NORMAL / UNCERTAIN                              │
   │  • Confidence: 0..1                                              │
   │  • Top Features: [(name, value, score), ...]                     │
   │  • FSM State: NORMAL / HYPOTHESIS / VERIFYING / FALL_CONFIRMED   │
   └─────────────────────────────────────────────────────────────────┘
```

## Detection Modes

### Demo Mode (`detection_mode: 'demo'`)
- **Use case**: Short video clips that may end right after the fall
- **Behavior**: Early FALL decision if high confidence detected within first few frames
- **Trigger**: `hypothesis.early_score_high` (default 0.65)

### Realtime Mode (`detection_mode: 'realtime'`)
- **Use case**: Webcam/live video with sufficient post-fall frames
- **Behavior**: Full verification with confirmation buffer
- **Confirmation**: `verification.confirm_frames` consecutive lying frames (default 8)

## Feature Categories

| Category | Feature | Description |
|----------|---------|-------------|
| **Impact** | `dy_peak` | Peak vertical velocity in window |
| | `dy_norm` | Normalized velocity (dy / baseline_h) |
| | `dy_acc` | Acceleration/jerk |
| | `impact_signature` | Combined impact pattern (spike + settle) |
| **Shape** | `height_drop_norm` | Height reduction vs baseline |
| | `width_increase` | Width increase vs baseline |
| | `dAR` | Frame-to-frame aspect ratio change |
| | `bbox_aspect_ratio` | Current W/H ratio |
| **Pose** | `hip_drop` | Change in hip midpoint Y |
| | `angle_change` | Delta from stable angle |
| | `torso_compaction` | Reduction in torso length |
| | `body_angle_deg` | Torso angle (reduced weight for cam0) |
| **Quality** | `keypoint_valid_ratio` | Fraction of confident keypoints |
| | `bbox_stability` | Inverse of bbox jitter |
| | `border_penalty` | Penalty when bbox touches edge |

## Configuration

### Key Parameters

```yaml
# Mode selection
detection_mode: 'realtime'  # or 'demo'
fps: 30

# Hypothesis (Tier 1)
hypothesis:
  early_score_thres: 0.35   # Trigger threshold
  early_score_high: 0.65    # Early FALL (demo mode)
  early_frames: 3           # Frames for early decision

# Verification (Tier 2)
verification:
  verify_score_thres: 0.55  # Verification threshold
  confirm_frames: 8         # Lying frames to confirm
  confirm_window: 30        # Timeout

# Feature weights
weights:
  impact: 0.35
  shape: 0.35
  pose: 0.30
```

### Threshold Tuning by FPS

| FPS | `confirm_frames` | `confirm_window` | `baseline_window` |
|-----|------------------|------------------|-------------------|
| 15  | 4                | 15               | 15                |
| 25  | 6                | 25               | 25                |
| 30  | 8                | 30               | 30                |
| 60  | 16               | 60               | 60                |

**Formula**: `frames = target_seconds × fps`

## Usage

### Evaluation on URFD Dataset

```bash
cd urfd_fall_yolo_pose
python scripts/eval_all.py \
    --root ../data \
    --index ../data/urfd_index.csv \
    --config configs/default.yaml
```

### Single Sequence Inference

```bash
python scripts/infer_sequence.py \
    --seq ../data/03_processed/fall-01-cam0 \
    --config configs/default.yaml
```

## Output

### FallDecision Object

```python
FallDecision(
    label='FALL',           # FALL, NORMAL, or UNCERTAIN
    confidence=0.87,        # Overall confidence
    fsm_state='FALL_CONFIRMED',
    top_features=[
        ('height_drop_norm', 0.35, 0.12),
        ('dy_peak', 15.2, 0.09),
        ('hip_drop', 28.5, 0.06)
    ],
    reason='Fall confirmed (lying=12 frames)'
)
```

### Video Overlay

Debug overlay shows:
- Bounding boxes (color = state)
- Skeleton keypoints
- Confidence bar
- Feature indicators (HD, DY, AR, IMP)
- Top contributing features

## Improvements Over Previous Version

1. **Reduced torso angle dependency** for cam0 (frontal falls)
2. **Dynamic baselines** for height/width (adapts to camera distance)
3. **UNCERTAIN label** for borderline cases
4. **Explainability** via top feature contributions
5. **Dual-tier detection** supporting both short clips and realtime
6. **Quality-aware scoring** penalizes unreliable keypoints/borders

## File Structure

```
urfd_fall_yolo_pose/
├── configs/
│   └── default.yaml       # Main configuration
├── scripts/
│   ├── eval_all.py        # Full dataset evaluation
│   ├── infer_sequence.py  # Single sequence inference
│   └── prepare_urfd.py    # Dataset preparation
└── src/urfd/
    ├── features.py        # Feature extraction + FeatureBuffer
    ├── rules.py           # FallScorer + weighted scoring
    ├── smoothing.py       # FallStateMachine (4-state FSM)
    ├── overlay.py         # Debug visualization
    ├── tracking.py        # Multi-person tracking
    └── yolo_pose.py       # YOLOv8-pose detector
```
