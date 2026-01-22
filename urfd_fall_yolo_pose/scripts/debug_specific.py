"""Debug specific sequences by processing and analyzing them."""
import sys
from pathlib import Path
import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.tracking import PrimaryPersonTracker
from src.urfd.fallback_tracker import FallbackTracker
from src.urfd.smoothing import FallStateMachine
from src.urfd.features import compute_frame_features
from src.urfd.utils import load_config

# Target sequences for debugging
DEBUG_SEQUENCES = ["adl-17", "adl-21", "adl-34", "adl-35", "fall-19", "fall-23"]

config = load_config("configs/default.yaml")
df_index = pd.read_csv("d:/Workspace for Learning/My Projects/UTH Projects/Human-anomaly-localization-YOLOv8s-pose/data/urfd_index.csv")

print("[DEBUG] Initializing YOLOv8s-pose detector...")
detector = YOLOPoseDetector(
    model_path=config["yolo_model"],
    imgsz=config["imgsz"],
    conf_thres=config["conf_thres"],
    iou_thres=config["iou_thres"],
    preprocess_lowlight=config.get("preprocess_lowlight", False),
    gamma=config.get("gamma", 1.3),
    clahe_clip=config.get("clahe_clip", 2.0),
    clahe_grid=config.get("clahe_grid", 8)
)

for seq_name in DEBUG_SEQUENCES:
    print(f"\n{'='*80}")
    print(f"[DEBUG] Processing {seq_name}")
    print('='*80)
    
    seq_info = df_index[df_index["seq_name"] == seq_name].iloc[0]
    frame_dir = Path(seq_info["frame_dir"])
    gt_label = seq_info["gt_label"]
    
    print(f"Ground Truth: {'FALL' if gt_label == 1 else 'ADL'}")
    print(f"Frame Dir: {frame_dir}")
    
    # Load frames
    frame_files = sorted(
        list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.jpg")),
        key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
    )
    
    if len(frame_files) == 0:
        print(f"[ERROR] No frames found!")
        continue
    
    print(f"Total Frames: {len(frame_files)}")
    
    frames = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(img)
    
    # Initialize
    tracker = PrimaryPersonTracker(config)
    fallback_tracker = FallbackTracker(
        tracker_type=config.get("fallback_tracker_type", "kcf"),
        max_gap=config.get("fallback_track_max_gap", 6)
    )
    state_machine = FallStateMachine(config)
    
    all_features = []
    all_states = []
    all_scores = []
    
    # Process
    for idx, frame in enumerate(frames):
        detections = detector.detect(frame)
        
        fallback_det = fallback_tracker.update(frame, detections)
        if fallback_det is not None:
            detections = [fallback_det]
        
        if len(detections) > 0 and fallback_det is None:
            if tracker.track_bbox is not None:
                fallback_tracker.initialize(frame, tracker.track_bbox)
        
        features = compute_frame_features(
            detections,
            config["keypoint_conf_thres"],
            config["dy_window"],
            all_features,
            tracker=tracker,
            frame_idx=idx
        )
        
        all_features.append(features)
        state, score = state_machine.update(features)
        all_states.append(state)
        all_scores.append(score)
    
    # Analyze results
    fall_confirmed_frames = [i for i, s in enumerate(all_states) if s == "FALL_CONFIRMED"]
    candidate_frames = [i for i, s in enumerate(all_states) if s == "CANDIDATE"]
    
    # Find first FALL_CONFIRMED
    first_confirm_frame = fall_confirmed_frames[0] if fall_confirmed_frames else -1
    
    # Count consecutive FALL_CONFIRMED at any point
    max_consecutive_fall = 0
    current_consecutive = 0
    for s in all_states:
        if s == "FALL_CONFIRMED":
            current_consecutive += 1
            max_consecutive_fall = max(max_consecutive_fall, current_consecutive)
        else:
            current_consecutive = 0
    
    pred_label = 1 if max_consecutive_fall >= config["min_confirm_duration_frames"] else 0
    pred_str = "FALL" if pred_label == 1 else "ADL"
    gt_str = "FALL" if gt_label == 1 else "ADL"
    
    status = "CORRECT" if pred_label == gt_label else "ERROR"
    if status == "ERROR":
        if gt_label == 1:
            status += " (FALSE NEGATIVE)"
        else:
            status += " (FALSE POSITIVE)"
    
    print(f"\n[RESULT] Prediction: {pred_str} | Ground Truth: {gt_str} | {status}")
    print(f"[STATS] CANDIDATE frames: {len(candidate_frames)}")
    print(f"[STATS] FALL_CONFIRMED frames: {len(fall_confirmed_frames)}")
    print(f"[STATS] Max consecutive FALL_CONFIRMED: {max_consecutive_fall} (threshold: {config['min_confirm_duration_frames']})")
    print(f"[STATS] First FALL_CONFIRMED frame: {first_confirm_frame}")
    
    if fall_confirmed_frames:
        print(f"[DETAIL] FALL_CONFIRMED frame indices: {fall_confirmed_frames[:20]}{'...' if len(fall_confirmed_frames) > 20 else ''}")
    
    # Key features at critical frames
    if status != "CORRECT":
        print(f"\n[ANALYSIS] Why did this fail?")
        
        if gt_label == 1 and pred_label == 0:
            # False Negative - should detect fall but didn't
            print("FALSE NEGATIVE: Fall not detected")
            print("Possible causes:")
            print("  - dy_peak too low (not enough motion)")
            print("  - angle/AR thresholds not met")
            print("  - Detection missing at critical frame")
            
            # Find frames with highest score
            max_score_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
            print(f"\nMax score frame: {max_score_idx} (score={all_scores[max_score_idx]:.3f})")
            feat = all_features[max_score_idx]
            print(f"  angle: {feat.get('angle', 'N/A')}")
            print(f"  ar: {feat.get('ar', 'N/A')}")
            print(f"  dy_peak: {feat.get('dy_peak', 'N/A')}")
            print(f"  height_drop: {feat.get('height_drop', 'N/A')}")
            
        elif gt_label == 0 and pred_label == 1:
            # False Positive - detected fall but shouldn't
            print("FALSE POSITIVE: False alarm")
            print("Possible causes:")
            print("  - Person bending/crouching (angle too low)")
            print("  - Quick motion (high dy_peak)")
            print("  - Sitting down quickly")
            
            # Analyze FALL_CONFIRMED frames
            print(f"\nFALL_CONFIRMED frames analysis:")
            for fc_idx in fall_confirmed_frames[:5]:
                feat = all_features[fc_idx]
                print(f"  Frame {fc_idx}: angle={feat.get('angle', 'N/A'):.1f}, ar={feat.get('ar', 'N/A'):.2f}, dy_peak={feat.get('dy_peak', 'N/A'):.1f}")

print(f"\n{'='*80}")
print("[DEBUG] All sequences processed")
print('='*80)
