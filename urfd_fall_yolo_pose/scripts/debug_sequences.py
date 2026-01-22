"""Debug tool to trace sequences frame-by-frame and save diagnostic snapshots."""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urfd.dataset import load_sequence_frames
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.features import compute_frame_features
from src.urfd.smoothing import FallStateMachine
from src.urfd.tracking import PrimaryPersonTracker
from src.urfd.fallback_tracker import FallbackTracker
from src.urfd.utils import load_config, set_seed

def draw_debug_info(frame, features, state, score, frame_idx):
    """Draw debug information on frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw bbox if available
    if features["bbox"] is not None:
        x1, y1, x2, y2 = [int(v) for v in features["bbox"]]
        
        # Color based on state
        if state == "FALL_CONFIRMED":
            color = (0, 0, 255)  # Red
        elif state == "CANDIDATE":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)
    
    # Draw text overlay
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    texts = [
        f"Frame: {frame_idx}",
        f"State: {state}",
        f"Score: {score:.3f}",
    ]
    
    if features["bbox"] is not None:
        texts.extend([
            f"Height: {features['height']:.1f}px" if features['height'] else "Height: N/A",
            f"AR: {features['bbox_aspect_ratio']:.2f}" if features['bbox_aspect_ratio'] else "AR: N/A",
            f"Angle: {features['body_angle_deg']:.1f}deg" if features['body_angle_deg'] is not None else "Angle: N/A",
            f"dy: {features['dy']:.1f}px",
            f"dy_peak: {features['dy_peak']:.1f}px",
        ])
    
    for text in texts:
        cv2.putText(frame, text, (10, y_offset), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y_offset), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y_offset += 25
    
    return frame

def debug_sequence(seq_info, detector, config, output_dir):
    """Debug a single sequence with detailed frame-by-frame logging."""
    seq_name = seq_info["seq_name"]
    frame_dir = Path(seq_info["frame_dir"])
    gt_label = seq_info["gt_label"]
    
    print(f"\n{'='*80}")
    print(f"Debugging: {seq_name} (GT={gt_label})")
    print(f"{'='*80}")
    
    # Load frames
    frame_files = sorted(
        list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.jpg")),
        key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
    )
    
    if len(frame_files) == 0:
        print(f"Warning: No frames found for {seq_name}")
        return
    
    frames = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(img)
    
    if len(frames) == 0:
        print(f"Warning: Failed to load frames for {seq_name}")
        return
    
    # Initialize
    tracker = PrimaryPersonTracker(config)
    fallback_tracker = FallbackTracker(
        tracker_type=config.get("fallback_tracker_type", "kcf"),
        max_gap=config.get("fallback_track_max_gap", 6)
    )
    state_machine = FallStateMachine(config)
    
    # Process frames
    all_features = []
    all_states = []
    all_scores = []
    
    # Track important events
    first_candidate_frame = -1
    first_confirm_frame = -1
    max_dy_peak_frame = -1
    max_dy_peak_value = 0
    max_height_drop_frame = -1
    max_height_drop_value = 0
    
    # CSV log data
    log_rows = []
    
    for idx, frame in enumerate(frames):
        # Detect
        detections = detector.detect(frame)
        fallback_det = fallback_tracker.update(frame, detections)
        if fallback_det is not None:
            detections = [fallback_det]
        
        if len(detections) > 0 and fallback_det is None:
            if tracker.track_bbox is not None:
                fallback_tracker.initialize(frame, tracker.track_bbox)
        
        # Features
        features = compute_frame_features(
            detections,
            config["keypoint_conf_thres"],
            config["dy_window"],
            all_features,
            tracker=tracker,
            frame_idx=idx
        )
        all_features.append(features)
        
        # State
        state, score = state_machine.update(features)
        all_states.append(state)
        all_scores.append(score)
        
        # Track events
        if state == "CANDIDATE" and first_candidate_frame == -1:
            first_candidate_frame = idx
        if state == "FALL_CONFIRMED" and first_confirm_frame == -1:
            first_confirm_frame = idx
        
        # Track max dy_peak
        if features["dy_peak"] > max_dy_peak_value:
            max_dy_peak_value = features["dy_peak"]
            max_dy_peak_frame = idx
        
        # Track max height_drop
        if len(state_machine.height_history) > 0:
            height_drop = state_machine._compute_height_drop(features["height"])
            if height_drop > max_height_drop_value:
                max_height_drop_value = height_drop
                max_height_drop_frame = idx
        
        # Log row
        tracker_mode = "FALLBACK" if fallback_det is not None else "YOLO"
        det_conf = detections[features["primary_person_idx"]].get("conf", 0.0) if features["primary_person_idx"] >= 0 and len(detections) > 0 else 0.0
        
        log_rows.append({
            "frame": idx,
            "state": state,
            "score": score,
            "is_candidate": state in ["CANDIDATE", "FALL_CONFIRMED"],
            "is_lying": features["body_angle_deg"] >= config["confirm_angle_thres"] if features["body_angle_deg"] is not None else False,
            "dy": features["dy"],
            "dy_peak": features["dy_peak"],
            "height": features["height"] if features["height"] is not None else 0,
            "height_drop": height_drop if features["bbox"] is not None else 0,
            "body_angle_deg": features["body_angle_deg"] if features["body_angle_deg"] is not None else 0,
            "bbox_ar": features["bbox_aspect_ratio"] if features["bbox_aspect_ratio"] is not None else 0,
            "missing_streak": state_machine.missing_streak,
            "tracker_mode": tracker_mode,
            "det_conf": det_conf
        })
    
    # Save CSV log
    seq_output_dir = output_dir / seq_name
    seq_output_dir.mkdir(parents=True, exist_ok=True)
    
    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(seq_output_dir / "frame_log.csv", index=False)
    
    # Print summary
    pred_label = 1 if first_confirm_frame >= 0 else 0
    print(f"\nSummary:")
    print(f"  GT: {gt_label}, Pred: {pred_label}")
    print(f"  First CANDIDATE: {first_candidate_frame}")
    print(f"  First FALL_CONFIRMED: {first_confirm_frame}")
    print(f"  Max dy_peak: {max_dy_peak_value:.1f}px at frame {max_dy_peak_frame}")
    print(f"  Max height_drop: {max_height_drop_value:.3f} at frame {max_height_drop_frame}")
    
    # Save key snapshots
    snapshot_frames = []
    
    if first_candidate_frame >= 0:
        snapshot_frames.append(("first_candidate", first_candidate_frame))
    if first_confirm_frame >= 0:
        snapshot_frames.append(("first_confirm", first_confirm_frame))
    if max_dy_peak_frame >= 0:
        snapshot_frames.append(("max_dy_peak", max_dy_peak_frame))
    if max_height_drop_frame >= 0:
        snapshot_frames.append(("max_height_drop", max_height_drop_frame))
    
    # Add recovery frame (if state changes back from FALL_CONFIRMED)
    for i in range(len(all_states) - 1):
        if all_states[i] == "FALL_CONFIRMED" and all_states[i+1] != "FALL_CONFIRMED":
            snapshot_frames.append(("recovery", i+1))
            break
    
    print(f"\nSaving {len(snapshot_frames)} snapshots...")
    for name, frame_idx in snapshot_frames:
        if 0 <= frame_idx < len(frames):
            snapshot = draw_debug_info(
                frames[frame_idx],
                all_features[frame_idx],
                all_states[frame_idx],
                all_scores[frame_idx],
                frame_idx
            )
            out_path = seq_output_dir / f"{name}_f{frame_idx:04d}.png"
            cv2.imwrite(str(out_path), snapshot)
            print(f"  - {out_path.name}")
    
    print(f"\nDebug outputs saved to: {seq_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Debug specific sequences")
    parser.add_argument("--root", type=str, required=True,
                        help="Path to URFD raw data directory")
    parser.add_argument("--index", type=str, required=True,
                        help="Path to index CSV file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--sequences", type=str, nargs="+", required=True,
                        help="Sequence names to debug (e.g., adl-17 fall-19)")
    parser.add_argument("--output", type=str, default="debug_out",
                        help="Output directory for debug files")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config["seed"])
    
    # Load index
    df_index = pd.read_csv(args.index)
    
    # Filter sequences
    df_debug = df_index[df_index["seq_name"].isin(args.sequences)]
    
    if len(df_debug) == 0:
        print(f"Error: No matching sequences found for {args.sequences}")
        return
    
    print(f"Debugging {len(df_debug)} sequences: {list(df_debug['seq_name'])}")
    
    # Initialize detector
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
    
    # Debug each sequence
    output_dir = Path(args.output)
    for _, row in df_debug.iterrows():
        debug_sequence(row, detector, config, output_dir)

if __name__ == "__main__":
    main()
