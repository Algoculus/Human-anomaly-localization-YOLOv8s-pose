import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urfd.dataset import load_sequence_frames
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.features import compute_frame_features
from src.urfd.smoothing import FallStateMachine
from src.urfd.tracking import PrimaryPersonTracker
from src.urfd.fallback_tracker import FallbackTracker
from src.urfd.overlay import create_overlay_video
from src.urfd.utils import load_config, set_seed

def infer_sequence(seq_path, config_path):
    """Run inference on a single URFD sequence.
    
    Args:
        seq_path: Path to sequence folder
        config_path: Path to config YAML file
    """
    # Load config
    config = load_config(config_path)
    set_seed(config["seed"])
    
    seq_path = Path(seq_path)
    seq_name = seq_path.name
    
    print(f"Processing sequence: {seq_name}")
    
    # Load frames
    frames, frame_paths = load_sequence_frames(seq_path)
    if len(frames) == 0:
        print(f"Error: No frames found for {seq_name}")
        return
    
    print(f"Loaded {len(frames)} frames")
    
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
    
    # Initialize trackers
    tracker = PrimaryPersonTracker(config)
    fallback_tracker = FallbackTracker(
        tracker_type=config.get("fallback_tracker_type", "kcf"),
        max_gap=config.get("fallback_track_max_gap", 6)
    )
    
    # Initialize state machine
    state_machine = FallStateMachine(config)
    
    # Process each frame
    all_features = []
    all_states = []
    all_scores = []
    primary_indices = []
    
    for idx, frame in enumerate(frames):
        # Run YOLO pose detection
        detections = detector.detect(frame)
        
        # Apply fallback tracker if YOLO missed
        fallback_det = fallback_tracker.update(frame, detections)
        if fallback_det is not None:
            detections = [fallback_det]
        
        # Initialize fallback tracker when we have valid primary detection
        if len(detections) > 0 and fallback_det is None:
            if tracker.track_state is not None and tracker.track_state["bbox"] is not None:
                fallback_tracker.initialize(frame, tracker.track_state["bbox"])
        
        # Compute features with tracking
        features = compute_frame_features(
            detections,
            config["keypoint_conf_thres"],
            config["dy_window"],
            all_features,
            tracker=tracker,
            frame_idx=idx
        )
        
        all_features.append(features)
        primary_indices.append(features["primary_person_idx"])
        
        # Update state machine
        state, score = state_machine.update(features)
        all_states.append(state)
        all_scores.append(score)
    
    # Determine sequence-level prediction
    fall_confirmed = any(s == "FALL_CONFIRMED" for s in all_states)
    pred_label = 1 if fall_confirmed else 0
    first_confirm_frame = -1
    if fall_confirmed:
        first_confirm_frame = next(i for i, s in enumerate(all_states) if s == "FALL_CONFIRMED")
    
    max_score = max(all_scores) if all_scores else 0.0
    
    print(f"Prediction: {pred_label} (Fall={fall_confirmed})")
    print(f"First confirm frame: {first_confirm_frame}")
    print(f"Max score: {max_score:.3f}")
    
    # Create overlay video
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{seq_name}_overlay.mp4"
    
    print(f"Creating overlay video: {output_path}")
    create_overlay_video(
        frames,
        [f["detections"] for f in all_features],
        all_states,
        all_scores,
        output_path,
        config["output_fps"],
        primary_indices=primary_indices
    )
    
    print(f"Done! Video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single URFD sequence")
    parser.add_argument("--seq", type=str, required=True,
                        help="Path to sequence folder")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    
    args = parser.parse_args()
    infer_sequence(args.seq, args.config)

if __name__ == "__main__":
    main()
