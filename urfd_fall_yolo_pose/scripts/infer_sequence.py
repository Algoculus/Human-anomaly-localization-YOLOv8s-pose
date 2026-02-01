import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urfd.dataset import load_sequence_frames
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.features import compute_frame_features, FeatureBuffer
from src.urfd.smoothing import FallStateMachine
from src.urfd.tracking import MultiPersonTracker
from src.urfd.overlay import create_overlay_video
from src.urfd.utils import load_config, set_seed

def infer_sequence(seq_path, config_path):
    """
    Run inference on a single URFD sequence.
    
    Args:
        seq_path: Path to sequence folder
        config_path: Path to config YAML file
    """
    config = load_config(config_path)
    set_seed(config["seed"])
    
    seq_path = Path(seq_path)
    seq_name = seq_path.name
    
    print(f"Processing sequence: {seq_name}")
    
    frames, frame_paths = load_sequence_frames(seq_path)
    if len(frames) == 0:
        print(f"Error: No frames found for {seq_name}")
        return
    
    print(f"Loaded {len(frames)} frames")
    
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
    
    tracker = MultiPersonTracker(config)
    state_machines = {}
    feature_buffers = {}
    all_tracks_data = []
    all_scores = []
    
    image_size = (frames[0].shape[1], frames[0].shape[0]) if len(frames) > 0 else None
    
    for idx, frame in enumerate(frames):
        detections = detector.detect(frame)
        matches = tracker.update(detections, config["keypoint_conf_thres"], idx)
        
        frame_track_data = {}
        frame_max_score = 0.0
        
        for tid in list(tracker.tracks.keys()):
            det = detections[matches[tid]] if tid in matches else None
            
            if tid not in state_machines:
                state_machines[tid] = FallStateMachine(config)
                feature_buffers[tid] = FeatureBuffer(
                    max_size=config.get("baseline_window", 30) * 2,
                    stable_motion_thres=config.get("stable_motion_thres", 2.0)
                )
            
            features = compute_frame_features(
                det,
                config,
                feature_buffers[tid],
                track_id=tid,
                frame_idx=idx,
                image_size=image_size
            )
            feature_buffers[tid].add(features)
            decision = state_machines[tid].update_v2(features, track_id=tid, frame_idx=idx)
            
            frame_max_score = max(frame_max_score, decision.confidence)
            
            frame_track_data[tid] = {
                \"bbox\": features.bbox,
                \"keypoints\": det[\"keypoints\"] if det else None,
                \"state\": decision.fsm_state,
                \"score\": decision.confidence,
                \"label\": decision.label,
                \"top_features\": decision.top_features,
                \"features\": features
            }
        
        all_tracks_data.append(frame_track_data)
        all_scores.append(frame_max_score)
    
    # Determine sequence-level prediction
    fall_confirmed = False
    for tid, sm in state_machines.items():
        track_states = [
            frame_data.get(tid, {}).get(\"state\", \"NONE\")
            for frame_data in all_tracks_data
        ]
        max_consecutive = 0
        current_consecutive = 0
        for s in track_states:
            if \"FALL\" in s:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        min_confirm_duration = config.get(\"min_confirm_duration_frames\", 3)
        if max_consecutive >= min_confirm_duration:
            fall_confirmed = True
            break
    
    pred_label = 1 if fall_confirmed else 0
    first_confirm_frame = -1
    if fall_confirmed:
        for i, frame_data in enumerate(all_tracks_data):
            if any(\"FALL\" in t.get(\"state\", \"\") for t in frame_data.values()):
                first_confirm_frame = i
                break
    
    max_score = max(all_scores) if all_scores else 0.0
    
    print(f\"Prediction: {pred_label} (Fall={fall_confirmed})\")
    print(f\"First confirm frame: {first_confirm_frame}\")
    print(f\"Max score: {max_score:.3f}\")
    
    output_dir = Path(config[\"output_dir\"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f\"{seq_name}_overlay.mp4\"
    
    print(f\"Creating overlay video: {output_path}\")
    create_overlay_video(
        frames,
        all_tracks_data,
        output_path,
        config[\"output_fps\"],
        debug_mode=config.get(\"debug_overlay\", True)
    )
    
    print(f\"Done! Video saved to {output_path}\")

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single URFD sequence")
    parser.add_argument("--seq", type=str, required=True, help="Path to sequence folder")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    
    args = parser.parse_args()
    infer_sequence(args.seq, args.config)

if __name__ == "__main__":
    main()
