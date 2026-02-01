import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urfd.dataset import load_sequence_frames
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.features import compute_frame_features, FeatureBuffer
from src.urfd.smoothing import FallStateMachine, log_decision
from src.urfd.tracking import MultiPersonTracker
from src.urfd.overlay import create_overlay_video
from src.urfd.eval import compute_metrics, save_metrics, create_evaluation_plots, save_evaluation_summary
from src.urfd.utils import load_config, set_seed, validate_config, get_depth_path_from_rgb, load_depth_map


def process_sequence(seq_info, detector, config, save_video=True):
    """
    Process a single sequence and return prediction results.
    
    Uses enhanced FallDecision output with confidence and top features.
    """
    seq_name = seq_info["seq_name"]
    frame_dir = Path(seq_info["frame_dir"])
    gt_label = seq_info["gt_label"]
    
    frame_files = sorted(
        list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.jpg")),
        key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
    )
    
    if len(frame_files) == 0:
        print(f"Warning: No frames found for {seq_name}")
        return None
    
    frames = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(img)
    
    if len(frames) == 0:
        print(f"Warning: Failed to load frames for {seq_name}")
        return None
    
    # Initialize tracker and per-track state machines
    tracker = MultiPersonTracker(config)
    state_machines = {}
    histories = {}
    feature_buffers = {}  # New: per-track feature buffers
    all_tracks_data = []
    sequence_max_scores = []
    all_decisions = []  # Store decisions for logging
    
    image_size = (frames[0].shape[1], frames[0].shape[0])
    
    for idx, frame in enumerate(frames):
        # Load associated depth map (Strategy 6)
        frame_path = str(frame_files[idx])
        depth_path = get_depth_path_from_rgb(frame_path)
        depth_map = load_depth_map(depth_path)
        
        detections = detector.detect(frame)
        matches = tracker.update(detections, config["keypoint_conf_thres"], idx)
        
        frame_track_data = {}
        max_frame_score = 0.0
        frame_decision = None
        
        active_tids = list(tracker.tracks.keys())
        
        for tid in active_tids:
            det = None
            if tid in matches:
                det = detections[matches[tid]]
            
            # Initialize per-track state machine and buffer
            if tid not in state_machines:
                state_machines[tid] = FallStateMachine(config)
                feature_buffers[tid] = FeatureBuffer(
                    max_size=config.get("baseline_window", 30) * 2,
                    stable_motion_thres=config.get("stable_motion_thres", 2.0)
                )
            
            # Compute features (simplified API)
            features = compute_frame_features(
                det,
                config,
                feature_buffers[tid],
                track_id=tid,
                frame_idx=idx,
                image_size=image_size,
                depth_map=depth_map
            )
            
            feature_buffers[tid].add(features)
            
            # Update state machine with v2 API
            decision = state_machines[tid].update_v2(features, track_id=tid, frame_idx=idx)
            
            if decision.confidence > max_frame_score:
                max_frame_score = decision.confidence
                frame_decision = decision
            
            frame_track_data[tid] = {
                "bbox": features.bbox,
                "keypoints": det["keypoints"] if det else None,
                "state": decision.fsm_state,
                "score": decision.confidence,
                "label": decision.label,
                "top_features": decision.top_features,
                "features": features  # Include for debug overlay
            }
        
        all_tracks_data.append(frame_track_data)
        sequence_max_scores.append(max_frame_score)
        if frame_decision:
            all_decisions.append(frame_decision)
    
    # Determine sequence-level prediction
    fall_confirmed = False
    fall_label_count = 0
    uncertain_count = 0
    
    for tid, sm in state_machines.items():
        track_states = []
        for frame_data in all_tracks_data:
            if tid in frame_data:
                track_states.append(frame_data[tid]["state"])
            else:
                track_states.append("NONE")
        
        # Count consecutive FALL states
        max_consecutive = 0
        current_consecutive = 0
        for s in track_states:
            if "FALL" in s:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        min_confirm = config.get("min_confirm_duration_frames", 2)
        if max_consecutive >= min_confirm:
            fall_confirmed = True
            break
    
    # Count labels for uncertain handling
    for frame_data in all_tracks_data:
        for tid, track_info in frame_data.items():
            label = track_info.get("label", "NORMAL")
            if label == "FALL":
                fall_label_count += 1
            elif label == "UNCERTAIN":
                uncertain_count += 1
    
    pred_label = 1 if fall_confirmed else 0
    max_score = max(sequence_max_scores) if sequence_max_scores else 0.0
    
    # Determine output label
    if fall_confirmed:
        output_label = "FALL"
    elif uncertain_count > len(all_tracks_data) * 0.3:
        output_label = "UNCERTAIN"
    else:
        output_label = "NORMAL"
    
    # Log summary
    print(f"  GT={gt_label}, Pred={pred_label} ({output_label}), Score={max_score:.3f}, "
          f"FALL_frames={fall_label_count}, UNCERTAIN_frames={uncertain_count}")
    
    # Save video
    if save_video:
        output_base = Path(config["output_dir"])
        cam_name = "cam0"
        if "cam1" in seq_name:
            cam_name = "cam1"
        
        if gt_label == 0:
            output_dir = output_base / "videos" / "ADL" / cam_name
        else:
            output_dir = output_base / "videos" / "FALL" / cam_name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{seq_name}_overlay.mp4"
        
        debug_mode = config.get("debug_overlay", True)
        create_overlay_video(frames, all_tracks_data, output_path, config["output_fps"],
                            debug_mode=debug_mode)
    
    return {
        "seq_name": seq_name,
        "gt_label": gt_label,
        "pred_label": pred_label,
        "pred_score": max_score,
        "output_label": output_label,
        "fall_frames": fall_label_count,
        "uncertain_frames": uncertain_count
    }


def eval_all(root_dir, index_path, config_path, no_videos=False):
    """
    Evaluate all URFD sequences with enhanced logging.
    """
    config = load_config(config_path)
    set_seed(config["seed"])
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Cleaning old outputs...")
    videos_dir = output_dir / "videos"
    plots_dir = output_dir / "plots"
    
    old_videos = []
    if videos_dir.exists():
        old_videos.extend(list(videos_dir.glob("**/*_overlay.mp4")))
    for video in old_videos:
        video.unlink()
        
    old_plots = []
    if plots_dir.exists():
        old_plots.extend(list(plots_dir.glob("*.png")))
    for plot in old_plots:
        plot.unlink()
        
    print(f"Deleted {len(old_videos)} old videos, {len(old_plots)} old plots")
    
    # Load index
    index_path = Path(index_path)
    if not index_path.exists():
        print(f"Error: Index file {index_path} not found.")
        print(f"Please ensure the index CSV exists at: {index_path}")
        sys.exit(1)
    
    df_index = pd.read_csv(index_path)
    print(f"Loaded {len(df_index)} sequences from index")
    
    # Print config summary
    print(f"\nConfiguration:")
    print(f"  Mode: {config.get('detection_mode', 'realtime')}")
    print(f"  Hypothesis threshold: {config.get('hypothesis', {}).get('early_score_thres', 0.35)}")
    print(f"  Verification threshold: {config.get('verification', {}).get('verify_score_thres', 0.55)}")
    print(f"  Confirm frames: {config.get('verification', {}).get('confirm_frames', 8)}")
    
    # Initialize detector
    print("\nInitializing YOLOv8s-pose detector...")
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
    
    # Process sequences
    results = []
    cam0_fall_results = []
    cam1_fall_results = []
    adl_results = []
    
    for idx, row in df_index.iterrows():
        # if row['seq_name'] not in target_seqs:
        #     continue
            
        print(f"\nProcessing {idx+1}/{len(df_index)}: {row['seq_name']}")
        res = process_sequence(row, detector, config, save_video=(not no_videos))
        if res is not None:
            results.append(res)
            
            # Categorize
            if res["gt_label"] == 1:
                if "cam0" in res["seq_name"]:
                    cam0_fall_results.append(res)
                else:
                    cam1_fall_results.append(res)
            else:
                adl_results.append(res)
    
    # Save predictions
    df_results = pd.DataFrame(results)
    pred_path = output_dir / "predictions.csv"
    df_results.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")
    
    # Compute metrics
    gt_labels = df_results["gt_label"].values
    pred_labels = df_results["pred_label"].values
    
    metrics = compute_metrics(gt_labels, pred_labels)
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    
    # Per-category metrics
    if cam0_fall_results:
        cam0_recall = sum(1 for r in cam0_fall_results if r["pred_label"] == 1) / len(cam0_fall_results)
        metrics["cam0_fall_recall"] = cam0_recall
    
    if cam1_fall_results:
        cam1_recall = sum(1 for r in cam1_fall_results if r["pred_label"] == 1) / len(cam1_fall_results)
        metrics["cam1_fall_recall"] = cam1_recall
    
    if adl_results:
        adl_specificity = sum(1 for r in adl_results if r["pred_label"] == 0) / len(adl_results)
        metrics["adl_specificity"] = adl_specificity
    
    # Update metrics file
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")
    
    # Create plots
    print("\nCreating evaluation plots...")
    plot_paths = create_evaluation_plots(metrics, output_dir)
    
    summary_path = output_dir / "eval_summary.json"
    save_evaluation_summary(metrics, config, plot_paths, summary_path)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Overall Precision:   {metrics['precision']:.4f}")
    print(f"Overall Recall:      {metrics['recall']:.4f}")
    print(f"Overall F1-Score:    {metrics['f1_score']:.4f}")
    print("-"*60)
    if "cam0_fall_recall" in metrics:
        print(f"CAM0 Fall Recall:    {metrics['cam0_fall_recall']:.4f}")
    if "cam1_fall_recall" in metrics:
        print(f"CAM1 Fall Recall:    {metrics['cam1_fall_recall']:.4f}")
    if "adl_specificity" in metrics:
        print(f"ADL Specificity:     {metrics['adl_specificity']:.4f}")
    print("="*60)
    
    # Print confusion matrix
    cm = metrics["confusion_matrix"]
    print(f"\nConfusion Matrix:")
    print(f"  TP={cm['TP']} (Falls detected)")
    print(f"  FN={cm['FN']} (Falls missed)")
    print(f"  FP={cm['FP']} (False alarms)")
    print(f"  TN={cm['TN']} (ADL correct)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fall detection on URFD dataset")
    parser.add_argument("--root", type=str, required=True, help="Root directory of URFD dataset")
    parser.add_argument("--index", type=str, required=True, help="Path to index CSV file")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--no_videos", action="store_true", help="Skip video generation")
    
    args = parser.parse_args()
    eval_all(args.root, args.index, args.config, args.no_videos)


if __name__ == "__main__":
    main()
