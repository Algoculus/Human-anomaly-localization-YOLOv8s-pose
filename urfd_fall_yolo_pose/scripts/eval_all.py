import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urfd.dataset import load_sequence_frames
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.features import compute_frame_features
from src.urfd.smoothing import FallStateMachine
from src.urfd.tracking import MultiPersonTracker
from src.urfd.overlay import create_overlay_video
from src.urfd.eval import compute_metrics, save_metrics, create_evaluation_plots, save_evaluation_summary
from src.urfd.utils import load_config, set_seed, validate_config
from prepare_urfd import prepare_urfd

def process_sequence(seq_info, detector, config, save_video=True):
    # Process a single sequence and return prediction results
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
    
    tracker = MultiPersonTracker(config)
    state_machines = {}
    histories = {}
    all_tracks_data = []
    sequence_max_scores = []
    
    for idx, frame in enumerate(frames):
        detections = detector.detect(frame)
        matches = tracker.update(detections, config["keypoint_conf_thres"], idx)
        
        frame_track_data = {}
        max_frame_score = 0.0
        
        active_tids = list(tracker.tracks.keys())
        
        for tid in active_tids:
            det = None
            if tid in matches:
                det = detections[matches[tid]]
            
            if tid not in state_machines:
                state_machines[tid] = FallStateMachine(config)
                histories[tid] = []
            
            image_size = (frames[0].shape[1], frames[0].shape[0]) if len(frames) > 0 else None
            features = compute_frame_features(
                det,
                config["keypoint_conf_thres"],
                config["dy_window"],
                histories[tid],
                track_id=tid,
                frame_idx=idx,
                image_size=image_size
            )
            
            histories[tid].append(features)
            state, score = state_machines[tid].update(features)
            
            if score > max_frame_score:
                max_frame_score = score
            
            frame_track_data[tid] = {
                "bbox": features["bbox"],
                "keypoints": det["keypoints"] if det else None,
                "state": state,
                "score": score
            }
        
        all_tracks_data.append(frame_track_data)
        sequence_max_scores.append(max_frame_score)
    
    fall_confirmed = False
    
    for tid, sm in state_machines.items():
        track_states = []
        for frame_data in all_tracks_data:
            if tid in frame_data:
                track_states.append(frame_data[tid]["state"])
            else:
                track_states.append("NONE")
        
        max_consecutive = 0
        current_consecutive = 0
        for s in track_states:
            if s == "FALL_CONFIRMED":
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        min_confirm = config.get("min_confirm_duration_frames", 3)
        if max_consecutive >= min_confirm:
            fall_confirmed = True
            break
    
    pred_label = 1 if fall_confirmed else 0
    max_score = max(sequence_max_scores) if sequence_max_scores else 0.0
    
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
        
        create_overlay_video(frames, all_tracks_data, output_path, config["output_fps"])
    
    return {
        "seq_name": seq_name,
        "gt_label": gt_label,
        "pred_label": pred_label,
        "pred_score": max_score
    }

def eval_all(root_dir, index_path, config_path, no_videos=False):
    # Evaluate all URFD sequences
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
    
    index_path = Path(index_path)
    if not index_path.exists():
        print(f"Index file {index_path} not found. Preparing dataset...")
        prepare_urfd(root_dir, index_path)
    
    df_index = pd.read_csv(index_path)
    print(f"Loaded {len(df_index)} sequences from index")
    
    print("Initializing YOLOv8s-pose detector...")
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
    
    results = []
    for idx, row in df_index.iterrows():
        print(f"\nProcessing {idx+1}/{len(df_index)}: {row['seq_name']}")
        res = process_sequence(row, detector, config, save_video=(not no_videos))
        if res is not None:
            results.append(res)
            print(f"  GT={res['gt_label']}, Pred={res['pred_label']}, Score={res['pred_score']:.3f}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    pred_path = output_dir / "predictions.csv"
    df_results.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")
    
    gt_labels = df_results["gt_label"].values
    pred_labels = df_results["pred_label"].values
    
    metrics = compute_metrics(gt_labels, pred_labels)
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")
    
    print("\nCreating evaluation plots...")
    plot_paths = create_evaluation_plots(metrics, output_dir)
    
    summary_path = output_dir / "eval_summary.json"
    save_evaluation_summary(metrics, config, plot_paths, summary_path)
    print(f"Summary saved to {summary_path}")
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--no_videos", action="store_true")
    
    args = parser.parse_args()
    eval_all(args.root, args.index, args.config, args.no_videos)

if __name__ == "__main__":
    main()
