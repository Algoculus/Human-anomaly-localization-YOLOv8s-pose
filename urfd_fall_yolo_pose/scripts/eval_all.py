import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urfd.dataset import load_sequence_frames
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.features import compute_frame_features
from src.urfd.smoothing import FallStateMachine
from src.urfd.overlay import create_overlay_video
from src.urfd.eval import compute_metrics, save_metrics, create_evaluation_plots, save_evaluation_summary
from src.urfd.utils import load_config, set_seed

# Import prepare logic
from prepare_urfd import prepare_urfd

def process_sequence(seq_info, detector, config, save_video=True):
    """Process a single sequence and return prediction results."""
    seq_name = seq_info["seq_name"]
    frame_dir = Path(seq_info["frame_dir"])
    gt_label = seq_info["gt_label"]
    
    # Load frames from frame_dir directly
    frame_files = sorted(
        list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.jpg")),
        key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
    )
    
    if len(frame_files) == 0:
        print(f"Warning: No frames found for {seq_name}")
        return None
    
    import cv2
    frames = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(img)
    
    if len(frames) == 0:
        print(f"Warning: Failed to load frames for {seq_name}")
        return None
    
    # Initialize state machine
    state_machine = FallStateMachine(config)
    
    # Process each frame
    all_features = []
    all_states = []
    all_scores = []
    
    for idx, frame in enumerate(frames):
        # Run YOLO pose detection
        detections = detector.detect(frame)
        
        # Compute features
        features = compute_frame_features(
            detections,
            config["keypoint_conf_thres"],
            config["dy_window"],
            all_features
        )
        
        all_features.append(features)
        
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
    
    # Create overlay video
    if save_video:
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{seq_name}_overlay.mp4"
        
        create_overlay_video(
            frames,
            [f["detections"] for f in all_features],
            all_states,
            all_scores,
            output_path,
            config["output_fps"]
        )
    
    return {
        "seq_name": seq_name,
        "gt_label": gt_label,
        "pred_label": pred_label,
        "pred_score": max_score,
        "first_confirm_frame": first_confirm_frame
    }

def eval_all(root_dir, index_path, config_path, no_videos=False):
    """Evaluate all URFD sequences.
    
    Args:
        root_dir: Path to URFD raw data directory
        index_path: Path to index CSV file
        config_path: Path to config YAML file
        no_videos: If True, skip video generation (default: False, generate all videos)
    """
    # Load config
    config = load_config(config_path)
    set_seed(config["seed"])
    
    # Prepare output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Delete old overlay videos
    print("Cleaning old overlay videos...")
    old_videos = list(output_dir.glob("*_overlay.mp4"))
    for video in old_videos:
        video.unlink()
    print(f"Deleted {len(old_videos)} old videos")
    
    # Check if index exists, if not prepare it
    index_path = Path(index_path)
    if not index_path.exists():
        print(f"Index file {index_path} not found. Preparing dataset...")
        prepare_urfd(root_dir, index_path)
    
    # Load index
    df_index = pd.read_csv(index_path)
    print(f"Loaded {len(df_index)} sequences from index")
    
    # Initialize detector once for all sequences
    print("Initializing YOLOv8s-pose detector...")
    detector = YOLOPoseDetector(
        model_path=config["yolo_model"],
        imgsz=config["imgsz"],
        conf_thres=config["conf_thres"],
        iou_thres=config["iou_thres"]
    )
    
    # Process all sequences
    results = []
    for idx, row in df_index.iterrows():
        print(f"\nProcessing {idx+1}/{len(df_index)}: {row['seq_name']}")
        result = process_sequence(row, detector, config, save_video=(not no_videos))
        if result is not None:
            results.append(result)
            print(f"  GT={result['gt_label']}, Pred={result['pred_label']}, Score={result['pred_score']:.3f}")
    
    # Save predictions
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    predictions_path = output_dir / "predictions.csv"
    df_results.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to {predictions_path}")
    
    # Compute metrics
    gt_labels = df_results["gt_label"].values
    pred_labels = df_results["pred_label"].values
    
    metrics = compute_metrics(gt_labels, pred_labels)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")
    
    # Create evaluation plots
    print("\nCreating evaluation plots...")
    plot_paths = create_evaluation_plots(metrics, output_dir)
    print(f"Plots saved:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")
    
    # Save evaluation summary
    summary_path = output_dir / "eval_summary.json"
    save_evaluation_summary(metrics, config, plot_paths, summary_path)
    print(f"Evaluation summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Total sequences: {len(results)}")
    print(f"  ADL: {sum(gt_labels == 0)}")
    print(f"  Fall: {sum(gt_labels == 1)}")
    print()
    print("Confusion Matrix:")
    print(f"  TP (Fall->Fall): {metrics['confusion_matrix']['TP']}")
    print(f"  TN (ADL->ADL): {metrics['confusion_matrix']['TN']}")
    print(f"  FP (ADL->Fall): {metrics['confusion_matrix']['FP']}")
    print(f"  FN (Fall->ADL): {metrics['confusion_matrix']['FN']}")
    print()
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate all URFD sequences")
    parser.add_argument("--root", type=str, required=True,
                        help="Path to URFD raw data directory")
    parser.add_argument("--index", type=str, required=True,
                        help="Path to index CSV file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--no_videos", action="store_true",
                        help="Skip overlay video generation (default: generate all videos)")
    
    args = parser.parse_args()
    eval_all(args.root, args.index, args.config, args.no_videos)

if __name__ == "__main__":
    main()
