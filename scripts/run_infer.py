"""
Batch Inference Script

Runs the Fall Detection System on URFall sequences.
Generates:
1. Annotated Videos (overlays)
2. CSV Logs (frame-by-frame scores and states)
"""

import argparse
import sys
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import get_config
from src.data.urfall_loader import load_all_sequences, SequenceData
from src.infer_pose import PoseInferencePipeline
from src.rules.features import FeatureExtractor
from src.rules.scoring import FallScorer
from src.rules.state_machine import FallStateMachine, StateMachineContext
from src.utils.visualization import Visualizer

def process_sequence(seq: SequenceData, config, output_dir: Path, save_video: bool = True):
    """
    Process a single sequence.
    """
    sequence_id = seq.sequence_id
    logger.info(f"Processing {sequence_id}...")
    
    # Initialize Pipeline Components
    pipeline = PoseInferencePipeline(config)
    feature_extractor = FeatureExtractor(config)
    scorer = FallScorer(config)
    fsm = FallStateMachine(config)
    context = StateMachineContext()
    visualizer = Visualizer(config)
    
    # Prepare Output
    results = []
    
    # Determine Output Directory
    if seq.sequence_type == "fall":
        video_out_dir = config.paths.output_videos_fall
    else:
        video_out_dir = config.paths.output_videos_adl
    video_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Video Writer
    video_writer = None
    if save_video and seq.cam0 and seq.cam0.frames:
        video_path = video_out_dir / f"{sequence_id}_output.mp4"
        sample_img = seq.cam0.frames[0].load_rgb()
        if sample_img is not None:
            h, w = sample_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    
    # Iterate Frames
    if not seq.cam0:
        logger.warning(f"No cam0 data for {sequence_id}")
        return
        
    for frame in tqdm(seq.cam0.frames, desc=f"Seq {sequence_id}", leave=False):
        # 1. Load Data
        image = frame.load_rgb()
        if image is None:
            continue
            
        # 2. Run Pose Inference
        pipe_out = pipeline.process_frame(image)
        poses = pipe_out['poses']
        primary_pose = None
        if poses:
            primary_pose = poses[0] # Assume primary person is first (highest conf)
            
        # 3. Extract Features (Fusion)
        # Get depth features from ground truth if available
        depth_feats = frame.depth_features
        # Get accelerometer data
        acc_data = {'sv_total': frame.acc_sv_total} if frame.acc_sv_total else None
        
        features = feature_extractor.extract(
            primary_pose, 
            image.shape[:2],
            depth_features=depth_feats,
            acc_data=acc_data
        )
        
        # 4. Compute Scores
        scores = scorer.calculate(features)
        
        # 5. Update State (and velocity for behavior label)
        context.recent_velocity = features.velocity_y
        state = fsm.update(scores, context)
        
        # 6. Record Result
        result_row = {
            'sequence': sequence_id,
            'frame': frame.frame_number,
            'time_ms': frame.timestamp_ms,
            'gt_label': frame.label,
            'pred_state': state.value,
            'behavior': context.behavior_label,
            'fall_score': scores.total_score,
            'drop_score': scores.sudden_drop_score,
            'prone_score': scores.prone_score,
            'impact_score': scores.impact_score,
            'sustained_lying_score': scores.sustained_lying_score,
            'body_orient': features.body_orientation,
            'velocity_y': features.velocity_y,
            'aspect_ratio': features.bbox_aspect_ratio,
        }
        results.append(result_row)
        
        # 7. Visualization
        if video_writer:
            viz_img = visualizer.draw_frame(
                image.copy(),
                frame.frame_number,
                state,
                scores.total_score,
                behavior_label=context.behavior_label,
                scores_dict={
                    'Fall Score': scores.total_score,
                    'Drop Score': scores.sudden_drop_score,
                    'Prone Score': scores.prone_score,
                    'Impact': scores.impact_score,
                    'Sustained': scores.sustained_lying_score,
                    'Drop Vel': features.normalized_drop_velocity,
                    'Orient(°)': features.body_orientation,
                    'AR (H/W)': features.bbox_aspect_ratio,
                },
                poses=poses
            )
            video_writer.write(viz_img)
            
        # Clean up
        frame.clear_cache()
    
    # Close video
    if video_writer:
        video_writer.release()
        
    # Save CSV Results (to organized subdirectory)
    if seq.sequence_type == "fall":
        csv_out_dir = config.paths.output_results_fall
    else:
        csv_out_dir = config.paths.output_results_adl
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = csv_out_dir / f"{sequence_id}_results.csv"
    df.to_csv(csv_path, index=False)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Run Fall Detection Inference")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of sequences")
    parser.add_argument("--output", type=str, default="output/results", help="Output directory")
    parser.add_argument("--no-video", action="store_true", help="Disable video generation")
    args = parser.parse_args()
    
    config = get_config()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Dataset
    logger.info("Loading dataset...")
    # Use cache to speed up loading metadata
    dataset = load_all_sequences(config) 
    # Note: load_all_sequences uses URFallDataset which we updated to default use_cache=True, 
    # but load_all_sequences helper might need update if we want to be explicit.
    # Let's rely on the default we just set in urfall_loader.py: load_all(..., use_cache=True)
    sequences = dataset.all_sequences
    
    # Filter/Limit
    if args.limit > 0:
        sequences = sequences[:args.limit]
        
    logger.info(f"Running inference on {len(sequences)} sequences...")
    
    all_results = []
    
    for seq in sequences:
        try:
            df = process_sequence(seq, config, output_dir, save_video=not args.no_video)
            if df is not None:
                all_results.append(df)
        except Exception as e:
            logger.error(f"Failed to process {seq.sequence_id}: {e}")
            import traceback
            traceback.print_exc()
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_dir / "all_results_summary.csv", index=False)
        logger.info(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
