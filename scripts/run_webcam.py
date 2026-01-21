import argparse
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.infer_pose import PoseInferencePipeline
from src.rules.features import FeatureExtractor
from src.rules.scoring import FallScorer
from src.rules.state_machine import FallStateMachine, StateMachineContext
from src.utils.visualization import Visualizer

def run_camera(source, config, show_window=True, save_path=None):
    """
    Run Fall Detection on a camera source (Webcam, RTSP, or Video File).
    """
    # Open Video Source
    # Check if source is an integer (webcam index)
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {source}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Source opened: {source} ({width}x{height} @ {fps} FPS)")

    # Initialize Pipeline
    pipeline = PoseInferencePipeline(config)
    feature_extractor = FeatureExtractor(config)
    scorer = FallScorer(config)
    fsm = FallStateMachine(config)
    context = StateMachineContext() # Reset context
    visualizer = Visualizer(config)
    
    # Video Writer (Optional)
    writer = None
    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        logger.info(f"Recording to: {out_path}")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # --- 1. Inference ---
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pipe_out = pipeline.process_frame(frame_rgb)
            poses = pipe_out['poses']
            
            primary_pose = None
            if poses:
                primary_pose = poses[0] # Track primary person
            
            # --- 2. Feature Extraction ---
            # Realtime Note: No accelerometer data available here
            # Features will rely purely on Pose (Vision-based)
            features = feature_extractor.extract(
                primary_pose, 
                (height, width),
                depth_features=None, # No depth in standard webcam
                acc_data=None        # No acc
            )
            
            # --- 3. Scoring ---
            scores = scorer.calculate(features)
            
            # --- 4. State Machine ---
            context.recent_velocity = features.velocity_y
            state = fsm.update(scores, context)
            
            # --- 5. Visualization ---
            viz_frame = visualizer.draw_frame(
                frame_rgb.copy(),
                frame_count,
                state,
                scores.total_score,
                behavior_label=context.behavior_label,
                scores_dict={
                    'Score': scores.total_score,
                    'Drop Score': scores.sudden_drop_score,
                    'Prone': scores.prone_score,
                    'State': state.value,
                    'Label': context.behavior_label
                },
                poses=poses
            )
            
            # Convert back to BGR for display/saving
            viz_frame_bgr = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)
            
            # Show
            if show_window:
                try:
                    cv2.imshow('Fall Detection - Realtime', viz_frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error as e:
                    logger.error("OpenCV GUI error: cv2.imshow not supported in this environment.")
                    logger.error("Consider installing 'opencv-python' (not headless) or run with --no-show")
                    logger.info("Switching to --no-show mode automatically...")
                    show_window = False
            
            # Save
            if writer:
                writer.write(viz_frame_bgr)
                
            # FPS Log
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                logger.info(f"FPS: {current_fps:.1f} | Label: {context.behavior_label}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info("Camera closed.")

def main():
    parser = argparse.ArgumentParser(description="Realtime Fall Detection from Camera")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0) or video path")
    parser.add_argument("--save", type=str, default=None, help="Path to save output video (e.g. output.mp4)")
    parser.add_argument("--no-show", action="store_true", help="Do not show window")
    args = parser.parse_args()
    
    config = get_config()
    
    run_camera(
        source=args.source,
        config=config,
        show_window=not args.no_show,
        save_path=args.save
    )

if __name__ == "__main__":
    main()
