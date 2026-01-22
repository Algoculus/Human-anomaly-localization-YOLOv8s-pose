"""
Real-time Fall Detection from Webcam/Camera - Fixed for Enhanced System

Runs fall detection on live camera feed with optimized visualization.
"""

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
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {source}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 100:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Source opened: {source} ({width}x{height} @ {fps} FPS)")

    # Initialize Pipeline
    pipeline = PoseInferencePipeline(config)
    feature_extractor = FeatureExtractor(config)
    scorer = FallScorer(config)
    fsm = FallStateMachine(config)
    context = StateMachineContext()  # Create context ONCE
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
    
    # Fall alert history for visual feedback
    alert_history = []
    max_alert_history = 90  # 3 seconds at 30fps

    try:
        logger.info("Starting real-time fall detection... Press 'q' to quit")
        logger.info("Press 'r' to reset state machine")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            frame_count += 1
            
            # --- 1. Inference ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pipe_out = pipeline.process_frame(frame_rgb)
            poses = pipe_out['poses']
            
            primary_pose = None
            if poses:
                primary_pose = poses[0]
            
            # --- 2. Feature Extraction ---
            features = feature_extractor.extract(
                primary_pose, 
                (height, width),
                depth_features=None,  # No depth in webcam
                acc_data=None         # No accelerometer
            )
            
            # --- 3. Scoring ---
            scores = scorer.calculate(features)
            
            # --- 4. State Machine Update - FIXED ---
            state = fsm.update(scores, features, context)
            
            # Track alerts
            if context.has_alerted and context.frames_since_alert < 10:
                alert_history.append(True)
            else:
                alert_history.append(False)
            
            if len(alert_history) > max_alert_history:
                alert_history.pop(0)
            
            # --- 5. Enhanced Visualization ---
            viz_frame = visualizer.draw_frame(
                frame_rgb.copy(),
                frame_count,
                state,
                scores.total_score,
                behavior_label=context.behavior_label,
                scores_dict={
                    'Total Score': scores.total_score,
                    'Drop': scores.sudden_drop_score,
                    'Prone': scores.prone_score,
                    'Impact': scores.impact_score,
                    'Sustained': scores.sustained_lying_score,
                    'Activity': features.activity_type,
                    'Orient': f"{features.body_orientation:.0f}°",
                    'Hip H': f"{features.hip_height_ratio:.2f}",
                },
                poses=poses
            )
            
            # Add alert overlay if recent alert
            if any(alert_history[-30:]):  # Alert in last 1 second
                # Red flashing border
                if frame_count % 10 < 5:
                    cv2.rectangle(viz_frame, (0, 0), (width-1, height-1), 
                                (0, 0, 255), thickness=8)
                
                # Alert text
                cv2.putText(viz_frame, "⚠ FALL ALERT ⚠", 
                           (width//2 - 150, height - 30),
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            
            # Add FPS counter
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                logger.info(f"FPS: {current_fps:.1f} | State: {state.value} | Activity: {features.activity_type}")
            
            # Convert back to BGR for display/saving
            viz_frame_bgr = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)
            
            # Show
            if show_window:
                try:
                    cv2.imshow('Fall Detection - Real-time', viz_frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        logger.info("User quit")
                        break
                    elif key == ord('r'):
                        logger.info("Resetting state machine...")
                        context = StateMachineContext()
                        feature_extractor.reset()
                        logger.info("State machine reset complete")
                        
                except cv2.error as e:
                    logger.error("OpenCV GUI error - switching to no-show mode")
                    show_window = False
            
            # Save
            if writer:
                writer.write(viz_frame_bgr)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        
        # Final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Session Summary:")
        logger.info(f"  Duration: {elapsed:.1f}s")
        logger.info(f"  Frames: {frame_count}")
        logger.info(f"  Avg FPS: {avg_fps:.1f}")
        logger.info(f"  Alerts triggered: {sum(alert_history)}")
        logger.info(f"{'='*60}")
        logger.info("Camera closed.")

def main():
    parser = argparse.ArgumentParser(description="Real-time Fall Detection from Camera")
    parser.add_argument("--source", type=str, default="0", 
                       help="Camera index (0) or video path or RTSP URL")
    parser.add_argument("--save", type=str, default=None, 
                       help="Path to save output video (e.g. output.mp4)")
    parser.add_argument("--no-show", action="store_true", 
                       help="Do not show window (headless mode)")
    args = parser.parse_args()
    
    config = get_config()
    
    logger.info(f"Starting fall detection with source: {args.source}")
    logger.info(f"Enhanced system active with:")
    logger.info(f"  - Activity discrimination (BENDING, FALLING, LYING)")
    logger.info(f"  - Multi-stage detection (Drop→Impact→Prone→Sustained)")
    logger.info(f"  - Temporal reasoning with fall sequence validation")
    
    run_camera(
        source=args.source,
        config=config,
        show_window=not args.no_show,
        save_path=args.save
    )

if __name__ == "__main__":
    main()