import cv2
import numpy as np

# COCO pose skeleton connections
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

# Colors
COLOR_NORMAL = (0, 255, 0)      # Green
COLOR_CANDIDATE = (0, 165, 255)  # Orange
COLOR_FALL = (0, 0, 255)         # Red

def draw_skeleton(img, keypoints, color, conf_thres=0.5):
    # Draw skeleton on image.
    for i, kp in enumerate(keypoints):
        x, y, conf = kp
        if conf >= conf_thres:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)
    
    # Draw skeleton connections
    for connection in SKELETON:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            kp1 = keypoints[idx1]
            kp2 = keypoints[idx2]
            if kp1[2] >= conf_thres and kp2[2] >= conf_thres:
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0]), int(kp2[1]))
                cv2.line(img, pt1, pt2, color, 2)

def create_overlay_video(frames, all_tracks_data, output_path, fps):
    # Create overlay video with detection results for multiple tracks.
    # Args:
    #   frames: List of BGR images
    #   all_tracks_data: List (per frame) of Dict (track_id -> {bbox, keypoints, state, score})
    #   output_path: Output video path
    #   fps: Output video FPS
    
    if len(frames) == 0:
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for idx, frame in enumerate(frames):
        overlay = frame.copy()
        
        frame_tracks = all_tracks_data[idx] if idx < len(all_tracks_data) else {}
        
        for tid, track_info in frame_tracks.items():
            state = track_info.get("state", "NORMAL")
            score = track_info.get("score", 0.0)
            
            # Determine color based on state
            if state == "NORMAL":
                color = COLOR_NORMAL
            elif state == "CANDIDATE":
                color = COLOR_CANDIDATE
            else:  # FALL_CONFIRMED (or FALL)
                color = COLOR_FALL
            
            # Draw bbox
            if "bbox" in track_info and track_info["bbox"] is not None:
                bbox = track_info["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and State
                label = f"ID:{tid} {state} {score:.2f}"
                cv2.putText(overlay, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw skeleton
            if "keypoints" in track_info and track_info["keypoints"] is not None:
                draw_skeleton(overlay, track_info["keypoints"], color)
        
        writer.write(overlay)
    
    writer.release()
