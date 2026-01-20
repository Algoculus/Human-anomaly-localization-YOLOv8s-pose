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
    """Draw skeleton on image.
    
    Args:
        img: Image to draw on
        keypoints: Keypoints array (17, 3)
        color: Color tuple (B, G, R)
        conf_thres: Confidence threshold for drawing
    """
    # Draw keypoints
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

def create_overlay_video(frames, detections_list, states, scores, output_path, fps):
    """Create overlay video with detection results.
    
    Args:
        frames: List of BGR images
        detections_list: List of detection lists per frame
        states: List of state strings per frame
        scores: List of fall scores per frame
        output_path: Output video path
        fps: Output video FPS
    """
    if len(frames) == 0:
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for idx, frame in enumerate(frames):
        overlay = frame.copy()
        
        detections = detections_list[idx]
        state = states[idx]
        score = scores[idx]
        
        # Determine color based on state
        if state == "NORMAL":
            color = COLOR_NORMAL
        elif state == "CANDIDATE":
            color = COLOR_CANDIDATE
        else:  # FALL_CONFIRMED
            color = COLOR_FALL
        
        # Find primary person
        if len(detections) > 0:
            primary_idx = max(range(len(detections)), key=lambda i: detections[i]["bbox_area"])
            det = detections[primary_idx]
            
            # Draw bbox
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw skeleton
            keypoints = det["keypoints"]
            draw_skeleton(overlay, keypoints, color)
        
        # Draw state text
        cv2.putText(overlay, f"State: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(overlay, f"Score: {score:.3f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        writer.write(overlay)
    
    writer.release()
