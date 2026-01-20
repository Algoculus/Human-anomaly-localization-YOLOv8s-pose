from ultralytics import YOLO
import numpy as np

class YOLOPoseDetector:
    """Wrapper for YOLOv8 pose detection."""
    
    def __init__(self, model_path="yolov8s-pose.pt", imgsz=640, conf_thres=0.25, iou_thres=0.45):
        """Initialize YOLO pose detector.
        
        Args:
            model_path: Path to YOLO model weights
            imgsz: Image size for inference
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
    
    def detect(self, image):
        """Run pose detection on an image.
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            detections: List of dicts, each containing:
                - bbox: [x1, y1, x2, y2]
                - conf: confidence score
                - keypoints: array of shape (17, 3) - [x, y, conf] for each keypoint
                - bbox_area: bbox width * height
        """
        results = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            # Extract keypoints if available
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
            else:
                keypoints = None
            
            for i in range(len(boxes)):
                bbox = boxes[i]
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                
                det = {
                    "bbox": bbox.tolist(),
                    "conf": float(confs[i]),
                    "bbox_area": float(bbox_area)
                }
                
                if keypoints is not None and i < len(keypoints):
                    det["keypoints"] = keypoints[i]
                else:
                    det["keypoints"] = np.zeros((17, 3))
                
                detections.append(det)
        
        return detections
