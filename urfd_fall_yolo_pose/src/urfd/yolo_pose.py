from ultralytics import YOLO
import numpy as np
import cv2
from .preprocessing import preprocess_lowlight

class YOLOPoseDetector:
    """
    Wrapper for YOLOv8 pose detection with optional preprocessing.
    
    Features:
    - Low-light preprocessing (gamma + CLAHE)
    - Configurable confidence and IoU thresholds
    - Returns keypoints in COCO format (17 keypoints)
    """
    
    def __init__(self, model_path="yolov8s-pose.pt", imgsz=640, conf_thres=0.25, 
                 iou_thres=0.45, preprocess_lowlight=False, gamma=1.3, 
                 clahe_clip=2.0, clahe_grid=8):
        """
        Initialize YOLO pose detector.
        
        Args:
            model_path: Path to YOLO model weights
            imgsz: Image size for inference (resized to this)
            conf_thres: Confidence threshold (reject detections below this)
            iou_thres: IoU threshold for NMS (Non-Maximum Suppression)
            preprocess_lowlight: Enable low-light preprocessing
            gamma: Gamma correction value for preprocessing
            clahe_clip: CLAHE clip limit for preprocessing
            clahe_grid: CLAHE grid size for preprocessing
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.preprocess_lowlight = preprocess_lowlight
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
    
    def _preprocess(self, image):
        """
        Apply low-light preprocessing if enabled.
        
        Uses DRY principle - delegates to preprocessing module.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            Preprocessed BGR image
        """
        if not self.preprocess_lowlight:
            return image
        # Reuse preprocessing function (DRY principle)
        return preprocess_lowlight(image, self.gamma, self.clahe_clip, self.clahe_grid)
    
    def detect(self, image):
        """
        Run pose detection on an image.
        
        Process:
        1. Apply preprocessing (if enabled)
        2. Run YOLO inference
        3. Extract bboxes, confidences, and keypoints
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            detections: List of dicts, each containing:
                - bbox: [x1, y1, x2, y2]
                - conf: confidence score
                - keypoints: array of shape (17, 3) - [x, y, conf]
                - bbox_area: bounding box area
        """
        # Apply low-light preprocessing if enabled
        processed_image = self._preprocess(image)
        
        # Run YOLO inference
        results = self.model.predict(
            processed_image,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4] - x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()  # [N]
            
            # Extract keypoints if available (COCO format: 17 keypoints)
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()  # [N, 17, 3]
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
                
                # Attach keypoints or zeros if not available
                if keypoints is not None and i < len(keypoints):
                    det["keypoints"] = keypoints[i]
                else:
                    det["keypoints"] = np.zeros((17, 3))
                
                detections.append(det)
        
        return detections
