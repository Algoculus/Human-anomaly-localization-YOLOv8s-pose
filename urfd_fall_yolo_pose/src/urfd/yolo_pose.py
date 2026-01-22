from ultralytics import YOLO
import numpy as np
import cv2

class YOLOPoseDetector:
    # YOLOv8 pose detection wrapper with optional low-light preprocessing
    
    def __init__(self, model_path="yolov8s-pose.pt", imgsz=640, conf_thres=0.25, iou_thres=0.45,
                 preprocess_lowlight=False, gamma=1.3, clahe_clip=2.0, clahe_grid=8):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.preprocess_lowlight = preprocess_lowlight
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
    
    def _preprocess(self, image):
        # Apply gamma correction + CLAHE for low-light scenes
        if not self.preprocess_lowlight:
            return image
        
        # Gamma correction + CLAHE on Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        
        # Gamma correction
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        y_gamma = cv2.LUT(y_channel, table)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
        y_clahe = clahe.apply(y_gamma)
        
        # Reconstruct
        ycrcb[:, :, 0] = y_clahe
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    def detect(self, image):
        # Run pose detection, returns list of detections with bbox, keypoints, conf
        # NOTE: Detects ALL people in frame, tracker selects primary person later
        # Apply preprocessing
        processed_image = self._preprocess(image)
        
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
