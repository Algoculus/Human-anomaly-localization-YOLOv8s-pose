import argparse
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# COCO keypoint indices for YOLOv8-Pose (17 points)
KP_NOSE = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12

# COCO-17 skeleton pairs (YOLOv8-Pose)
SKELETON = [
    (5, 6),   # shoulders
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 11), (6, 12),  # torso to hips
    (11, 12),          # hips
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16) # right leg
]

HEAD = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes-ears
    (0, 5), (0, 6)                   # nose to shoulders (nhìn "YOLO" hơn)
]

def draw_skeleton(frame: np.ndarray, kp: np.ndarray, color=(0, 255, 0), conf=0.4):
    # vẽ điểm
    for i in range(kp.shape[0]):
        x, y, c = kp[i]
        if c >= conf:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    # vẽ xương
    for a, b in (SKELETON + HEAD):
        xa, ya, ca = kp[a]
        xb, yb, cb = kp[b]
        if ca >= conf and cb >= conf:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)


def _pt(kp: np.ndarray, idx: int) -> Tuple[float, float, float]:
    """Return (x, y, conf) for a given keypoint index."""
    return float(kp[idx, 0]), float(kp[idx, 1]), float(kp[idx, 2])


def _center(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, min(a[2], b[2])


def _torso_angle_deg(shoulder_xy: Tuple[float, float], hip_xy: Tuple[float, float]) -> float:
    """
    Angle of torso wrt vertical axis in degrees.
    0° = perfectly vertical; 90° = horizontal.
    """
    dx = shoulder_xy[0] - hip_xy[0]
    dy = shoulder_xy[1] - hip_xy[1]
    # Angle between torso vector and vertical: atan2(|dx|, |dy|)
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def _bbox_from_keypoints(kp: np.ndarray, conf_thres: float = 0.25) -> Optional[Tuple[float, float, float, float]]:
    xs = kp[:, 0]
    ys = kp[:, 1]
    cs = kp[:, 2]
    mask = cs >= conf_thres
    if not np.any(mask):
        return None
    x1 = float(np.min(xs[mask]))
    y1 = float(np.min(ys[mask]))
    x2 = float(np.max(xs[mask]))
    y2 = float(np.max(ys[mask]))
    return x1, y1, x2, y2


def _in_rect(xy: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
    x, y = xy
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class Thresholds:
    """Thresholds đã được điều chỉnh để ít nhảy nhãn và dễ bắt FALL hơn."""
    kp_conf: float = 0.40  # Tăng từ 0.25 để yêu cầu keypoint confidence cao hơn
    fall_vy_norm: float = 0.12  # Giảm từ 0.18 để nhạy hơn với chuyển động rơi
    fall_angle_delta_deg: float = 12.0  # Giảm từ 20.0 để nhạy hơn với thay đổi góc
    lying_angle_deg: float = 70.0  # Tăng từ 60.0 để chắc chắn hơn khi nằm
    flat_wh_ratio: float = 1.45  # Tăng từ 1.2 để chắc chắn hơn khi bbox "bẹt"
    inactivity_seconds: float = 5.0  # Tăng từ 2.5 để chờ lâu hơn trước khi báo inactivity
    inactivity_std_norm: float = 0.010  # normalized by bbox height


class FallFSM:
    """
    Minimal, explainable FSM:
    NORMAL -> (fall cues) -> FALL_DETECTED (latched briefly) -> LYING -> INACTIVITY
    Safe zone suppresses alarms when hip is inside the zone.
    """

    def __init__(self, fps: float, thresholds: Thresholds):
        self.fps = max(1.0, float(fps))
        self.th = thresholds

        self.state: str = "NORMAL"
        self.last_angle: Optional[float] = None
        self.last_hip_y: Optional[float] = None
        self.last_bbox_h: Optional[float] = None

        self.hip_hist: Deque[Tuple[float, float, float]] = deque(maxlen=int(self.th.inactivity_seconds * self.fps))
        self.fall_latch_frames = int(0.6 * self.fps)  # show "FALL DETECTED" ~0.6s
        self._fall_latch = 0

    def update(
        self,
        hip_xy: Tuple[float, float],
        shoulder_xy: Tuple[float, float],
        bbox: Tuple[float, float, float, float],
        safe_zone: Optional[Tuple[int, int, int, int]],
        is_safe: bool,
    ) -> str:
        x1, y1, x2, y2 = bbox
        bbox_h = max(1.0, y2 - y1)
        bbox_w = max(1.0, x2 - x1)
        wh_ratio = bbox_w / bbox_h

        angle = _torso_angle_deg(shoulder_xy, hip_xy)
        hip_y = hip_xy[1]

        # motion cues
        vy_norm = 0.0
        angle_delta = 0.0
        if self.last_hip_y is not None and self.last_bbox_h is not None:
            vy_norm = (hip_y - self.last_hip_y) / max(1.0, self.last_bbox_h)
        if self.last_angle is not None:
            angle_delta = abs(angle - self.last_angle)

        # lying cue
        is_lying = (angle >= self.th.lying_angle_deg) and (wh_ratio >= self.th.flat_wh_ratio)

        # fall cue: sudden downward hip movement + quick posture change
        is_fall = (vy_norm >= self.th.fall_vy_norm) and (angle_delta >= self.th.fall_angle_delta_deg)

        # inactivity cue: small hip variance while lying for T seconds
        self.hip_hist.append((hip_xy[0], hip_xy[1], bbox_h))
        is_inactive = False
        if is_lying and len(self.hip_hist) == self.hip_hist.maxlen:
            ys = np.array([h[1] for h in self.hip_hist], dtype=np.float32)
            hs = np.array([h[2] for h in self.hip_hist], dtype=np.float32)
            std_norm = float(np.std(ys / np.maximum(hs, 1.0)))
            is_inactive = std_norm <= self.th.inactivity_std_norm

        # FSM transitions
        if is_fall and not is_safe:
            self._fall_latch = self.fall_latch_frames
            self.state = "FALL DETECTED"
        elif self._fall_latch > 0:
            self._fall_latch -= 1
            self.state = "FALL DETECTED"
        else:
            if is_safe and is_lying:
                self.state = "SAFE ZONE (SLEEP/REST)"
            elif is_inactive and not is_safe:
                self.state = "INACTIVITY"
            elif is_lying:
                self.state = "LYING"
            else:
                self.state = "NORMAL"

        self.last_angle = angle
        self.last_hip_y = hip_y
        self.last_bbox_h = bbox_h

        return self.state


class TrackManager:
    """
    Quản lý FSM riêng cho từng track_id (mỗi người).
    Cleanup FSM khi track_id biến mất sau N frames.
    """

    def __init__(self, fps: float, thresholds: Thresholds, max_missing_frames: int = 30):
        self.fps = fps
        self.th = thresholds
        self.max_missing_frames = max_missing_frames
        self.fsms: Dict[int, FallFSM] = {}  # track_id -> FallFSM
        self.last_seen: Dict[int, int] = {}  # track_id -> frame_count
        self.frame_count = 0

    def get_or_create_fsm(self, track_id: int) -> FallFSM:
        if track_id not in self.fsms:
            self.fsms[track_id] = FallFSM(fps=self.fps, thresholds=self.th)
        return self.fsms[track_id]

    def update_tracks(self, track_ids: list):
        """Cập nhật last_seen và cleanup FSM cũ."""
        self.frame_count += 1
        seen = set(track_ids)
        # Cập nhật last_seen cho các track hiện tại
        for tid in seen:
            self.last_seen[tid] = self.frame_count
        # Cleanup FSM cho track đã biến mất quá lâu
        to_remove = []
        for tid, last_frame in self.last_seen.items():
            if tid not in seen and (self.frame_count - last_frame) > self.max_missing_frames:
                to_remove.append(tid)
        for tid in to_remove:
            # Chỉ xóa nếu tồn tại trong dict (tránh KeyError)
            if tid in self.fsms:
                del self.fsms[tid]
            if tid in self.last_seen:
                del self.last_seen[tid]

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Màu riêng cho mỗi track_id (để dễ phân biệt)."""
        # Hash track_id để có màu ổn định
        np.random.seed(track_id % 1000)
        return tuple(int(c) for c in np.random.randint(50, 255, 3))


class SafeZoneUI:
    def __init__(self):
        self.drawing = False
        self.p0: Optional[Tuple[int, int]] = None
        self.p1: Optional[Tuple[int, int]] = None
        self.rect: Optional[Tuple[int, int, int, int]] = None

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.p0 = (x, y)
            self.p1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.p1 = (x, y)
            if self.p0 and self.p1:
                x1 = min(self.p0[0], self.p1[0])
                y1 = min(self.p0[1], self.p1[1])
                x2 = max(self.p0[0], self.p1[0])
                y2 = max(self.p0[1], self.p1[1])
                self.rect = (x1, y1, x2, y2)

    def draw(self, frame: np.ndarray) -> None:
        if self.p0 and self.p1 and (self.drawing or self.rect is None):
            cv2.rectangle(frame, self.p0, self.p1, (255, 255, 0), 2)
        if self.rect is not None:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "SAFE ZONE", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def load_safe_zone(path: str) -> Optional[Tuple[int, int, int, int]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    r = d.get("rect")
    if not r or len(r) != 4:
        return None
    return int(r[0]), int(r[1]), int(r[2]), int(r[3])


def save_safe_zone(path: str, rect: Tuple[int, int, int, int]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rect": list(rect)}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam, or path to video file")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="YOLOv8 pose model path/name")
    ap.add_argument("--safe-zone", default="outputs/safe_zone.json", help="Path to save/load safe zone rectangle")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    source = 0 if str(args.source) == "0" else args.source

    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    th = Thresholds()
    track_mgr = TrackManager(fps=fps, thresholds=th, max_missing_frames=30)
    ui = SafeZoneUI()

    safe_rect = load_safe_zone(args.safe_zone)

    # Flag để chỉ warning 1 lần về tracking (dùng list để có thể modify trong nested scope)
    tracking_warned = [False]

    win = "URFall Demo - Multi-Person Tracking (press z=reset safe zone, s=save, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, ui.mouse_cb)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Keep last loaded rect unless user is drawing/has new rect
        if ui.rect is not None:
            safe_rect = ui.rect

        # Dùng track() thay vì predict() để có tracking ID
        res = model.track(frame, imgsz=args.imgsz, verbose=False, persist=True, tracker="bytetrack.yaml")

        track_ids = []
        tracking_available = True  # Flag để kiểm tra tracking có hoạt động không
        if res and len(res) > 0 and res[0].keypoints is not None:
            boxes = res[0].boxes
            keypoints = res[0].keypoints

            # Lấy track IDs (nếu có)
            if boxes.id is not None:
                track_ids_list = boxes.id.cpu().numpy().astype(int).tolist()
            else:
                # Nếu không có tracking, tạo ID tạm theo index
                tracking_available = False
                track_ids_list = list(range(len(boxes)))
                # Warning chỉ hiện 1 lần (frame đầu tiên có detection)
                if not tracking_warned[0]:
                    print("⚠️  WARNING: Tracking không hoạt động (boxes.id is None).")
                    print("   → ID sẽ không ổn định theo thời gian (dùng index tạm).")
                    print("   → Giải pháp: pip install -U ultralytics")
                    tracking_warned[0] = True

            # Lặp qua từng detection (mỗi người)
            for idx, kp in enumerate(keypoints.data):
                if idx >= len(track_ids_list):
                    continue
                track_id = track_ids_list[idx]
                track_ids.append(track_id)

                kp_np = kp.cpu().numpy()  # (17, 3)
                bbox = _bbox_from_keypoints(kp_np, conf_thres=th.kp_conf)
                if bbox is None:
                    continue

                # Lấy keypoints cần thiết
                lh = _pt(kp_np, KP_L_HIP)
                rh = _pt(kp_np, KP_R_HIP)
                ls = _pt(kp_np, KP_L_SHOULDER)
                rs = _pt(kp_np, KP_R_SHOULDER)

                hip = _center(lh, rh)
                sh = _center(ls, rs)
                hip_xy = (hip[0], hip[1])
                sh_xy = (sh[0], sh[1])

                # Lấy hoặc tạo FSM cho track_id này
                fsm = track_mgr.get_or_create_fsm(track_id)
                is_safe = bool(safe_rect is not None and _in_rect(hip_xy, safe_rect))
                label = fsm.update(
                    hip_xy=hip_xy,
                    shoulder_xy=sh_xy,
                    bbox=bbox,
                    safe_zone=safe_rect,
                    is_safe=is_safe,
                )

                # Màu theo track_id và state
                base_color = track_mgr.get_color(track_id)
                if label in ("FALL DETECTED", "INACTIVITY"):
                    color = (0, 0, 255)  # Đỏ cho cảnh báo
                elif label.startswith("SAFE ZONE"):
                    color = (0, 255, 255)  # Vàng cho safe zone
                elif label == "LYING":
                    color = (255, 170, 0)  # Cam cho lying
                else:
                    color = base_color  # Màu riêng cho mỗi người khi NORMAL

                # Vẽ bbox
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Vẽ khung xương kiểu YOLO
                draw_skeleton(frame, kp_np, color=color, conf=th.kp_conf)

                # Vẽ label + track_id
                label_text = f"ID:{track_id} {label}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Cập nhật tracking manager (cleanup FSM cũ)
        track_mgr.update_tracks(track_ids)

        # Draw safe zone overlay
        if safe_rect is not None:
            x1, y1, x2, y2 = safe_rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "SAFE ZONE", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw info
        num_tracks = len(track_ids)
        info_text = f"Tracking: {num_tracks} person(s)"
        if not tracking_available and num_tracks > 0:
            info_text += " [NO TRACKING]"
        cv2.putText(frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "z:reset safe zone | s:save | q:quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("z"):
            ui.rect = None
            ui.p0 = None
            ui.p1 = None
        if k == ord("s") and safe_rect is not None:
            save_safe_zone(args.safe_zone, safe_rect)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

