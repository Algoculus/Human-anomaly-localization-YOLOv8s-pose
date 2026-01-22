import argparse
import csv
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from ultralytics import YOLO

from urfall_demo_fsm import FallFSM, Thresholds, _bbox_from_keypoints, _center, _in_rect, _pt, KP_L_HIP, KP_L_SHOULDER, KP_R_HIP, KP_R_SHOULDER


def iter_videos(data_dir: str) -> List[Tuple[str, int]]:
    """
    Event-level ground truth (weak):
    - fall-*-cam0.mp4 -> y=1
    - adl-*-cam0.mp4  -> y=0
    """
    items: List[Tuple[str, int]] = []
    for name in os.listdir(data_dir):
        if not name.endswith(".mp4"):
            continue
        lower = name.lower()
        if lower.startswith("fall-") and "cam0" in lower:
            items.append((os.path.join(data_dir, name), 1))
        elif lower.startswith("adl-") and "cam0" in lower:
            items.append((os.path.join(data_dir, name), 0))
    items.sort(key=lambda x: x[0])
    return items


def predict_clip(
    model: YOLO,
    video_path: str,
    th: Thresholds,
    alert_labels: Tuple[str, ...],
    safe_rect: Optional[Tuple[int, int, int, int]] = None,
    max_seconds: Optional[float] = None,
    imgsz: int = 640,
) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    fsm = FallFSM(fps=fps, thresholds=th)
    alerted = False
    frames = 0
    alerted_frame = None

    max_frames = None
    if max_seconds is not None:
        max_frames = int(max_seconds * fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        if max_frames is not None and frames > max_frames:
            break

        res = model.predict(frame, imgsz=imgsz, verbose=False)
        label = "NORMAL"
        if res and res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
            kp = res[0].keypoints.data[0].cpu().numpy()  # (17, 3)
            bbox = _bbox_from_keypoints(kp, conf_thres=th.kp_conf)
            if bbox is not None:
                hip = _center(_pt(kp, KP_L_HIP), _pt(kp, KP_R_HIP))
                sh = _center(_pt(kp, KP_L_SHOULDER), _pt(kp, KP_R_SHOULDER))
                hip_xy = (hip[0], hip[1])
                sh_xy = (sh[0], sh[1])
                is_safe = bool(safe_rect is not None and _in_rect(hip_xy, safe_rect))
                label = fsm.update(hip_xy=hip_xy, shoulder_xy=sh_xy, bbox=bbox, safe_zone=safe_rect, is_safe=is_safe)

        if label in alert_labels:
            alerted = True
            alerted_frame = frames
            break  # event-level: first alert is enough

    cap.release()
    return {
        "video": os.path.basename(video_path),
        "frames_seen": frames,
        "pred": 1 if alerted else 0,
        "first_alert_frame": alerted_frame if alerted else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="data/UR_Fall_Detection_Dataset/data",
        help="Folder containing fall/adl mp4 files (cam0)",
    )
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-seconds", type=float, default=None, help="Optional cap for faster evaluation")
    ap.add_argument("--out", default="outputs/urfall_event_metrics.csv")
    args = ap.parse_args()

    th = Thresholds()
    model = YOLO(args.model)

    items = iter_videos(args.data_dir)
    if not items:
        raise RuntimeError(f"No cam0 mp4 files found in: {args.data_dir}")

    # Event-level decision: count as positive if we ever output these labels.
    alert_labels = ("FALL DETECTED", "INACTIVITY")

    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[Dict] = []

    for path, yt in items:
        r = predict_clip(
            model=model,
            video_path=path,
            th=th,
            alert_labels=alert_labels,
            safe_rect=None,
            max_seconds=args.max_seconds,
            imgsz=args.imgsz,
        )
        rows.append({"gt": yt, **r})
        y_true.append(yt)
        y_pred.append(int(r["pred"]))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "gt",
                "pred",
                "first_alert_frame",
                "frames_seen",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print("Event-level evaluation (weak GT by filename):")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Recall={rec:.3f} Specificity={spec:.3f} Precision={prec:.3f} F1={f1:.3f}")
    print(f"Saved per-video predictions to: {args.out}")
    print("Thresholds used:", asdict(th))


if __name__ == "__main__":
    main()

