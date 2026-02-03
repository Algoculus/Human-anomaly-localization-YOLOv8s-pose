# Giải Thích Chi Tiết Source Code

Tài liệu này cung cấp cái nhìn sâu sắc vào từng module của source code trong `src/urfd/`.

## 1. Module Track & Detect

### `class YOLOPoseDetector` ([yolo_pose.py])

Wrapper cho mô hình YOLOv8 với preprocessing tích hợp.

- `conf_thres=0.25`: Ngưỡng tự tin.
- `iou_thres=0.45`: Ngưỡng NMS.
- `_preprocess(image)`: Gọi `preprocess_lowlight()` theo nguyên tắc DRY.
- `detect(image)`: Trả về `{'bbox', 'keypoints', 'conf'}`.

### `class MultiPersonTracker` ([tracking.py])

Greedy Matching với Cost = $(1 - IoU) + Dist$.

---

## 2. Module Logic Cốt Lõi

### [features.py]

- `compute_frame_features`: Tính toán đặc trưng vật lý.
- **`image_size`**: Tuple (width, height) để kiểm tra border integrity.
- **`is_touching_border`**: Cờ báo hiệu BBox bị cắt.

### [rules.py] - 4-Path Detection

```python
# Path 1: Posture + Impact (disabled when border-clipped)
if angle >= 55.0 and AR >= 1.15 and dy_peak >= 5.0:
    angle_ar_condition = True

# Path 2: Height Drop
height_drop_condition = height_drop >= 0.18

# Path 3: Fast Motion
dy_condition = dy >= 10.0

# Path 4: High-Angle (NEW - catches slow/frontal falls)
if angle >= 65.0 and dy_peak >= 3.0:
    high_angle_condition = True
```

### `class FallStateMachine` ([smoothing.py])

- 3 trạng thái: `NORMAL` → `CANDIDATE` → `FALL_CONFIRMED`.
- `update(features)`: Chuyển trạng thái, tính score EMA.
- `_check_recovery()`: Kiểm tra đứng dậy.

---

## 3. Cấu hình ([configs/default.yaml])

| Parameter                     | Value | Purpose                           |
| ----------------------------- | ----- | --------------------------------- |
| `angle_thres`                 | 55.0  | Ngưỡng góc cho Path 1             |
| `ar_thres`                    | 1.15  | Ngưỡng AR cho Path 1              |
| `impact_dy_thres`             | 5.0   | Ngưỡng dy_peak cho Path 1         |
| `dy_peak_thres`               | 8.0   | Ngưỡng xác nhận fast motion       |
| `dy_fall_thres`               | 10.0  | Ngưỡng dy cho Path 3              |
| `height_drop_thres`           | 0.18  | Ngưỡng sụt giảm chiều cao         |
| `high_angle_thres`            | 65.0  | Ngưỡng góc cao cho Path 4         |
| `border_margin`               | 8     | Khoảng cách border check          |
| `confirm_frames`              | 3     | Số frame duy trì tư thế nằm       |
| `min_confirm_duration_frames` | 2     | Số frame FALL_CONFIRMED liên tiếp |

---

**Last Updated**: February 3, 2026
