# Technical Documentation - Fall Detection System

## Tài liệu Kỹ Thuật - Hệ Thống Phát Hiện Té Ngã

---

## 📋 Mục Lục / Table of Contents

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Pipeline Xử Lý Chính](#2-pipeline-xử-lý-chính)
3. [Module 1: YOLO Pose Detection](#3-module-1-yolo-pose-detection)
4. [Module 2: Tracking (Theo Dõi)](#4-module-2-tracking-theo-dõi)
5. [Module 3: Features Extraction](#5-module-3-features-extraction-trích-xuất-đặc-trưng)
6. [Module 4: Rule Engine](#6-module-4-rule-engine-động-cơ-luật)
7. [Module 5: Finite State Machine (FSM)](#7-module-5-finite-state-machine-fsm)
8. [Các Tham Số Cấu Hình](#8-các-tham-số-cấu-hình)
9. [Luồng Dữ Liệu Đầy Đủ](#9-luồng-dữ-liệu-đầy-đủ)

---

## 1. Tổng Quan Hệ Thống

### 1.1 Kiến trúc Tổng Thể

Hệ thống phát hiện té ngã sử dụng kiến trúc pipeline đa giai đoạn:

```
Input Video
    ↓
[Frame Extraction] ← Tách từng frame từ video
    ↓
[YOLO Pose Detection] ← Phát hiện người và keypoints
    ↓
[Multi-Person Tracking] ← Theo dõi người qua các frame
    ↓
[Features Extraction] ← Trích xuất đặc trưng vật lý
    ↓
[Rule Engine] ← Áp dụng luật phát hiện té ngã
    ↓
[Finite State Machine] ← Temporal smoothing và xác nhận
    ↓
Output: FALL/NORMAL + Confidence Score
```

### 1.2 Nguyên Lý Hoạt Động

Hệ thống hoạt động dựa trên 3 nguyên lý chính:

1. **Geometric Analysis (Phân tích hình học)**: Sử dụng góc cơ thể, tỷ lệ khung hình
2. **Motion Analysis (Phân tích chuyển động)**: Theo dõi vận tốc theo phương đứng
3. **Temporal Coherence (Tính liên tục thời gian)**: Loại bỏ nhiễu và xác nhận qua FSM

---

## 2. Pipeline Xử Lý Chính

### 2.1 Entry Point: `infer_sequence.py`

File: `urfd_fall_yolo_pose/scripts/infer_sequence.py`

#### Chức năng

Script chính để xử lý một chuỗi video hoàn chỉnh.

#### Input

- `seq_path`: Đường dẫn đến folder chứa frames của video
- `config_path`: Đường dẫn file cấu hình YAML

#### Output

- Video overlay với bounding boxes, states, scores
- Nhãn dự đoán: 0 (NORMAL) hoặc 1 (FALL)
- Frame đầu tiên phát hiện té ngã
- Điểm tin cậy tối đa

#### Luồng Xử Lý

```python
def infer_sequence(seq_path, config_path):
    # 1. Load cấu hình và frames
    config = load_config(config_path)
    frames, frame_paths = load_sequence_frames(seq_path)

    # 2. Khởi tạo các module
    detector = YOLOPoseDetector(...)           # Phát hiện pose
    tracker = PrimaryPersonTracker(config)     # Theo dõi người chính
    fallback_tracker = FallbackTracker(...)    # Backup tracker
    state_machine = FallStateMachine(config)   # FSM

    # 3. Vòng lặp xử lý từng frame
    for idx, frame in enumerate(frames):
        # 3.1. Phát hiện người và keypoints
        detections = detector.detect(frame)

        # 3.2. Fallback tracking nếu mất detection
        fallback_det = fallback_tracker.update(frame, detections)
        if fallback_det is not None:
            detections = [fallback_det]

        # 3.3. Trích xuất features
        features = compute_frame_features(
            detections, config, all_features, tracker, idx
        )

        # 3.4. Cập nhật FSM
        state, score = state_machine.update(features)

        # 3.5. Lưu kết quả
        all_states.append(state)
        all_scores.append(score)

    # 4. Xác định nhãn cuối cùng
    fall_confirmed = any(s == "FALL_CONFIRMED" for s in all_states)

    # 5. Tạo video output
    create_overlay_video(frames, detections, all_states, all_scores, ...)
```

---

## 3. Module 1: YOLO Pose Detection

### 3.1 File: `urfd/yolo_pose.py`

#### Class: `YOLOPoseDetector`

##### Chức năng

Wrapper cho YOLOv8-Pose để phát hiện người và 17 keypoints COCO.

##### Input (Constructor)

```python
YOLOPoseDetector(
    model_path="yolov8s-pose.pt",    # Đường dẫn model weights
    imgsz=640,                        # Kích thước resize image
    conf_thres=0.25,                  # Ngưỡng confidence
    iou_thres=0.45,                   # Ngưỡng IoU cho NMS
    preprocess_lowlight=True,         # Bật xử lý ánh sáng yếu
    gamma=1.3,                        # Gamma correction
    clahe_clip=2.0,                   # CLAHE clip limit
    clahe_grid=8                      # CLAHE grid size
)
```

##### Output của `detect(image)`

```python
[
    {
        "bbox": [x1, y1, x2, y2],      # Tọa độ bounding box
        "conf": 0.87,                   # Độ tin cậy detection
        "bbox_area": 45320.5,           # Diện tích bbox
        "keypoints": np.array([         # 17 keypoints COCO
            [x, y, conf],  # 0: Nose
            [x, y, conf],  # 1: Left Eye
            ...            # ... (17 điểm)
            [x, y, conf]   # 16: Right Ankle
        ])
    },
    ...  # Nhiều người nếu có
]
```

##### COCO Keypoints Layout (17 điểm)

```
0: Nose (Mũi)
1: Left Eye                    2: Right Eye
3: Left Ear                    4: Right Ear
5: Left Shoulder               6: Right Shoulder
7: Left Elbow                  8: Right Elbow
9: Left Wrist                 10: Right Wrist
11: Left Hip                  12: Right Hip
13: Left Knee                 14: Right Knee
15: Left Ankle                16: Right Ankle
```

##### Preprocessing Low-Light

**Mục đích**: Tăng chất lượng ảnh trong điều kiện thiếu sáng

**Quy trình**:

1. **Gamma Correction**: Tăng độ sáng tổng thể

   ```
   output = 255 * ((input / 255) ^ (1/gamma))
   với gamma = 1.3 (làm sáng hơn)
   ```

2. **CLAHE (Contrast Limited AHE)**: Tăng độ tương phản cục bộ
   - Chia ảnh thành lưới 8x8
   - Áp dụng histogram equalization cho mỗi ô
   - Giới hạn clip = 2.0 để tránh nhiễu

---

## 4. Module 2: Tracking (Theo Dõi)

### 4.1 File: `urfd/tracking.py`

#### Class: `MultiPersonTracker`

##### Chức năng

Theo dõi nhiều người qua các frame sử dụng IoU-based matching.

##### Thuật Toán Tracking

**Bước 1: Tính Cost Matrix**

Cho mỗi cặp (track_old, detection_new):

```python
# Dự đoán vị trí mới dựa vào velocity
pred_center = track.center + track.velocity
pred_bbox = reconstruct_bbox(pred_center, track.width, track.height)

# Tính IoU (Intersection over Union)
iou = compute_iou(pred_bbox, det.bbox)

# Tính khoảng cách tâm bbox
center_dist = compute_center_distance(pred_bbox, det.bbox)

# Cost = thấp hơn = khớp tốt hơn
cost = (1.0 - iou) + center_dist
```

**Bước 2: Greedy Matching**

```python
# Sắp xếp tất cả cặp theo cost tăng dần
matches = sort_by_cost(all_pairs)

# Gán lần lượt, tránh trùng lặp
for (cost, track_id, det_id) in matches:
    if not assigned(track_id) and not assigned(det_id):
        assign(track_id, det_id)
```

**Bước 3: Cập Nhật Tracks**

```python
for track_id, det_id in matches:
    # Cập nhật vị trí
    track.center = new_center
    track.bbox = new_bbox

    # Cập nhật velocity (EMA)
    velocity = new_center - old_center
    track.velocity = 0.7 * track.velocity + 0.3 * velocity

    # Reset missing counter
    track.missing_count = 0
```

**Bước 4: Xử Lý Unmatched**

```python
# Tracks không khớp: tăng missing_count
for unmatched_track in unmatched_tracks:
    track.missing_count += 1
    if track.missing_count > MAX_MISSING:
        delete_track(track)

# Detections không khớp: tạo track mới
for unmatched_det in unmatched_detections:
    create_new_track(det)
```

##### Fall Likelihood Score

**Mục đích**: Ưu tiên tracking người có khả năng ngã cao.

```python
def compute_fall_likelihood_score(bbox, keypoints):
    # 1. Aspect Ratio (40% weight)
    ar = width / height
    ar_score = min(ar / 2.0, 1.0)  # Cao hơn = nằm nhiều hơn

    # 2. Vertical Position (30% weight)
    center_y = (y1 + y2) / 2
    y_score = min(center_y / 500, 1.0)  # Thấp hơn = gần mặt đất

    # 3. Body Angle (30% weight)
    angle_deg = compute_torso_angle(keypoints)
    angle_score = min(angle_deg / 90, 1.0)  # Cao hơn = nằm ngang

    return 0.4 * ar_score + 0.3 * y_score + 0.3 * angle_score
```

---

## 5. Module 3: Features Extraction (Trích Xuất Đặc Trưng)

### 5.1 File: `urfd/features.py`

#### Function: `compute_frame_features()`

##### Chức năng

Trích xuất các đặc trưng vật lý từ detection và history.

##### Input

```python
compute_frame_features(
    detection,              # Dict detection từ YOLO
    keypoint_conf_thres,   # Ngưỡng confidence keypoint (0.5)
    dy_window,             # Cửa sổ tính vận tốc (5 frames)
    history,               # Lịch sử features các frame trước
    track_id,              # ID của track
    frame_idx,             # Index frame hiện tại
    image_size             # (width, height) của ảnh
)
```

##### Output

```python
{
    "track_id": 1,
    "bbox": [x1, y1, x2, y2],
    "center_y": cy,
    "height": h,
    "width": w,
    "bbox_aspect_ratio": w/h,
    "shoulder_mid": [x, y],
    "hip_mid": [x, y],
    "body_angle_deg": 35.7,
    "feature_valid": True,
    "dy": 2.5,                    # Vận tốc tức thời
    "dy_velocity": 3.2,           # Vận tốc trung bình
    "dy_peak": 8.1,               # Vận tốc đỉnh
    "is_touching_border": False,
    "detection": {...}
}
```

---

### 5.2 Chi Tiết Các Đại Lượng Vật Lý

#### 5.2.1 Bounding Box Features

**1. Center Y (Tâm Theo Trục Đứng)**

```python
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2  # Quan trọng cho fall detection
```

- **Ý nghĩa**: Vị trí đứng của người
- **Sử dụng**: Tính vận tốc rơi, phát hiện chạm đất

**2. Bbox Dimensions**

```python
width = x2 - x1
height = y2 - y1
```

**3. Aspect Ratio (Tỷ Lệ Khung)**

```python
aspect_ratio = width / height
```

- **AR < 1.0**: Tư thế đứng (chiều cao > chiều rộng)
- **AR ≈ 1.0**: Tư thế ngồi hoặc cúi
- **AR > 1.15**: Tư thế nằm (chiều rộng > chiều cao)
- **AR > 1.35**: Chắc chắn đang nằm

**Ví dụ**:

```
Đứng:       Nằm:
┌─┐         ┌────────┐
│ │         │        │
│ │         └────────┘
│ │         AR = 2.0
└─┘         → FALL!
AR = 0.5
→ NORMAL
```

---

#### 5.2.2 Body Angle (Góc Cơ Thể)

**Định nghĩa**: Góc giữa vector thân (vai → hông) và trục đứng (Y).

**Công thức**:

```python
# Bước 1: Tính điểm giữa vai
shoulder_mid = [
    (left_shoulder[x] + right_shoulder[x]) / 2,
    (left_shoulder[y] + right_shoulder[y]) / 2
]

# Bước 2: Tính điểm giữa hông
hip_mid = [
    (left_hip[x] + right_hip[x]) / 2,
    (left_hip[y] + right_hip[y]) / 2
]

# Bước 3: Vector thân
dx = shoulder_mid[x] - hip_mid[x]  # Độ lệch ngang
dy = shoulder_mid[y] - hip_mid[y]  # Độ lệch dọc

# Bước 4: Tính góc với trục đứng
angle_rad = arctan2(|dx|, |dy| + epsilon)
angle_deg = degrees(angle_rad)
```

**Minh họa**:

```
Đứng thẳng (0°):         Nghiêng (45°):        Nằm ngang (90°):

    ●● (vai)                  ●● (vai)             ●●────▓▓ (vai-hông)
    ▓▓                          ╲                 (nằm hoàn toàn)
    ▓▓ (thân)                    ╲ ▓▓
    ▓▓                            ╲
    ▓▓ (hông)                      ▓▓ (hông)

angle = 0°               angle = 45°           angle = 90°
→ NORMAL                 → SUSPECT             → FALL!
```

**Ngưỡng**:

- `angle < 48°`: Tư thế đứng/ngồi bình thường
- `48° ≤ angle < 55°`: Vùng cảnh báo
- `55° ≤ angle < 65°`: Có thể đang ngã
- `angle ≥ 65°`: Rất có thể đã ngã

**Ưu điểm**: Chính xác, không bị ảnh hưởng bởi kích thước người

**Nhược điểm**: Yêu cầu keypoints có confidence cao (≥ 0.5)

---

#### 5.2.3 Vertical Velocity (Vận Tốc Đứng)

**Định nghĩa**: Tốc độ di chuyển của tâm bbox theo trục Y (pixels/frame).

**1. Instantaneous dy (Vận tốc tức thời)**

```python
dy = current_frame.center_y - previous_frame.center_y
```

- **dy > 0**: Di chuyển xuống (Y tăng trong OpenCV)
- **dy < 0**: Di chuyển lên
- **|dy|**: Tốc độ di chuyển

**2. dy_velocity (Vận tốc trung bình)**

```python
# Lấy dy_window frames gần nhất (mặc định = 5)
dy_samples = [abs(frame.dy) for frame in last_5_frames]
dy_velocity = mean(dy_samples)
```

**3. dy_peak (Vận tốc đỉnh)**

```python
# Vận tốc lớn nhất trong cửa sổ
dy_peak = max(dy_samples)
```

**Ví dụ Minh Họa**:

```
Frame:    1    2    3    4    5    6    7    8
center_y: 100  102  105  110  118  130  148  172

dy:       -    2    3    5    8    12   18   24  (pixels/frame)
dy_vel:   -    2    2.5  3.3  4.6  7.0  10.6 13.4
dy_peak:  -    2    3    5    8    12   18   24

Phân tích:
- Frame 1-3: Chuyển động chậm, bình thường
- Frame 4-5: Bắt đầu tăng tốc (ngã?)
- Frame 6-8: Tăng tốc mạnh → ĐANG NGÃ!
```

**Ngưỡng Phát Hiện**:

```python
dy_fall_thres = 10.0      # Vận tốc nhanh = ngã
dy_peak_thres = 8.0       # Đỉnh vận tốc = va chạm
impact_dy_thres = 5.0     # Va chạm nhẹ
```

**Ý nghĩa Vật Lý**:

Khi người ngã:

1. **Giai đoạn rơi tự do**: dy tăng nhanh (trọng lực)
2. **Va chạm mặt đất**: dy_peak đạt cực đại
3. **Sau va chạm**: dy giảm về 0 (nằm yên)

---

#### 5.2.4 Border Integrity Check

**Mục đích**: Phát hiện khi bbox bị cắt ở mép ảnh.

**Vấn đề**: Khi người ở gần mép, bbox bị cắt → Aspect Ratio không chính xác.

```python
BORDER_MARGIN = 8  # pixels

is_touching_border = (
    x1 <= BORDER_MARGIN or           # Mép trái
    y1 <= BORDER_MARGIN or           # Mép trên
    x2 >= img_width - BORDER_MARGIN or   # Mép phải
    y2 >= img_height - BORDER_MARGIN     # Mép dưới
)
```

**Xử lý**:

- Nếu `is_touching_border = True` → Vô hiệu hóa Path 1 trong Rule Engine
- Chỉ dùng height drop và dy để phát hiện

---

## 6. Module 4: Rule Engine (Động Cơ Luật)

### 6.1 File: `urfd/rules.py`

#### Function 1: `check_fall_candidate()`

##### Chức năng

Kiểm tra frame hiện tại có phải **ứng viên té ngã** không.

##### Input

```python
check_fall_candidate(
    features,               # Dict features từ extraction
    angle_thres=55.0,      # Ngưỡng góc cơ thể
    ar_thres=1.15,         # Ngưỡng aspect ratio
    height_drop=0.15,      # Height drop hiện tại
    height_drop_thres=0.18,# Ngưỡng height drop
    dy_fall_thres=10.0,    # Ngưỡng vận tốc nhanh
    impact_dy_thres=5.0,   # Ngưỡng va chạm
    high_angle_thres=65.0  # Ngưỡng góc rất ngang
)
```

##### Output

```python
True   # Frame là fall candidate
False  # Frame không phải fall candidate
```

##### Logic: 4 Đường Dẫn (Paths) Song Song

**PATH 1: POSTURE-BASED DETECTION** (Phát hiện dựa trên tư thế)

```python
angle_ar_condition = (
    body_angle >= 55.0       AND
    aspect_ratio >= 1.15     AND
    dy_peak >= 5.0           AND
    NOT is_touching_border   # Vô hiệu hóa nếu mép ảnh
)
```

**Điều kiện**:

- Góc cơ thể nghiêng (≥ 55°)
- Tỷ lệ khung nằm ngang (AR ≥ 1.15)
- Có va chạm (dy_peak ≥ 5.0)

**Ví dụ**: Ngã sang một bên, cơ thể nằm ngang với va chạm nhẹ.

---

**PATH 2: HEIGHT DROP DETECTION** (Phát hiện giảm chiều cao)

```python
height_drop_condition = (height_drop >= 0.18)
```

**Công thức Height Drop**:

```python
# Lấy median của 12 frames gần nhất (trừ frame hiện tại)
baseline_height = median(last_12_heights[:-1])

# Tính tỷ lệ giảm
height_drop = (baseline_height - current_height) / baseline_height
height_drop = clip(height_drop, 0, 1)
```

**Ví dụ**:

```
Baseline: height = 200px (đứng)
Current:  height = 160px (nằm)
→ height_drop = (200 - 160) / 200 = 0.20 > 0.18
→ CANDIDATE!
```

**Ý nghĩa**: Người co lại → Có thể đã ngã.

---

**PATH 3: FAST MOTION DETECTION** (Phát hiện chuyển động nhanh)

```python
dy_condition = (dy >= 10.0)
```

**Ý nghĩa**: Vận tốc rơi xuống nhanh ≥ 10 pixels/frame.

**Ví dụ**:

```
Frame 5: center_y = 150
Frame 6: center_y = 162
→ dy = 12 > 10.0
→ CANDIDATE! (đang rơi nhanh)
```

---

**PATH 4: HIGH-ANGLE DETECTION** (Phát hiện góc rất nằm)

```python
high_angle_condition = (
    body_angle >= 65.0  AND
    dy_peak >= 3.0
)
```

**Mục đích**: Bắt các trường hợp ngã chậm, ngã về phía trước.

**Điều kiện**:

- Góc cơ thể rất ngang (≥ 65°)
- Có chút chuyển động (dy_peak ≥ 3.0)

**Ví dụ**: Người từ từ ngã xuống giường, yoga ngã chậm.

---

**Kết Luận Path**:

```python
is_candidate = (
    angle_ar_condition      OR
    height_drop_condition   OR
    dy_condition            OR
    high_angle_condition
)
```

Chỉ cần **một trong bốn** điều kiện đúng → Frame là candidate.

---

### 6.2 Function 2: `check_lying_posture()`

##### Chức năng

Kiểm tra frame hiện tại có tư thế **nằm** không (để xác nhận).

##### Input

```python
check_lying_posture(
    features,
    confirm_angle_thres=48.0,
    confirm_ar_thres=1.35
)
```

##### Logic: OR với Ngưỡng Relaxed

**Điều kiện 1: Góc Nằm (relaxed 10°)**

```python
angle_condition = (body_angle >= 48.0 - 10.0)
                = (body_angle >= 38.0)
```

**Điều kiện 2: AR Nằm (relaxed 0.20)**

```python
ar_condition = (aspect_ratio >= 1.35 - 0.20)
             = (aspect_ratio >= 1.15)
```

**Kết luận**:

```python
is_lying = angle_condition OR ar_condition
```

**Lý do Relaxed**:

- Tăng recall (bắt được nhiều fall hơn)
- Áp dụng ở bước xác nhận (đã qua candidate)
- Chấp nhận false positive tạm thời (FSM sẽ lọc)

---

## 7. Module 5: Finite State Machine (FSM)

### 7.1 File: `urfd/smoothing.py`

#### Class: `FallStateMachine`

##### Chức năng

Máy trạng thái để temporal smoothing và xác nhận té ngã qua thời gian.

##### 3 States

```
┌─────────┐
│ NORMAL  │ ← Trạng thái bình thường
└─────────┘
     │
     ├──(candidate frames ≥ 2)──→ ┌────────────┐
     │                             │ CANDIDATE  │
     │                             └────────────┘
     │                                   │
     │                                   ├──(lying frames ≥ 3 + impact)──→ ┌─────────────────┐
     │                                   │                                  │ FALL_CONFIRMED  │
     │                                   │                                  └─────────────────┘
     │                                   │                                          │
     │                                   ├──(recovery)──→ ┌─────────┐             │
     │                                                     │ NORMAL  │ ←───(recovery)
     │                                                     └─────────┘
     │
     └────────────────────────────────────────────────────────────────────────┘
```

---

### 7.2 State Variables (Biến Trạng Thái)

```python
self.state = "NORMAL"                  # Trạng thái hiện tại
self.candidate_history = []            # Lịch sử candidate frames [0,1,1,0,...]
self.confirm_history = []              # Lịch sử lying posture [0,0,1,1,1,...]
self.missing_streak = 0                # Số frame liên tục không có detection
self.height_history = []               # Lịch sử chiều cao bbox
self.score_accumulator = 0.0           # Điểm EMA (Exponential Moving Average)
self.fall_confirmed = False            # Cờ đã xác nhận fall
self.recovery_history = []             # Lịch sử recovery frames
self.dy_peak_history = []              # Lịch sử dy_peak
self.motion_settled = False            # Cờ chuyển động đã ổn định
```

---

### 7.3 Core Functions

#### 7.3.1 `_compute_height_drop(current_height)`

**Mục đích**: Tính normalized height drop.

**Thuật toán**:

```python
def _compute_height_drop(current_height):
    # Thêm vào history
    self.height_history.append(current_height)

    # Giữ lại tối đa height_window frames (12)
    if len(self.height_history) > 12:
        self.height_history = self.height_history[-12:]

    # Lấy baseline = median của history (trừ frame hiện tại)
    baseline_heights = self.height_history[:-1]
    median_height = median(baseline_heights)

    # Tính drop
    drop = (median_height - current_height) / median_height
    drop = clip(drop, 0, 1)

    return drop
```

**Ví dụ**:

```
History: [200, 198, 202, 199, 201, 160, ...]
                                    ^^^
Baseline = median([200, 198, 202, 199, 201]) = 200
Current = 160

drop = (200 - 160) / 200 = 0.20
```

---

#### 7.3.2 `_compute_frame_score(features, height_drop)`

**Mục đích**: Tính điểm fall cho frame hiện tại.

**Công thức Weighted Sum**:

```python
score = 0.0

# Component 1: Angle (30%)
if body_angle is not None:
    angle_norm = min(body_angle / 90.0, 1.0)
    score += 0.30 * angle_norm

# Component 2: Aspect Ratio (25%)
ar_norm = min(aspect_ratio / 2.0, 1.0)
score += 0.25 * ar_norm

# Component 3: Height Drop (25%)
score += 0.25 * height_drop

# Component 4: Vertical Velocity (20%)
if dy > 0:
    dy_norm = min(dy / 50.0, 1.0)
    score += 0.20 * dy_norm

return clip(score, 0, 1)
```

**Ví dụ**:

```
Frame có:
- body_angle = 72° → angle_norm = 72/90 = 0.8
- aspect_ratio = 1.8 → ar_norm = 1.8/2 = 0.9
- height_drop = 0.24
- dy = 15 → dy_norm = 15/50 = 0.3

score = 0.30*0.8 + 0.25*0.9 + 0.25*0.24 + 0.20*0.3
      = 0.24 + 0.225 + 0.06 + 0.06
      = 0.585
```

---

#### 7.3.3 `_check_slow_transition(features)`

**Mục đích**: Phát hiện chuyển động chậm (không phải ngã).

**Thuật toán**:

```python
def _check_slow_transition(features):
    # Lưu dy_peak vào history
    self.dy_peak_history.append(features["dy_peak"])

    # Giữ lại dy_long_window frames (10)
    if len(self.dy_peak_history) > 10:
        self.dy_peak_history.pop(0)

    # Kiểm tra max trong dy_short_window frames (3)
    if len(self.dy_peak_history) >= 3:
        max_dy_peak = max(self.dy_peak_history[-3:])

        # Nếu max < slow_lie_max_dy_peak (18.0)
        # → Chuyển động chậm (yoga, ngủ)
        return max_dy_peak < 18.0

    return False
```

**Ví dụ**:

```
dy_peak_history = [2.1, 3.5, 4.2, 2.8, 3.1, 5.0]
                                      ├────┬────┤
max_recent = max([2.8, 3.1, 5.0]) = 5.0

5.0 < 18.0 → True → Slow transition!
```

---

#### 7.3.4 `_check_recovery(features)`

**Mục đích**: Phát hiện người đứng dậy sau khi nằm (false positive).

**Thuật toán**:

```python
def _check_recovery(features):
    # Kiểm tra tư thế đứng via keypoints
    is_upright_keypoint = (body_angle < 55.0)

    # Kiểm tra tư thế đứng via bbox
    is_upright_bbox = (aspect_ratio < 1.25)

    is_upright = is_upright_keypoint OR is_upright_bbox

    # Lưu lịch sử (1 = upright, 0 = lying)
    self.recovery_history.append(1 if is_upright else 0)
    if len(self.recovery_history) > 10:
        self.recovery_history.pop(0)

    # Kiểm tra chiều cao đã phục hồi chưa
    height_recovered = False
    if len(self.height_history) >= 10:
        recent_max = max(self.height_history[-10:])
        baseline = median(self.height_history[:30])
        if recent_max / baseline > 0.70:
            height_recovered = True

    # Logic khác nhau theo state
    if self.state == "FALL_CONFIRMED":
        # Quick recovery: 3/4 frames upright
        if len(self.recovery_history) >= 4:
            if sum(self.recovery_history[-4:]) >= 3 and height_recovered:
                return True
    else:
        # Standard recovery: 1/3 của window
        if len(self.recovery_history) >= 5:
            if sum(self.recovery_history[-5:]) >= 2 and height_recovered:
                return True

    return False
```

---

### 7.4 Main Function: `update(features)`

**Chức năng**: Cập nhật FSM với frame mới.

**Input**: `features` dict từ `compute_frame_features()`

**Output**: `(state, score)` tuple

**Luồng Xử Lý Chi Tiết**:

```python
def update(self, features):
    # ========================
    # BƯỚC 1: XỬ LÝ MISSING DETECTION
    # ========================
    if features["bbox"] is None:
        self.missing_streak += 1
        if self.missing_streak > 5:  # max_missing_streak
            self.score_accumulator *= 0.5  # Decay score
        return self.state, self.score_accumulator
    else:
        self.missing_streak = 0

    # ========================
    # BƯỚC 2: TÍNH METRICS
    # ========================
    height_drop = self._compute_height_drop(features["height"])
    frame_score = self._compute_frame_score(features, height_drop)

    # Cập nhật score bằng EMA
    decay = 0.85  # score_decay
    self.score_accumulator = (
        decay * self.score_accumulator +
        (1 - decay) * frame_score
    )

    # ========================
    # BƯỚC 3: KIỂM TRA CANDIDATE VÀ LYING
    # ========================
    is_candidate = check_fall_candidate(
        features, angle_thres=55.0, ar_thres=1.15,
        height_drop, height_drop_thres=0.18,
        dy_fall_thres=10.0, impact_dy_thres=5.0
    )

    is_lying = check_lying_posture(
        features, confirm_angle_thres=48.0,
        confirm_ar_thres=1.35
    )

    # ========================
    # BƯỚC 4: STATE TRANSITIONS
    # ========================

    # ─────── STATE: NORMAL ───────
    if self.state == "NORMAL":
        # Lưu candidate history
        self.candidate_history.append(1 if is_candidate else 0)
        if len(self.candidate_history) > 3:  # cand_enter_frames + tolerance
            self.candidate_history.pop(0)

        # Đếm số candidate frames
        candidate_count = sum(self.candidate_history)

        # Chuyển sang CANDIDATE nếu đủ frames (≥ 2)
        if candidate_count >= 2:  # cand_enter_frames
            self.state = "CANDIDATE"
            self.confirm_history = []
            self.score_accumulator += 0.05  # score_boost_candidate

    # ─────── STATE: CANDIDATE ───────
    elif self.state == "CANDIDATE":
        # Lưu lying history
        self.confirm_history.append(1 if is_lying else 0)
        if len(self.confirm_history) > 10:  # confirm_window
            self.confirm_history.pop(0)

        lying_count = sum(self.confirm_history)

        # Kiểm tra recovery
        has_recovered = self._check_recovery(features)

        if has_recovered:
            # RECOVERY → Reset về NORMAL
            self.state = "NORMAL"
            self.candidate_history = []
            self.score_accumulator *= 0.2
            return self.state, self.score_accumulator

        # Kiểm tra điều kiện xác nhận
        if lying_count >= 3:  # confirm_frames
            dy_peak = features["dy_peak"]

            # Điều chỉnh ngưỡng dy nếu có height drop
            effective_dy_thres = 8.0  # dy_peak_thres
            if height_drop >= 0.22:
                effective_dy_thres *= 0.70  # Relax 30%

            # 3 paths để confirm
            fast_motion = (dy_peak >= effective_dy_thres)
            strong_height = (height_drop >= 0.28 and lying_count >= 3)
            moderate_height = (height_drop >= 0.22 and lying_count >= 6)

            can_confirm = fast_motion or strong_height or moderate_height

            if can_confirm:
                # CONFIRM FALL
                self.state = "FALL_CONFIRMED"
                self.fall_confirmed = True
                self.score_accumulator = 1.0  # score_boost_confirmed

        # Timeout: không confirm được
        elif not is_candidate and len(self.confirm_history) >= 10:
            if lying_count < 1:  # confirm_frames - tolerance
                # Reset về NORMAL
                self.state = "NORMAL"
                self.candidate_history = []

    # ─────── STATE: FALL_CONFIRMED ───────
    elif self.state == "FALL_CONFIRMED":
        # Kiểm tra recovery
        has_recovered = self._check_recovery(features)

        if has_recovered:
            # RECOVERY → Reset toàn bộ
            self.state = "NORMAL"
            self.fall_confirmed = False
            self.score_accumulator = 0.0
            self.candidate_history = []
            self.confirm_history = []
            self.recovery_history = []
        else:
            # Duy trì confirmed
            self.score_accumulator = 1.0

    # ========================
    # BƯỚC 5: RETURN
    # ========================
    self.score_accumulator = clip(self.score_accumulator, 0, 1)
    return self.state, self.score_accumulator
```

---

### 7.5 Ví Dụ Hoạt Động FSM

#### Scenario 1: True Fall (Ngã Thật)

```
Frame | dy_peak | body_angle | AR   | is_candidate | is_lying | State           | Score
------|---------|------------|------|--------------|----------|-----------------|------
1     | 1.2     | 15°        | 0.48 | False        | False    | NORMAL          | 0.05
2     | 1.5     | 18°        | 0.52 | False        | False    | NORMAL          | 0.06
3     | 3.2     | 22°        | 0.61 | False        | False    | NORMAL          | 0.08
4     | 8.5     | 42°        | 0.98 | True ✓       | False    | NORMAL          | 0.15
5     | 15.2    | 58°        | 1.35 | True ✓       | True ✓   | CANDIDATE       | 0.35
6     | 12.1    | 68°        | 1.62 | True ✓       | True ✓   | CANDIDATE       | 0.52
7     | 8.2     | 71°        | 1.71 | True ✓       | True ✓   | FALL_CONFIRMED  | 1.00
8     | 3.1     | 73°        | 1.75 | True ✓       | True ✓   | FALL_CONFIRMED  | 1.00
9     | 2.0     | 72°        | 1.73 | False        | True ✓   | FALL_CONFIRMED  | 1.00
...
```

**Phân tích**:

- Frame 1-3: Bình thường
- Frame 4: Bắt đầu có dy_peak cao → candidate
- Frame 5: Góc tăng, AR tăng → vẫn candidate
- Frame 6: Đủ 3 lying frames + dy_peak cao → CONFIRM
- Frame 7+: Duy trì FALL_CONFIRMED

---

#### Scenario 2: False Positive - Yoga (Nằm Chậm)

```
Frame | dy_peak | body_angle | AR   | is_candidate | is_lying | State      | Score | Notes
------|---------|------------|------|--------------|----------|------------|-------|-------
1     | 0.8     | 12°        | 0.45 | False        | False    | NORMAL     | 0.03  |
2     | 1.2     | 18°        | 0.52 | False        | False    | NORMAL     | 0.04  |
3     | 2.5     | 28°        | 0.78 | False        | False    | NORMAL     | 0.08  |
4     | 3.1     | 38°        | 1.05 | False        | True     | NORMAL     | 0.12  | is_lying via relaxed AR
5     | 4.2     | 45°        | 1.25 | False        | True     | NORMAL     | 0.18  | Chưa đủ candidate
6     | 4.8     | 52°        | 1.42 | True ✓       | True     | NORMAL     | 0.24  | height_drop > 0.18
7     | 5.1     | 58°        | 1.55 | True ✓       | True     | CANDIDATE  | 0.32  | ≥ 2 candidate frames
8     | 4.9     | 61°        | 1.62 | True ✓       | True     | CANDIDATE  | 0.38  | lying_count = 3
9     | 4.5     | 63°        | 1.68 | True ✓       | True     | CANDIDATE  | 0.42  | BUT: dy_peak < 8.0
10    | 4.0     | 64°        | 1.70 | True ✓       | True     | CANDIDATE  | 0.45  | slow_transition = True
11    | 3.8     | 65°        | 1.72 | True ✓       | True     | CANDIDATE  | 0.47  | Vẫn chưa confirm
12    | 12.5    | 28°        | 0.82 | False        | False    | NORMAL     | 0.15  | RECOVERY! Đứng dậy
```

**Phân tích**:

- Frame 1-6: Từ từ nằm xuống
- Frame 7: Vào CANDIDATE (height drop)
- Frame 8-11: Đủ lying frames NHƯNG dy_peak < 8.0 (slow transition)
  → Không confirm được
- Frame 12: Người đứng lên → Recovery → Reset NORMAL

**Lý do không confirm**:

```python
lying_count >= 3       → True (có 5 frames)
fast_motion            → False (dy_peak = 5.1 < 8.0)
strong_height          → False (height_drop = 0.20 < 0.28)
moderate_height        → False (lying_count = 5 < 6)
→ can_confirm = False
```

---

#### Scenario 3: False Positive - Ngồi Xuống Ghế

```
Frame | dy_peak | body_angle | AR   | height_drop | is_candidate | is_lying | State      | Notes
------|---------|------------|------|-------------|--------------|----------|------------|-------
1     | 1.0     | 10°        | 0.42 | 0.0         | False        | False    | NORMAL     |
2     | 2.5     | 12°        | 0.45 | 0.0         | False        | False    | NORMAL     |
3     | 6.8     | 15°        | 0.48 | 0.05        | False        | False    | NORMAL     | Bắt đầu ngồi
4     | 11.2    | 18°        | 0.52 | 0.12        | True ✓       | False    | NORMAL     | dy > 10.0 (Path 3)
5     | 13.5    | 22°        | 0.58 | 0.20        | True ✓       | False    | CANDIDATE  | height_drop > 0.18 (Path 2)
6     | 8.2     | 25°        | 0.65 | 0.28        | True ✓       | False    | CANDIDATE  | Ngồi xuống ghế
7     | 3.1     | 28°        | 0.72 | 0.30        | True ✓       | False    | CANDIDATE  | lying_count = 0
8     | 1.5     | 30°        | 0.75 | 0.32        | False        | False    | NORMAL     | Timeout, reset
9     | 0.8     | 32°        | 0.78 | 0.32        | False        | False    | NORMAL     | Ngồi ổn định
```

**Phân tích**:

- Frame 3-6: Chuyển động ngồi xuống nhanh
- Frame 4-5: Trigger candidate (dy + height_drop)
- Frame 6-7: Nhưng không có lying posture (AR < 1.15, angle < 48°)
- Frame 8: Timeout → Reset NORMAL

**Lý do không confirm**:

```python
lying_count >= 3  → False (có 0 frames lying)
→ Không thể confirm
```

---

## 8. Các Tham Số Cấu Hình

### 8.1 File: `configs/default.yaml`

#### 8.1.1 YOLO Model

```yaml
yolo_model: 'yolov8s-pose.pt' # Model weights
imgsz: 640 # Kích thước inference
conf_thres: 0.25 # Confidence threshold
iou_thres: 0.45 # NMS IoU threshold
keypoint_conf_thres: 0.5 # Keypoint confidence
```

#### 8.1.2 Preprocessing

```yaml
preprocess_lowlight: true # Bật low-light processing
gamma: 1.3 # Gamma correction
clahe_clip: 2.0 # CLAHE clip limit
clahe_grid: 8 # CLAHE grid size
```

#### 8.1.3 Fall Detection Rules

```yaml
# Candidate detection
angle_thres: 55.0 # Góc cơ thể candidate
ar_thres: 1.15 # Aspect ratio candidate
high_angle_thres: 65.0 # Góc rất ngang

# Confirmation
confirm_angle_thres: 48.0 # Góc xác nhận (relaxed)
confirm_ar_thres: 1.35 # AR xác nhận

# Height drop
height_drop_thres: 0.18 # Ngưỡng height drop
height_drop_thres_strong: 0.28 # Ngưỡng mạnh

# Velocity
dy_fall_thres: 10.0 # Vận tốc ngã nhanh
dy_peak_thres: 8.0 # Đỉnh vận tốc
impact_dy_thres: 5.0 # Va chạm nhẹ

# Windows
dy_window: 5 # Cửa sổ tính dy
height_window: 12 # Cửa sổ height drop
```

#### 8.1.4 Temporal Smoothing (FSM)

```yaml
# Candidate transition
cand_enter_frames: 2 # Số frames để vào CANDIDATE
cand_tolerance: 1 # Dung sai

# Confirmation transition
confirm_frames: 3 # Số frames lying để confirm
confirm_tolerance: 2 # Dung sai
confirm_window: 10 # Cửa sổ xác nhận

# Recovery
recovery_window: 10 # Cửa sổ recovery
recovery_upright_angle_thres: 55.0
recovery_ar_thres: 1.25
recovery_height_recover_ratio: 0.70

# Slow transition gate
slow_lie_max_dy_peak: 18.0 # Ngưỡng dy chậm
dy_short_window: 3
dy_long_window: 10

# Motion settling
motion_settle_window: 8
settle_dy_abs_thres: 10.0

# Scoring
score_decay: 0.85 # EMA decay
score_boost_candidate: 0.05 # Boost khi vào CANDIDATE
score_boost_confirmed: 1.0 # Set 1.0 khi CONFIRMED
```

#### 8.1.5 Tracking

```yaml
# Primary person tracking
track_iou_thres: 0.3
track_center_dist_thres: 1.5
track_max_missing: 8

# Fallback tracking
fallback_tracker_type: 'kcf' # KCF tracker
fallback_track_max_gap: 6
```

#### 8.1.6 Output

```yaml
output_fps: 25
output_dir: 'outputs'
min_confirm_duration_frames: 2 # Tối thiểu FALL_CONFIRMED liên tục
seed: 42 # Reproducibility
```

---

## 9. Luồng Dữ Liệu Đầy Đủ

### 9.1 Sơ Đồ Chi Tiết

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT VIDEO FRAMES                         │
│                     (640x480 BGR images)                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   YOLO POSE DETECTION                             │
│  - Input:  BGR image (640x480)                                    │
│  - Process: YOLOv8s-pose inference                                │
│  - Output: List of detections                                     │
│      [{                                                           │
│        "bbox": [x1,y1,x2,y2],                                     │
│        "conf": 0.87,                                              │
│        "keypoints": [[x,y,c], ...] (17 points)                    │
│      }]                                                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MULTI-PERSON TRACKING                          │
│  - Input:  Detections list                                        │
│  - Process:                                                       │
│      1. Compute cost matrix (IoU + center distance)               │
│      2. Greedy matching                                           │
│      3. Update tracks, create new, delete stale                   │
│  - Output: {track_id: detection_index}                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FEATURES EXTRACTION                             │
│  - Input:  Detection, track_id, history                           │
│  - Process:                                                       │
│      1. Bbox features (center_y, width, height, AR)               │
│      2. Keypoint-based angle (shoulder-hip vector)                │
│      3. Velocity (dy, dy_velocity, dy_peak)                       │
│      4. Border integrity check                                    │
│  - Output: features dict                                          │
│      {                                                            │
│        "track_id": 1,                                             │
│        "bbox": [x1,y1,x2,y2],                                     │
│        "center_y": 285.5,                                         │
│        "height": 198.2,                                           │
│        "width": 87.3,                                             │
│        "bbox_aspect_ratio": 0.44,                                 │
│        "shoulder_mid": [320, 180],                                │
│        "hip_mid": [318, 250],                                     │
│        "body_angle_deg": 12.5,                                    │
│        "feature_valid": True,                                     │
│        "dy": 2.3,                                                 │
│        "dy_velocity": 3.1,                                        │
│        "dy_peak": 5.8,                                            │
│        "is_touching_border": False                                │
│      }                                                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       RULE ENGINE                                 │
│  - Input:  features dict, config thresholds                       │
│  - Process:                                                       │
│      1. check_fall_candidate() → 4 paths                          │
│         - Path 1: Posture (angle + AR + impact)                   │
│         - Path 2: Height drop                                     │
│         - Path 3: Fast motion (dy)                                │
│         - Path 4: High angle                                      │
│      2. check_lying_posture() → OR logic                          │
│         - Angle ≥ 38° OR AR ≥ 1.15                                │
│  - Output:                                                        │
│      is_candidate: bool                                           │
│      is_lying: bool                                               │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FINITE STATE MACHINE                            │
│  - Input:  features, is_candidate, is_lying                       │
│  - State Variables:                                               │
│      - state: "NORMAL" | "CANDIDATE" | "FALL_CONFIRMED"           │
│      - candidate_history: [0,1,1,0,...]                           │
│      - confirm_history: [0,0,1,1,1,...]                           │
│      - score_accumulator: float [0, 1]                            │
│      - height_history, recovery_history, ...                      │
│  - Process:                                                       │
│      1. Handle missing detections                                 │
│      2. Compute height_drop, frame_score                          │
│      3. Update score (EMA)                                        │
│      4. State transitions:                                        │
│         NORMAL → CANDIDATE: candidate_count ≥ 2                   │
│         CANDIDATE → FALL_CONFIRMED:                               │
│            lying_count ≥ 3 AND                                    │
│            (dy_peak ≥ 8.0 OR height_drop ≥ 0.28)                  │
│         * → NORMAL: recovery detected                             │
│  - Output:                                                        │
│      state: string                                                │
│      score: float [0, 1]                                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FINAL PREDICTION                               │
│  - Input:  all_states list for entire video                       │
│  - Process:                                                       │
│      1. Check if any state == "FALL_CONFIRMED"                    │
│      2. Count max consecutive FALL_CONFIRMED                      │
│      3. If max_consecutive ≥ min_confirm_duration_frames (2)      │
│         → pred_label = 1 (FALL)                                   │
│      4. Else → pred_label = 0 (NORMAL)                            │
│  - Output:                                                        │
│      pred_label: 0 or 1                                           │
│      first_confirm_frame: int                                     │
│      max_score: float                                             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     OUTPUT VIDEO                                  │
│  - Overlay bounding boxes                                         │
│  - State labels (NORMAL/CANDIDATE/FALL_CONFIRMED)                 │
│  - Confidence scores                                              │
│  - Keypoint skeleton                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

### 9.2 Ví Dụ Trace Đầy Đủ Một Frame

**Frame #5 của video fall-01**

#### Input Frame

```
Image: 640x480 BGR
Person detected: 1 person
```

#### YOLO Detection Output

```python
detections = [
    {
        "bbox": [245, 128, 412, 398],
        "conf": 0.89,
        "bbox_area": 45090,
        "keypoints": [
            [328, 145, 0.92],  # 0: Nose
            [335, 142, 0.88],  # 1: Left Eye
            [321, 142, 0.91],  # 2: Right Eye
            ...,
            [285, 185, 0.78],  # 5: Left Shoulder
            [371, 188, 0.82],  # 6: Right Shoulder
            ...,
            [292, 268, 0.71],  # 11: Left Hip
            [364, 271, 0.75],  # 12: Right Hip
            ...
        ]
    }
]
```

#### Tracking Output

```python
matches = {
    track_id=1: detection_index=0
}

track_1 = {
    "bbox": [245, 128, 412, 398],
    "center": [328.5, 263],
    "velocity": [0.8, 2.3],
    "last_seen": 5,
    "missing_count": 0
}
```

#### Features Extraction

**Step 1: Bbox features**

```python
x1, y1, x2, y2 = 245, 128, 412, 398
cx = (245 + 412) / 2 = 328.5
cy = (128 + 398) / 2 = 263
w = 412 - 245 = 167
h = 398 - 128 = 270
aspect_ratio = 167 / 270 = 0.618
```

**Step 2: Keypoints angle**

```python
left_shoulder = [285, 185, 0.78]
right_shoulder = [371, 188, 0.82]
left_hip = [292, 268, 0.71]
right_hip = [364, 271, 0.75]

# All confidence > 0.5 → valid
shoulder_mid = [(285+371)/2, (185+188)/2] = [328, 186.5]
hip_mid = [(292+364)/2, (268+271)/2] = [328, 269.5]

dx = abs(328 - 328) = 0
dy = abs(186.5 - 269.5) = 83

angle_rad = arctan2(0, 83) = 0
angle_deg = 0° → Standing upright
```

**Step 3: Velocity**

```python
# Previous frame center_y = 260.7
dy = 263 - 260.7 = 2.3

# History dy: [1.8, 2.1, 2.5, 2.2, 2.3]
dy_velocity = mean([1.8, 2.1, 2.5, 2.2, 2.3]) = 2.18
dy_peak = max([1.8, 2.1, 2.5, 2.2, 2.3]) = 2.5
```

**Step 4: Border check**

```python
img_w, img_h = 640, 480
BORDER_MARGIN = 8

is_touching_border = (
    245 <= 8 or         # False
    128 <= 8 or         # False
    412 >= 632 or       # False
    398 >= 472          # False
) = False
```

**Features Output**:

```python
features = {
    "track_id": 1,
    "bbox": [245, 128, 412, 398],
    "center_y": 263,
    "height": 270,
    "width": 167,
    "bbox_aspect_ratio": 0.618,
    "shoulder_mid": [328, 186.5],
    "hip_mid": [328, 269.5],
    "body_angle_deg": 0.0,
    "feature_valid": True,
    "dy": 2.3,
    "dy_velocity": 2.18,
    "dy_peak": 2.5,
    "is_touching_border": False
}
```

#### Rule Engine

**check_fall_candidate()**:

```python
# PATH 1: Posture
angle_ar_condition = (
    0.0 >= 55.0 AND          # False
    0.618 >= 1.15 AND        # False
    2.5 >= 5.0 AND           # False
    NOT False
) = False

# PATH 2: Height drop
height_drop = 0.05           # Computed from history
height_drop_condition = (0.05 >= 0.18) = False

# PATH 3: Fast motion
dy_condition = (2.3 >= 10.0) = False

# PATH 4: High angle
high_angle_condition = (
    0.0 >= 65.0 AND          # False
    2.5 >= 3.0               # False
) = False

→ is_candidate = False
```

**check_lying_posture()**:

```python
angle_condition = (0.0 >= 38.0) = False
ar_condition = (0.618 >= 1.15) = False

→ is_lying = False
```

#### FSM Update

```python
state = "NORMAL"  # Current state
score_accumulator = 0.08

# Missing detection?
features["bbox"] is not None → No

# Compute metrics
height_drop = 0.05
frame_score = (
    0.3 * (0/90) +           # angle
    0.25 * (0.618/2) +       # AR
    0.25 * 0.05 +            # height_drop
    0.2 * (2.3/50)           # dy
) = 0 + 0.077 + 0.0125 + 0.0092 = 0.099

# EMA update
score_accumulator = 0.85 * 0.08 + 0.15 * 0.099
                  = 0.068 + 0.015 = 0.083

# State transition
is_candidate = False
→ No transition, stay NORMAL

# Update history
candidate_history.append(0)  # Not candidate
→ candidate_history = [0, 0, 0]

→ Output: ("NORMAL", 0.083)
```

---

## 10. Tóm Tắt Kỹ Thuật

### 10.1 Các Đại Lượng Vật Lý Chính

| Đại Lượng        | Công Thức               | Ý Nghĩa            | Ngưỡng                |
| ---------------- | ----------------------- | ------------------ | --------------------- |
| **Aspect Ratio** | `w / h`                 | Tỷ lệ nằm/đứng     | > 1.15 = nằm          |
| **Body Angle**   | `arctan2(dx, dy)`       | Góc nghiêng cơ thể | > 55° = ngã           |
| **dy**           | `cy_t - cy_(t-1)`       | Vận tốc rơi        | > 10 px/f = ngã nhanh |
| **dy_peak**      | `max(dy_window)`        | Đỉnh vận tốc       | > 8 = va chạm         |
| **height_drop**  | `(median - h) / median` | Giảm chiều cao     | > 0.18 = co lại       |

### 10.2 Các Bước Pipeline

1. **Detection**: YOLO → 17 keypoints + bbox
2. **Tracking**: IoU matching → track_id
3. **Extraction**: Tính các đại lượng vật lý
4. **Rule Engine**: 4 paths OR logic → candidate
5. **FSM**: Temporal smoothing → FALL_CONFIRMED

### 10.3 Điểm Mạnh

✅ **Multi-path detection**: Bắt được nhiều loại té ngã
✅ **Temporal smoothing**: Loại bỏ false positive
✅ **Border-aware**: Xử lý bbox bị cắt
✅ **Recovery detection**: Tránh báo sai kéo dài
✅ **Configurable**: Dễ dàng tune parameters

### 10.4 Hạn Chế

❌ **Chậm reaction**: Cần 2-3 frames để confirm (0.1s @ 25fps)
❌ **Multi-person**: Chỉ track 1 người chính
❌ **Occlusion**: Yếu khi bị che khuất
❌ **Lighting**: Cần preprocessing cho low-light

---

## 11. Kết Luận

Hệ thống sử dụng một pipeline chặt chẽ kết hợp:

- **Computer Vision** (YOLO Pose)
- **Physics-based features** (góc, vận tốc, tỷ lệ)
- **Rule-based logic** (4 paths)
- **State machine** (temporal coherence)

Để đạt được khả năng phát hiện té ngã **chính xác** và **robust** trong điều kiện thực tế.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-25
**Author**: AI Technical Writer
