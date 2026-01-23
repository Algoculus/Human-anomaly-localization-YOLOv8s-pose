# Giải Thích Chi Tiết Source Code

Tài liệu này cung cấp cái nhìn sâu sắc vào từng module của source code, giải thích chức năng của các hàm và ý nghĩa của các tham số cấu hình trong `src/urfd/`.

## 1. Module Track & Detect ([tracking.py], [yolo_pose.py])

### `class YOLOPoseDetector` ([yolo_pose.py])

Wrapper cho mô hình YOLOv8.

- **[__init__]**:
    - `conf_thres=0.25`: Ngưỡng tự tin tối thiểu để chấp nhận một detection.
    - `iou_thres=0.45`: Ngưỡng NMS (Non-Maximum Suppression) để loại bỏ các box trùng lặp.
    - `clahe_clip=2.0`, `clahe_grid=8`: Tham số cho thuật toán cân bằng histogram thích ứng, dùng để tăng sáng ảnh tối.
- **[detect(image)]**:
    - Trả về list các dict: `{'bbox': [x1,y1,x2,y2], 'keypoints': [[x,y,conf], ...], ...}`.

### `class MultiPersonTracker` ([tracking.py])

Bộ theo dõi đa đối tượng sử dụng thuật toán gán ID đơn giản nhưng hiệu quả.

- **[update(detections, ...)]**:
    - Dự báo vị trí (Prediction): $Pos_{new} = Pos_{old} + Velocity_{avg}$.
    - Tính Cost Matrix: Kết hợp IoU và khoảng cách tâm.
    - **Logic Missing**: Nếu một track không được match trong `track_max_missing` (8 frames), nó sẽ bị xóa khỏi bộ nhớ.

---

## 2. Module Logic Cốt Lõi ([features.py], [rules.py], [smoothing.py])

Đây là "bộ não" của hệ thống, nơi các quyết định được đưa ra.

### [features.py]

- **[compute_frame_features]**:
    - Tính toán vector đặc trưng vật lý.
    - `dy_window=5`: Cửa sổ thời gian dùng để tìm vận tốc đỉnh ($dy_{peak}$).
    - Xử lý logic `feature_valid`: Chỉ coi góc cơ thể là hợp lệ nếu độ tự tin của 2 vai và 2 hông đều > `keypoint_conf_thres` (0.5).

### [rules.py]

Chứa các ngưỡng cứng (Hard Thresholds) được định nghĩa trong config.

- **[check_fall_candidate]**: Kiểm tra sơ bộ.
    - `angle_thres=55.0`: Nếu nghiêng quá 55 độ -> Nghi vấn.
    - `ar_thres=1.15`: Nếu BBox hơi bè ra -> Nghi vấn.
    - `dy_fall_thres=13.5`: Nếu rơi nhanh hơn 13.5 px/frame -> Nghi vấn.
- **[check_lying_posture]**: Kiểm tra tư thế nằm (ngưỡng thấp hơn để tăng Recall).
    - `confirm_angle_thres=48.0`.
    - `confirm_ar_thres=1.35`.

### `class FallStateMachine` ([smoothing.py])

Máy trạng thái quản lý sự ổn định của quyết định.

- **Các biến trạng thái**:
    - `state`: Trạng thái hiện tại (NORMAL/CANDIDATE/FALL_CONFIRMED).
    - `candidate_history`: Bộ đệm lưu kết quả check candidate gần đây.
    - `score_accumulator`: Điểm số tích lũy (Exponential Moving Average).
- **[update(features)]**:
    - Tính height_drop dựa trên trung vị của `height_window` (12 frames) gần nhất.
    - Cập nhật score: `score = 0.85 * score + 0.15 * new_score`.
    - Xử lý chuyển trạng thái (State Transition) như mô tả trong tài liệu Pipeline.
- **[_check_recovery]**:
    - Kiểm tra nếu người đứng dậy (Upright) trong `recovery_window` (10 frames).
    - Yêu cầu: Góc < 55 độ HOẶC Tỷ lệ chiều cao hồi phục > 70% (`recovery_height_recover_ratio=0.7`).

---

## 3. Module Đánh Giá ([eval.py], [dataset.py])

### [dataset.py]

- **[load_sequence_frames]**:
    - Hỗ trợ đệ quy tìm kiếm ảnh.
    - Sắp xếp file theo số: `lambda p: int(filter(isdigit, p))` để tránh lỗi thứ tự `1, 10, 2`.

### [eval.py]

- **[compute_metrics]**:
    - Chuyển input list thành numpy array.
    - Tính TP, TN, FP, FN nhanh chóng bằng boolean indexing.
    - Xử lý chia cho 0 (division by zero) bằng cách kiểm tra mẫu số > 0.
- **[plot_confusion_matrix]**: Sử dụng `seaborn.heatmap` để vẽ ma trận màu xanh (Blues).

---

## 4. Cấu hình ([configs/default.yaml])

Đây là nơi tập trung toàn bộ tham số của hệ thống.

- **Nhóm Threshold**: `angle_thres`, `ar_thres`, `dy_fall_thres` quyết định độ nhạy của việc bắt ngã.
- **Nhóm Temporal**: `confirm_frames`, `confirm_window` quyết định thời gian cần thiết để khẳng định một cú ngã (tránh báo giả do nhiễu).
- **Nhóm Recovery**: `recovery_window` quyết định tốc độ hệ thống reset về trạng thái bình thường.
