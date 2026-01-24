# BÁO CÁO BÀI TẬP LỚN

## Hệ Thống Phát Hiện Ngã (Fall Detection) sử dụng YOLOv8 Pose Estimation

## Tóm tắt (Abstract)

Bài báo cáo trình bày quá trình xây dựng hệ thống phát hiện ngã tự động dựa trên thị giác máy tính, giải quyết vấn đề giám sát an toàn cho người cao tuổi. Chúng tôi đề xuất phương pháp tiếp cận hai giai đoạn: (1) Trích xuất khung xương (Pose Extraction) thời gian thực sử dụng **YOLOv8s-pose**; (2) Phân tích hành vi bằng **Finite State Machine (FSM)** dựa trên các đặc trưng động học.

**Đặc điểm nổi bật**:
- **4-Path Detection**: Kết hợp 4 đường dẫn phát hiện (Posture+Impact, Height Drop, Fast Motion, High-Angle).
- **Border Integrity Check**: Vô hiệu hóa AR khi BBox bị cắt ở biên ảnh.
- **Impact Verification**: Yêu cầu $dy_{peak} \geq 5$ để giảm FP từ yoga/ngủ.

Hệ thống được kiểm thử trên bộ dữ liệu chuẩn **URFD** (70 sequences), đạt **Accuracy 90.00%**, **Precision 92.59%**, **Recall 83.33%**, **Specificity 95%**, và **F1-Score 87.72%**.

---

## 1. Giới thiệu (Introduction)

* **Bối cảnh**: Té ngã là rủi ro lớn đối với người cao tuổi sống một mình.
* **Mục tiêu**: Phát triển hệ thống AI phát hiện ngã tự động từ camera giám sát thông thường.
* **Đóng góp**:
    *   Tích hợp YOLOv8-pose tiên tiến cho bài toán Fall Detection.
    *   Đề xuất thuật toán FSM lọc nhiễu hiệu quả với 4 đường dẫn phát hiện.
    *   **Path 4 (High-Angle Detection)**: Phát hiện ngã chậm và ngã trực diện.
    *   Cung cấp bộ mã nguồn hoàn chỉnh với module đánh giá chi tiết.

---

## 2. Tổng quan lý thuyết (Related Work / Background)

* **YOLOv8**: Phiên bản mới nhất của dòng YOLO, cung cấp khả năng phát hiện keypoints với độ chính xác cao và tốc độ thời gian thực ($>30$ FPS).
* **Phương pháp phát hiện ngã**:
    *   *Dựa trên ngoại hình (Appearance-based)*: CNN phân loại ảnh trực tiếp.
    *   *Dựa trên dáng bộ (Pose-based)*: Trích xuất khung xương rồi phân tích.

---

## 3. Dataset & Tiền xử lý dữ liệu

### 3.1 Dataset: UR Fall Detection Dataset (URFD)
* **Nguồn**: University of Rzeszow.
* **Số lượng**: 70 video clips (30 Falls + 40 ADL).

### 3.2 Tiền xử lý (Preprocessing)
* **Gamma Correction & CLAHE**: Cân bằng sáng cục bộ trên kênh Luma.
* **Resize**: Chuẩn hóa ảnh về $640 \times 640$ pixel.

---

## 4. Phương pháp đề xuất (Methodology)

### 4.1 Kiến trúc Pipeline
`Input Frame` → `Preprocessing` → `Pose Estimation` → `Tracking` → `Feature Extraction` → `State Machine` → `Alert`

### 4.2 4-Path Fall Detection Rules
| Path | Điều kiện | Mục đích |
|------|-----------|----------|
| **Path 1** | $\theta \geq 55°$ AND $AR \geq 1.15$ AND $dy_{peak} \geq 5$ | Posture + Impact |
| **Path 2** | $Drop \geq 0.18$ | Height Drop |
| **Path 3** | $dy \geq 10$ px/frame | Fast Motion |
| **Path 4** | $\theta \geq 65°$ AND $dy_{peak} \geq 3$ | High-Angle (slow/frontal falls) |

### 4.3 Máy trạng thái (State Machine)
3 trạng thái: `NORMAL` → `CANDIDATE` → `FALL_CONFIRMED`

---

## 5. Thực nghiệm & Kết quả

### 5.1 Thiết lập thực nghiệm
* **Model**: YOLOv8s-pose.
* **Criterion**: Sequence được coi là Fall nếu `FALL_CONFIRMED` duy trì $\ge 2$ frames.

### 5.2 Kết quả định lượng

| Metric | Giá trị (%) | Ý nghĩa |
| :--- | :--- | :--- |
| **Accuracy** | **90.00** | Độ chính xác tổng thể tốt. |
| **Precision** | **92.59** | Ít báo động giả. |
| **Recall** | **83.33** | Phát hiện được đa số các ca ngã. |
| **Specificity** | **95.00** | Phân biệt rất tốt hành động thường ngày. |
| **F1-Score** | **87.72** | Cân bằng tốt. |

**Confusion Matrix (70 sequences)**:
* TP = 25, TN = 38, FP = 2, FN = 5

### 5.3 Phân tích lỗi (Error Analysis)
* **Bỏ sót (FN)**: 5 trường hợp - ngã quá chậm hoặc góc quay trực diện.
* **Báo giả (FP)**: 2 trường hợp - hành động nằm nghỉ nhanh trên ghế sofa.

---

## 6. Kết luận & Hướng phát triển

* **Kết luận**: Hệ thống đạt F1-Score 87.72%, chứng minh tính hiệu quả và khả năng ứng dụng thực tế.
* **Hướng phát triển**:
    * Tích hợp 3D Lifting để giải quyết vấn đề góc quay trực diện.
    * Tối ưu hóa Tracker cho các tình huống bị che khuất.

---

**Last Updated**: January 24, 2026
