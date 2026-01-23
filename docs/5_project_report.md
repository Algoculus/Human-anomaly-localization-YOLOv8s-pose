# BÁO CÁO BÀI TẬP LỚN

## Hệ Thống Phát Hiện Ngã (Fall Detection) sử dụng YOLOv8 Pose Estimation

## Tóm tắt (Abstract)
(0.5 trang)

Bài báo cáo trình bày quá trình xây dựng hệ thống phát hiện ngã tự động dựa trên thị giác máy tính, giải quyết vấn đề giám sát an toàn cho người cao tuổi. Chúng tôi đề xuất phương pháp tiếp cận hai giai đoạn: (1) Trích xuất khung xương (Pose Extraction) thời gian thực sử dụng **YOLOv8s-pose**; (2) Phân tích hành vi bằng **Finite State Machine (FSM)** dựa trên các đặc trưng động học (Vận tốc rơi, Góc nghiêng, Tỷ lệ khung hình). Hệ thống được kiểm thử trên bộ dữ liệu chuẩn **URFD** (70 sequences), đạt độ chính xác **88.57%**, Recall **83.33%** và F1-Score **86.21%**, chứng minh khả năng ứng dụng thực tế cao với chi phí tính toán thấp.

---

## 1. Giới thiệu (Introduction)
(1 trang)

* **Bối cảnh**: Té ngã là rủi ro lớn đối với người cao tuổi sống một mình. Các giải pháp camera truyền thống đòi hỏi người giám sát liên tục, trong khi thiết bị đeo (wearables) thường bị quên lãng.
* **Mục tiêu**: Phát triển hệ thống AI có khả năng tự động phát hiện cú ngã từ camera giám sát thông thường, đưa ra cảnh báo tức thì mà không cần can thiệp của con người.
* **Đóng góp**:
    *   Tích hợp YOLOv8-pose tiên tiến cho bài toán Fall Detection.
    *   Đề xuất thuật toán FSM lọc nhiễu hiệu quả, giảm thiểu báo động giả.
    *   Cung cấp bộ mã nguồn hoàn chỉnh với module đánh giá chi tiết.

---

## 2. Tổng quan lý thuyết (Related Work / Background)
(1 trang)

* **Object Detection & Pose Estimation**:
    *   **YOLOv8**: Phiên bản mới nhất của dòng YOLO, cung cấp khả năng phát hiện keypoints (Pose) với độ chính xác cao và tốc độ thời gian thực ($>30$ FPS trên GPU phổ thông).
* **Phương pháp phát hiện ngã**:
    *   *Dựa trên ngoại hình (Appearance-based)*: Dùng CNN phân loại ảnh trực tiếp. Nhược điểm: Kém bền vững khi thay đổi góc quay.
    *   *Dựa trên dáng bộ (Pose-based)*: Trích xuất khung xương rồi phân tích. Ưu điểm: Bền vững hơn, bảo vệ quyền riêng tư tốt hơn (chỉ cần tọa độ khớp).

---

## 3. Dataset & Tiền xử lý dữ liệu
(1-1.5 trang)

### 3.1 Dataset: UR Fall Detection Dataset (URFD)
* **Nguồn**: University of Rzeszow.
* **Số lượng**: 70 video clips (Sequences).
    *   30 Clips Ngã (Positive Samples - Label 1).
    *   40 Clips Sinh hoạt thường ngày (Negative Samples - Label 0).
* **Đặc điểm**: Quay bằng camera Kinect (nhưng chỉ dùng kênh RGB), độ phân giải cao, ánh sáng ổn định.

### 3.2 Tiền xử lý (Preprocessing)
* **Gamma Correction & CLAHE**: Áp dụng cân bằng sáng cục bộ trên kênh Luma để làm rõ chi tiết trong vùng tối, giúp YOLO nhận diện tốt hơn ở điều kiện thiếu sáng.
* **Resize**: Chuẩn hóa ảnh về kích thước $640 \times 640$ pixel (stride 32) phù hợp với input của YOLOv8.

---

## 4. Phương pháp đề xuất (Methodology)
(2 trang)

### 4.1 Kiến trúc Pipeline
Hệ thống xử lý theo luồng: `Input Frame` $\rightarrow$ `Pose Check` $\rightarrow$ `Tracking` $\rightarrow$ `Feature Analysis` $\rightarrow$ `State Machine` $\rightarrow$ `Alert`.

### 4.2 Đặc trưng vật lý
Chúng tôi sử dụng vector đặc trưng $F = [\theta, AR, dy, Drop]$:
1.  **Góc nghiêng ($\theta$)**: Góc giữa vector thân người và phương thẳng đứng. Ngã thường có $\theta > 45^\circ$.
2.  **Tỷ lệ khung hình ($AR$)**: Tỷ lệ Rộng/Cao. Người nằm có $AR > 1.0$.
3.  **Vận tốc rơi ($dy$)**: Tốc độ di chuyển trọng tâm xuống dưới. Cú ngã có $dy$ lớn đột biến.
4.  **Sụt giảm chiều cao ($Drop$)**: Tỷ lệ mất chiều cao so với trạng thái đứng.

### 4.3 Máy trạng thái (State Machine)
Sử dụng FSM 3 trạng thái để mô hình hóa quá trình ngã:
*   `NORMAL`: Trạng thái bình thường.
*   `CANDIDATE`: Phát hiện dấu hiệu bất thường (Nghiêng người, Rơi nhanh).
*   `FALL_CONFIRMED`: Xác nhận ngã sau khi kiểm tra tính ổn định của tư thế nằm trong cửa sổ thời gian (Temporal Window).

---

## 5. Thực nghiệm & Kết quả
(2 trang)

### 5.1 Thiết lập thực nghiệm
*   **Model**: YOLOv8s-pose.
*   **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Specificity (trên tập dữ liệu 70 clips).
*   **Criterion**: Sequence được coi là Fall nếu phát hiện trạng thái `FALL_CONFIRMED` duy trì $\ge 2$ frames.

### 5.2 Kết quả định lượng

| Metric | Giá trị (%) | Ý nghĩa |
| :--- | :--- | :--- |
| **Accuracy** | **88.57** | Độ chính xác tổng thể tốt. |
| **Precision** | **89.29** | Ít báo động giả (High trust). |
| **Recall** | **83.33** | Phát hiện được đa số các ca ngã. |
| **Specificity** | **92.50** | Phân biệt rất tốt hành động thường ngày. |
| **F1-Score** | **86.21** | Cân bằng tốt. |

### 5.3 Phân tích lỗi (Error Analysis)
*   **Bỏ sót (False Negative)**: 5 trường hợp. Nguyên nhân chủ yếu do góc quay trực diện (Frontal View) khiến tỷ lệ AR và góc nghiêng 2D không thay đổi rõ rệt khi ngã hướng về phía camera.
*   **Báo giả (False Positive)**: 3 trường hợp. Do hành động nằm nghỉ nhanh trên ghế sofa hoặc cúi người nhặt đồ với tốc độ cao.

---

## 6. Kết luận & Hướng phát triển
(0.5 trang)

*   **Kết luận**: Hệ thống đề xuất đã chứng minh tính hiệu quả và khả thi cho việc triển khai giám sát an toàn. Việc kết hợp Deep Learning (YOLO) cho nhận diện pose và FSM cho phân tích logic giúp hệ thống vừa chính xác vừa dễ giải thích (Explainable AI).
*   **Hướng phát triển**:
    *   Tích hợp module 3D Lifting để chuyển Keypoints 2D sang 3D, giải quyết vấn đề góc quay trực diện.
    *   Tối ưu hóa Tracker để xử lý tốt hơn các tình huống bị che khuất một phần (Occlusion).

---

## 8. Phụ lục (Appendix)
*   **Project Structure**: (Tham khảo cấu trúc thư mục trong tài liệu Source Code).
*   **Source Code**: Được tổ chức theo mô hình module hóa cao, dễ dàng mở rộng và bảo trì.
