# Giải Thích Metrics và Logic Đánh Giá

## 1. Logic Đánh Giá (Evaluation Protocol)

Hệ thống đánh giá hiệu năng dựa trên bài toán **Phân loại Video (Binary Sequence Classification)**.

*   **Đơn vị đánh giá**: 1 Sequence (Video clip).
*   **Quy tắc dự đoán**:
    *   Một sequence được dự đoán là **Fall (1)** nếu hệ thống phát hiện trạng thái `FALL_CONFIRMED` duy trì liên tục trong ít nhất $K$ frames (trong config `min_confirm_duration_frames` = 2).
    *   Ngược lại, dự đoán là **ADL (0)**.

## 2. Confusion Matrix (Ma trận nhầm lẫn)

Cho tập dữ liệu gồm $N$ sequences. Ký hiệu $y_i$ là nhãn thực tế (GT) và $\hat{y}_i$ là nhãn dự đoán của sequence thứ $i$.

$$
y_i, \hat{y}_i \in \{0, 1\}
$$

Các thành phần được tính như sau (Code: `eval.py:compute_metrics`):

*   **True Positive (TP)**: Số lượng sequence thực tế là Ngã và mô hình dự đoán đúng là Ngã.
    $$ TP = \sum_{i=1}^{N} \mathbb{I}(y_i=1 \land \hat{y}_i=1) $$

*   **True Negative (TN)**: Số lượng sequence thực tế là ADL và mô hình dự đoán đúng là ADL.
    $$ TN = \sum_{i=1}^{N} \mathbb{I}(y_i=0 \land \hat{y}_i=0) $$

*   **False Positive (FP - Báo động giả)**: Thực tế là ADL nhưng mô hình báo nhầm là Ngã.
    $$ FP = \sum_{i=1}^{N} \mathbb{I}(y_i=0 \land \hat{y}_i=1) $$

*   **False Negative (FN - Bỏ sót)**: Thực tế là Ngã nhưng mô hình không phát hiện được.
    $$ FN = \sum_{i=1}^{N} \mathbb{I}(y_i=1 \land \hat{y}_i=0) $$

## 3. Các Chỉ Số Đánh Giá (Metrics)

### 3.1 Accuracy (Độ chính xác toàn cục)
Tỷ lệ dự đoán đúng trên tổng số mẫu.

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

### 3.2 Precision (Độ chính xác dương tính)
Độ tin cậy của các cảnh báo. Precision càng cao, tỷ lệ báo động giả càng thấp.

$$ \text{Precision} = \frac{TP}{TP + FP + \epsilon} $$

*( $\epsilon$ là số rất nhỏ để tránh chia cho 0)*

### 3.3 Recall (Độ nhạy - Sensitivity)
Khả năng phát hiện các ca ngã thực tế. Recall quan trọng nhất trong bài toán y tế/an toàn.

$$ \text{Recall} = \frac{TP}{TP + FN + \epsilon} $$

### 3.4 Specificity (Độ đặc hiệu)
Khả năng nhận diện đúng các hoạt động bình thường, tránh nhầm lẫn ADL thành Fall.

$$ \text{Specificity} = \frac{TN}{TN + FP + \epsilon} $$

### 3.5 F1-Score
Trung bình điều hòa giữa Precision và Recall, dùng để đánh giá sự cân bằng của mô hình.

$$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall} + \epsilon} $$

## 4. Visualization Code

Quy trình tính toán diễn ra như sau:
1.  Tập hợp list `gt_labels` và `pred_labels` từ toàn bộ sequences.
2.  Chuyển đổi sang numpy array.
3.  Sử dụng các phép toán vector hóa (`numpy.sum`, `&`) để tính TP, TN, FP, FN cực nhanh.
4.  Kết quả được lưu vào [metrics.json] và hiển thị trực quan bằng thư viện `seaborn` (Heatmap cho Confusion Matrix) và `matplotlib` (Bar chart cho các chỉ số).
