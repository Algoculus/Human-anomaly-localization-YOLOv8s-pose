# Phân tích Dataset: UR Fall Detection Dataset (URFD)

## 1. Tổng quan Dataset

*   **Tên dataset**: UR Fall Detection Dataset (URFD).
*   **Loại dữ liệu**: Chuỗi ảnh RGB (Image Sequences).
*   **Mục tiêu**: Phát hiện hành động ngã ở cấp độ Video (Sequence-level Fall Detection).
*   **Phân loại (Classes)**:
    *   **Class 0 (ADL - Activities of Daily Living)**: Các hoạt động sinh hoạt thường ngày như đi bộ, ngồi xuống ghế, nhặt đồ vật.
    *   **Class 1 (Fall)**: Các hành động ngã (té về phía trước, phía sau, sang ngang).

## 2. Quy trình Load Dữ liệu (Source Code Logic)

### 2.1 Thứ tự ưu tiên tìm kiếm
Với mỗi thư mục sequence (ví dụ: `fall-01`), hệ thống sẽ tìm kiếm thư mục chứa ảnh RGB theo thứ tự ưu tiên sau:

1.  `path/to/sequence/sequence_name/` (Thư mục lồng nhau cùng tên).
2.  `path/to/sequence/` (Thư mục hiện tại).
3.  `path/to/sequence/cam0/rgb/` (Cấu trúc chuẩn của URFD gốc).
4.  `path/to/sequence/cam0-rgb/`
5.  `path/to/sequence/rgb/`

### 2.2 Sắp xếp Frame (Numeric Sort)
Thay vì sắp xếp theo tên file mặc định (Alphabetic Sort, ví dụ: `1.png`, `10.png`, `2.png`), code sử dụng **Numeric Sort** để đảm bảo đúng trình tự thời gian:
$$ \text{key} = \text{int(frame\_name)} $$
Điều này đảm bảo frame `10.png` luôn đứng sau `2.png`.

### 2.3 Định dạng ảnh
*   Hỗ trợ: `*.png`, `*.jpg`.
*   Color Space: Ảnh được đọc bằng `cv2.imread`, trả về định dạng **BGR** (Blue-Green-Red).

## 3. Quy trình Gán Nhãn (Labeling Strategy)

### 3.1 Quy tắc Tiền tố (Prefix Rule)
Hệ thống duyệt qua tên của thư mục sequence (sau khi loại bỏ hậu tố `-cam0-rgb` nếu có):

*   Nếu tên bắt đầu bằng `fall-` $\rightarrow$ Gán **Label 1 (Fall)**.
*   Nếu tên bắt đầu bằng `adl-` $\rightarrow$ Gán **Label 0 (ADL)**.

### 3.2 Metadata
Sau khi quét toàn bộ thư mục `data/raw`, hệ thống sinh ra file index `data/urfd_index.csv` chứa các trường thông tin quan trọng:

*   `seq_name`: Tên định danh (VD: `fall-01`).
*   `seq_path`: Đường dẫn tuyệt đối đến thư mục gốc của sequence.
*   `gt_label`: Nhãn chuẩn (0 hoặc 1).
*   `frame_dir`: Đường dẫn thực tế chứa các file ảnh (đã resolve xong).
*   `num_frames`: Tổng số frame trong sequence.

**Lưu ý quan trọng**: Nhãn `gt_label` áp dụng cho **toàn bộ sequence**. Trong bài toán hiện tại, chúng ta đánh giá mô hình dựa trên khả năng phát hiện "Có ngã hay không" trong cả video, chứ không phải gán nhãn từng frame (Frame-level annotation) vì dataset gốc URFD chỉ cung cấp nhãn mức video.
