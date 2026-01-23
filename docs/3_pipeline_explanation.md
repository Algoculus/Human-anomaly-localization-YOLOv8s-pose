# Giải Thích Chi Tiết Pipeline và Logic Toán Học

## 1. Lưu Đồ Thuật Toán (Algorithm Flowchart)

Quá trình xử lý diễn ra tuần tự cho từng frame $t$ của video:

1.  **Input**: Frame ảnh màu $I_t \in \mathbb{R}^{H \times W \times 3}$.
2.  **Preprocessing ([yolo_pose.py])**:
    *   Chuyển đổi không gian màu BGR $\rightarrow$ YCrCb.
    *   Áp dụng **Gamma Correction** ($ \gamma=1.3 $) trên kênh Y: $ Y' = \left( \frac{Y}{255} \right)^{1/\gamma} \times 255 $.
    *   Áp dụng **CLAHE** (Clip Limit=2.0, Grid=8x8) trên kênh Y.
    *   Chuyển ngược lại YCrCb $\rightarrow$ BGR.
3.  **Pose Estimation**: Model YOLOv8s-pose dự đoán tập hợp các detections $D_t = \{d_1, d_2, ...\}$.
4.  **Multi-Person Tracking ([tracking.py])**: Gán ID cho detections để duy trì tính liên tục.
5.  **Feature Extraction ([features.py])**: Tính toán vector đặc trưng $F_t$ cho từng người.
6.  **State Machine Detection ([smoothing.py])**: Cập nhật trạng thái $S_t$ và điểm số $Score_t$.

---

## 2. Tracking: Hàm Chi Phí (Cost Function)

Hệ thống sử dụng giải thuật Greedy Matching để ghép nối detection mới ($D$) với track cũ ($T$).
Ma trận chi phí $C$ giữa track $i$ và detection $j$ được tính như sau:

$$ C_{i,j} = (1 - \text{IoU}(T_i, D_j)) + \text{Dist}(Center(T_i), Center(D_j)) $$

*   $\text{IoU}$: Intersection over Union của Bounding Box.
*   $\text{Dist}$: Khoảng cách Euclidean chuẩn hóa.
*   Cặp $(i, j)$ được chọn nếu $C_{i,j}$ nhỏ nhất và thoả mãn ngưỡng $C_{i,j} < 2.0$.

Trong trường hợp YOLO mất dấu (Occlusion/Motion Blur), **Fallback Tracker (KCF)** sẽ được kích hoạt nếu mất dấu dưới `max_gap=6` frames, giúp duy trì BBox bằng phương pháp Correlation Filter truyền thống.

---

## 3. Trích Xuất Đặc Trưng (Feature Extraction Logic)

Với mỗi track, các đặc trưng vật lý sau được tính toán ([features.py]):

### 3.1 Góc Nghiêng Cơ Thể (Body Angle)
*   **Điểm mốc**:
    $$ M_{shoulder} = \frac{L_{sh} + R_{sh}}{2}, \quad M_{hip} = \frac{L_{hip} + R_{hip}}{2} $$
*   **Vector thân**: $\vec{v} = M_{shoulder} - M_{hip} = (\Delta x, \Delta y)$.
*   **Góc lệch với phương thẳng đứng**:
    $$ \theta = \arctan\left(\frac{|\Delta x|}{|\Delta y| + \epsilon}\right) \times \frac{180}{\pi} $$
    *   $\theta \approx 0^\circ$: Đứng thẳng.
    *   $\theta \approx 90^\circ$: Nằm ngang.

### 3.2 Tỷ Lệ Khung Hình (Aspect Ratio - AR)
Dùng làm phương án dự phòng khi mất Keypoints.
$$ AR = \frac{\text{Width}_{bbox}}{\text{Height}_{bbox}} $$

### 3.3 Vận Tốc Rơi (Vertical Velocity $dy$)
*   Vận tốc tức thời: $dy_{inst} = y_{center}^{(t)} - y_{center}^{(t-1)}$
*   Vận tốc đỉnh (Peak):
    $$ dy_{peak} = \max(dy_{inst}) \quad \text{trong cửa sổ } W=5 \text{ frames} $$
    Giá trị $dy_{peak}$ cao thể hiện cú va chạm mạnh (impact).

### 3.4 Sụt Giảm Chiều Cao (Normalized Height Drop)
$$ Drop = \frac{\text{median}(H_{history}) - H_{current}}{\text{median}(H_{history})} $$
*   Ngưỡng cảnh báo: `height_drop_thres = 0.22` (giảm 22% chiều cao).

---

## 4. Máy Trạng Thái Hữu Hạn (Finite State Machine - FSM)

Logic trung tâm nằm ở [smoothing.py], quản lý việc chuyển đổi giữa 3 trạng thái: `NORMAL`, `CANDIDATE`, `FALL_CONFIRMED`.

### 4.1 Tính Điểm Frame (Frame Score)
Mỗi frame được chấm điểm $Score_{frame} \in [0, 1]$:
$$ Score = 0.3 \cdot \theta_{norm} + 0.25 \cdot AR_{norm} + 0.25 \cdot Drop + 0.2 \cdot dy_{norm} $$
Score này được tích lũy theo thời gian với hệ số suy giảm `decay = 0.85`:
$$ S_t = 0.85 \cdot S_{t-1} + 0.15 \cdot Score_{frame} $$

### 4.2 Điều Kiện Chuyển Trạng Thái

#### Từ `NORMAL` $\rightarrow$ `CANDIDATE`
Nếu thoả mãn ít nhất 1 trong 3 nhóm điều kiện (OR logic) trong `cand_enter_frames` (2 frames):
1.  **Angle + AR**: $\theta > 55^\circ$ AND $AR > 1.15$.
2.  **Height Drop**: $Drop > 0.22$.
3.  **Fast Motion**: $dy > 13.5$ pixels/frame.

#### Từ `CANDIDATE` $\rightarrow$ `FALL_CONFIRMED`
Cần sự kết hợp của nhiều yếu tố trong cửa sổ xác nhận (`confirm_window=10`). Hệ thống sử dụng logic đa nhánh (Multi-path Logic) để tăng độ nhạy (Recall):

*   **Điều kiện tiên quyết**: Phải duy trì tư thế nằm (Lying Posture: $\theta > 48^\circ$ hoặc $AR > 1.35$) trong ít nhất `confirm_frames` (3 frames).
*   **VÀ thoả mãn 1 trong 3 nhánh sau**:
    1.  **Fast Motion**: $dy_{peak} > 12.5$.
    2.  **Strong Drop**: $Drop > 0.3$.
    3.  **Sustained Lie**: Nằm rất lâu (> 6 frames) và có $Drop > 0.22$.

#### Cơ Chế Recovery (Tự động Reset)
Nếu phát hiện dấu hiệu đứng dậy ([check_recovery](smoothing.py#141-203)), hệ thống lập tức chuyển về `NORMAL`:
*   Góc cơ thể trở lại thẳng ($\theta < 55^\circ$).
*   Chiều cao phục hồi ($H_{curr} > 0.7 \times H_{base}$).
*   Duy trì trong 4 frames liên tiếp.

Cơ chế này giúp loại bỏ báo động giả khi người chỉ cúi xuống nhặt đồ rồi đứng lên ngay.
