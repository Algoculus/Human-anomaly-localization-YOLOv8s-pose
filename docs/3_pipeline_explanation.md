# Giải Thích Chi Tiết Pipeline và Logic Toán Học

## 1. Lưu Đồ Thuật Toán (Algorithm Flowchart)

Quá trình xử lý diễn ra tuần tự cho từng frame $t$ của video:

1.  **Input**: Frame ảnh màu $I_t \in \mathbb{R}^{H \times W \times 3}$.
2.  **Preprocessing ([preprocessing.py])**: Gamma Correction + CLAHE trên kênh Y.
3.  **Pose Estimation ([yolo_pose.py])**: YOLOv8s-pose dự đoán detections.
4.  **Multi-Person Tracking ([tracking.py])**: Gán ID cho detections.
5.  **Feature Extraction ([features.py])**: Tính toán $F_t$ bao gồm Border Integrity Check.
6.  **State Machine Detection ([smoothing.py])**: Cập nhật trạng thái $S_t$ và $Score_t$.

---

## 2. Trích Xuất Đặc Trưng (Feature Extraction)

### 2.1 Góc Nghiêng Cơ Thể (Body Angle)

$$ \theta = \arctan\left(\frac{|\Delta x|}{|\Delta y| + \epsilon}\right) \times \frac{180}{\pi} $$

- $\theta \approx 0^\circ$: Đứng thẳng. $\theta \approx 90^\circ$: Nằm ngang.

### 2.2 Tỷ Lệ Khung Hình (Aspect Ratio)

$$ AR = \frac{\text{Width}_{bbox}}{\text{Height}_{bbox}} $$

### 2.3 Vận Tốc Rơi ($dy$)

$$ dy*{peak} = \max(dy*{inst}) \quad \text{trong cửa sổ } W=5 \text{ frames} $$

### 2.4 Border Integrity Check

```python
if (x1 <= 8 or y1 <= 8 or x2 >= img_w - 8 or y2 >= img_h - 8):
    is_touching_border = True
```

Khi `is_touching_border = True`, Path 1 bị vô hiệu hóa.

---

## 3. 4-Path Fall Detection Rules

| Path       | Điều kiện                                                     | Mục đích                        |
| ---------- | ------------------------------------------------------------- | ------------------------------- |
| **Path 1** | $\theta \geq 55°$ AND $AR \geq 1.15$ AND $dy_{peak} \geq 5.0$ | Posture + Impact                |
| **Path 2** | $Drop \geq 0.18$                                              | Height Drop                     |
| **Path 3** | $dy \geq 10.0$ px/frame                                       | Fast Motion                     |
| **Path 4** | $\theta \geq 65°$ AND $dy_{peak} \geq 3.0$                    | High-Angle (slow/frontal falls) |

> **Path 4 (NEW)**: Được thêm vào để bắt các ca ngã chậm hoặc ngã trực diện bị bỏ sót.

---

## 4. Máy Trạng Thái (FSM)

### 4.1 Điều Kiện Chuyển Trạng Thái

#### `NORMAL` → `CANDIDATE`

Ít nhất 1 trong 4 Path = True trong `cand_enter_frames` (2 frames).

#### `CANDIDATE` → `FALL_CONFIRMED`

- Duy trì tư thế nằm ≥ 3 frames (`confirm_frames`).
- VÀ thoả mãn: Fast Motion ($dy_{peak} \geq 8.0$) HOẶC Strong Drop ($Drop \geq 0.28$) HOẶC Sustained Lie.

#### Recovery

Nếu người đứng dậy ($\theta < 55°$, $H_{curr} > 0.70 \times H_{base}$) trong 4 frames → Reset về `NORMAL`.

---

**Last Updated**: February 3, 2026
