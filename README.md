# Fall Detection & Localization System (YOLOv8-pose)
*Hệ thống Phát hiện và Định vị Ngã sử dụng YOLOv8-pose*

## Overview (Tổng quan)

This project implements a real-time Fall Detection system combining **YOLOv8-pose** (skeleton tracking), **Depth Features** (from URFall dataset), and **Accelerometer Data**. It uses a multi-modal Finite State Machine (FSM) to accurately detect falls while minimizing false alarms.

Dự án này triển khai hệ thống phát hiện ngã thời gian thực kết hợp **YOLOv8-pose**, **Tính năng Độ sâu** và **Dữ liệu Gia tốc kế**. Hệ thống sử dụng Máy trạng thái hữu hạn (FSM) đa phương thức để phát hiện chính xác các cú ngã và giảm thiểu báo động giả.

---

## 🚀 Key Features (Tính năng chính)

1.  **Multi-modal Fusion**: Combines Pose (RGB), Depth (Geometry), and Accelerometer (Impact).
2.  **Robust Logic**: Uses normalized velocity (body-heights/sec) for scale-invariant drop detection.
3.  **Visual Dashboard**: Neon-style overlay with real-time risk scores and state indicators.
4.  **Complete Evaluation**: Includes scripts for batch inference and detailed metric calculation (ROC, AUC, etc.).
5.  **FastAPI Demo**: Web interface for easy testing.

---

## 📂 Directory Structure (Cấu trúc thư mục)

```
d:/Human-anomaly-localization-YOLOv8s-pose/
├── app/                  # Demo Application (FastAPI)
├── data/                 # Dataset (URFall)
├── output/               # All generated results go here
│   ├── videos/           # Annotated inference videos
│   ├── plots/            # Evaluation plots (ROC, Confusion Matrix)
│   └── metrics/          # CSV reports
├── scripts/              # Execution Scripts
│   ├── run_infer.py      # Run batch inference
│   └── run_eval.py       # Compute metrics
├── src/                  # Source Code
│   ├── config.py         # Configuration (Thresholds, Paths)
│   ├── rules/            # Core Logic (Scoring, FSM)
│   └── utils/            # Helpers (Visualization, Tracking)
└── yolov8s-pose.pt       # Pre-trained Model
```

---

## 🛠️ Usage (Hướng dẫn sử dụng)

### 1. Installation (Cài đặt)
```bash
pip install -r requirements.txt
```

### 2. Run Inference (Chạy suy luận)
To process the dataset and generate videos/CSVs:
*Để xử lý tập dữ liệu và tạo video/CSV kết quả:*

```bash
python scripts/run_infer.py --output output/infer_run --video
```

- **Output**: Check `output/infer_run/` for `.mp4` videos and `.csv` logs.
- *Kết quả: Kiểm tra thư mục `output/infer_run/` để xem video và file log.*

### 3. Run Evaluation (Đánh giá)
To calculate accuracy, AUC, and generate plots on the Test set:
*Để tính toán độ chính xác và vẽ biểu đồ đánh giá trên tập Test:*

```bash
python scripts/run_eval.py
```

- **Plots**: Saved to `output/eval_run/plots/`
- **Metrics**: Saved to `output/eval_run/metrics_summary.txt`

### 4. Run Demo App (Chạy Demo)
Start the local server:
*Khởi động server cục bộ:*

```bash
uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` to test the API.

---

## ⚙️ Configuration & Tuning (Cấu hình)

Edit `src/config.py` to tune detection sensitivity:
*Chỉnh sửa `src/config.py` để điều chỉnh độ nhạy:*

- **`drop_threshold`**: Sensitivity to sudden drops (Default: 0.05 body-heights/frame).
- **`fall_alert_threshold`**: Total risk score to trigger alert (Default: 0.8).
- **`weights`**: Adjust importance of Drop vs. Prone vs. Impact components.

---

## 📊 Methodology (Phương pháp)

### Risk Score Calculation
$$ Score = w_{drop} \cdot S_{drop} + w_{prone} \cdot S_{prone} + w_{impact} \cdot S_{impact} + w_{lie} \cdot S_{lie} $$

### State Machine (Máy trạng thái)
- **NORMAL**: Upright posture, low risk.
- **FALLING**: Detected sudden drop or impact.
- **LYING**: Confirmed fall (high prone score after falling).
- **LYING_NO_FALL**: Lying down slowly (resting) - No Alert.

---

## 📈 Performance

- **Accuracy**: >91%
- **ROC-AUC**: >0.98
- **Event Recall**: 100% (All falls detected)

---

## Credits
- Dataset: URFall Detection Dataset
- Model: Ultralytics YOLOv8