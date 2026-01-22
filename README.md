# Phát hiện và giám sát bất thường khi té ngã trên tập URFD với YOLOv8s-pose

Ngắn gọn: Dự án triển khai phát hiện ngã và giám sát bất thường dựa trên pose estimation kết hợp YOLOv8s-pose trên tập URFD (UR Fall Detection Dataset).

## Nội dung
- Mục tiêu: phát hiện sự kiện té ngã (fall) và cảnh báo bất thường thời gian thực.
- Phương pháp: YOLOv8s-pose để ước lượng khung xương (keypoints) → luật/ML trên pose để phân loại ngã.
- Dataset: URFD (video/canvas, nhãn fall/non-fall, khung thời gian).

## Yêu cầu
- Python 3.8+
- PyTorch 1.12+ (tùy GPU)
- ultralytics (YOLOv8) tương thích với phiên bản PyTorch
- opencv-python, scikit-learn, numpy, pandas, matplotlib

Ví dụ cài:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install ultralytics opencv-python scikit-learn pandas matplotlib
```

## Cấu trúc đề xuất
- data/
    - URFD/
        - videos/
        - labels/ (frame-level hoặc json keypoints)
- configs/
    - yolo_pose_cfg.yaml
- weights/
- src/
    - dataset.py
    - train.py
    - infer.py
    - evaluate.py
- README.md

## Chuẩn hóa dữ liệu (tóm tắt)
1. Trích frame từ video URFD.
2. Chuyển nhãn video → nhãn frame (fall/non-fall) hoặc segment.
3. Nếu cần, annotate keypoints hoặc dùng YOLOv8s-pose để sinh keypoints không giám sát.
4. Chia train/val/test (ví dụ 70/15/15 theo video).

## Huấn luyện (tóm tắt)
- Sử dụng cấu hình YOLOv8s-pose, batch và lr phù hợp GPU.
- Lệnh mẫu:
```
python src/train.py --data data/urfd.yaml --cfg configs/yolo_pose_cfg.yaml --epochs 50 --batch 8 --weights yolov8s-pose.pt
```
- Theo dõi loss pose và accuracy phân loại fall.

## Inference & Giám sát thời gian thực
- Chạy stream video hoặc camera:
```
python src/infer.py --weights weights/best.pt --source 0
```
- Luồng: nhận keypoints → tính feature (center of mass, angle giữa thân/chân, tốc độ keypoint) → áp bộ quy tắc / model phụ để phát hiện ngã → hiển thị bounding + cảnh báo.

## Demo nhanh (YOLOv8-Pose + FSM + Safe Zone + Multi-Person Tracking)

**Tính năng:**
- ✅ **Multi-person detection**: Phát hiện nhiều người cùng lúc
- ✅ **Tracking (ByteTrack)**: Mỗi người có ID ổn định theo thời gian
- ✅ **FSM riêng cho từng người**: Người A ngã, người B đứng vẫn bình thường
- ✅ **Safe Zone**: Loại bỏ false alarm khi nằm ngủ/ghế sofa

Cài thư viện:
```
pip install -r requirements.txt
```

Test YOLO pose webcam (tuỳ chọn):
```
yolo pose predict model=yolov8n-pose.pt source=0 show=True
```

Chạy demo FSM với tracking (webcam):
```
python src/urfall_demo_fsm.py --source 0 --model yolov8n-pose.pt
```

Chạy demo FSM (video UR Fall):
```
python src/urfall_demo_fsm.py --source data/UR_Fall_Detection_Dataset/data/fall-01-cam0.mp4 --model yolov8n-pose.pt
```

Phím tắt:
- `z`: vẽ lại Safe Zone (kéo chuột trái để chọn hình chữ nhật)
- `s`: lưu Safe Zone vào `outputs/safe_zone.json`
- `q`: thoát

**Output:**
- Mỗi người có bbox màu riêng (theo track ID)
- Label theo từng người: `ID:X FALL DETECTED`, `ID:X LYING`, `ID:X INACTIVITY`, `ID:X NORMAL`, `ID:X SAFE ZONE (SLEEP/REST)`
- FSM tự động cleanup khi người rời khỏi frame (sau 30 frames không thấy)

## Đánh giá nhanh trên UR Fall (event-level)

Vì trong bản dataset đang có ở máy bạn **chưa thấy** các file frame-level GT như `urfall-cam0-falls.csv`/`urfall-cam0-adls.csv`, script dưới đây đánh giá **event-level theo clip** (fall-*.mp4 là 1, adl-*.mp4 là 0):
```
python src/urfall_eval_event.py --data-dir data/UR_Fall_Detection_Dataset/data --model yolov8n-pose.pt
```

Script sẽ xuất:
- Confusion matrix (TP/FP/TN/FN)
- Recall, Specificity, Precision, F1
- CSV dự đoán theo video ở `outputs/urfall_event_metrics.csv`

## Đánh giá
- Metrics: Precision, Recall, F1-score, AUC cho phân loại frame/segment; mAP cho pose nếu có ground-truth.
- Thực nghiệm: báo cáo trên từng video; báo lỗi false positive/negative.

## Mẹo cải thiện
- Fine-tune thresholds bằng validation set.
- Dùng temporal smoothing / LSTM trên chuỗi keypoints.
- Augmentation video (brightness, occlusion, blur).
- Ensembles hoặc transfer learning từ model pose lớn hơn.

## Bảo mật & đạo đức
- Chú ý quyền riêng tư khi xử lý video người thật; xóa dữ liệu nhận diện cá nhân nếu cần.

## Tài liệu tham khảo
- URFD dataset (trích dẫn gốc khi công bố).
- YOLOv8, ultralytics repo.

## Giấy phép
- Ghi rõ license cho code (ví dụ MIT) và tuân thủ license dataset URFD.

<!-- Liên hệ / Issues -->
- Mô tả, issues và cải tiến: mở issues trên repo.