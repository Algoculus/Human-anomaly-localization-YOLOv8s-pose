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