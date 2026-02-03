# Fall Detection System - Backend

FastAPI server that acts as the central hub for the Fall Detection System. It processes video frames received via WebSockets, runs inference using the YOLOv8s-pose model, and broadcasts telemetry/alarms.

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python)
- **Server**: Uvicorn
- **AI Engine**: Ultralytics YOLOv8s-pose (loaded from `urfd_fall_yolo_pose`)
- **Communication**: WebSockets

## 📂 Project Structure

```
backend/
├── app/
│   ├── main.py            # Application entry point & API routes
│   └── ws/
│       └── connection_manager.py # WebSocket connection handler
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Conda (recommended)

### Installation

```bash
# Create environment
conda create -n fall_detection python=3.10
conda activate fall_detection

# Install dependencies
cd web/backend
pip install fastapi uvicorn websockets ultralytics opencv-python
```

> **Note**: This backend relies on the `urfd_fall_yolo_pose` module. Ensure the project root is in your PYTHONPATH or run from the correct directory level if needed.

### Running the Server

```bash
# From web/backend directory
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The server will start on `http://127.0.0.1:8000`.

## 🔌 API Endpoints

- `GET /`: Health check.
- `WS /ws`: WebSocket endpoint for bidirectional communication (frames -> inference -> telemetry).

## 📡 WebSocket Protocol

### Messages sent by Client (Camera)

```json
{
    "type": "register",
    "role": "camera",
    "roomId": "room-1",
    "deviceId": "cam-01"
}
```

```json
{
    "type": "frame",
    "data": "<base64_encoded_image>",
    "ts": 1234567890
}
```

### Messages sent by Server

```json
{
  "type": "telemetry",
  "tracks": [
    {
      "id": 1,
      "state": "NORMAL",
      "score": 0.95,
      "bbox": [x1, y1, x2, y2],
      "keypoints": [...]
    }
  ],
  "alarm": false
}
```
