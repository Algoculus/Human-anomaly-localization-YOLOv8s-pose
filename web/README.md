# 🚀 Fall Detection Production Demo

Production-grade fall detection system with real-time WebSocket streaming and offline video processing.

## 🏗️ Architecture

```
production-demo/
├── backend/          # FastAPI service
│   ├── app/
│   │   ├── main.py
│   │   ├── api/      # REST endpoints
│   │   ├── ws/       # WebSocket hub
│   │   ├── adapters/ # Real-time frame adapter
│   │   └── models/   # Pydantic models
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/         # Vite + React + TypeScript
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── lib/
│   └── package.json
└── core_ai/          # Existing AI (as Python package)
    └── (symlink to ../urfd_fall_yolo_pose)
```

## 📦 Features

### Backend (FastAPI)

- ✅ WebSocket room-based broadcast (camera ↔ receivers)
- ✅ Real-time frame-by-frame processing
- ✅ Alarm detection with snapshot
- ✅ Offline video upload & processing
- ✅ CORS, logging, rate limiting
- ✅ Health check & version info

### Frontend (React)

- ✅ **Camera Simulator Mode**: Send frames via WebSocket
- ✅ **Alert Receiver Mode**: Receive alarms + snapshots
- ✅ Video upload & overlay viewer
- ✅ shadcn/ui components
- ✅ Real-time telemetry dashboard

## 🚀 Quick Start

### Local Development

**Backend:**

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd frontend
pnpm install
pnpm dev
```

Open: http://localhost:5173

### Production Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for EC2 + Vercel instructions.

## 📡 WebSocket Protocol

### Registration

```json
{
  "type": "register",
  "role": "camera" | "receiver",
  "deviceId": "cam-001",
  "roomId": "demo-1"
}
```

### Frame (Camera → Server)

```json
{
    "type": "frame",
    "roomId": "demo-1",
    "frameId": 123,
    "ts": 1234567890,
    "encoding": "jpeg",
    "data": "<base64>"
}
```

### Telemetry (Server → Camera)

```json
{
    "type": "telemetry",
    "roomId": "demo-1",
    "frameId": 123,
    "state": "CANDIDATE",
    "score": 0.83,
    "isCandidate": true,
    "isLying": false,
    "bbox": [100, 50, 200, 300],
    "latencyMs": 12
}
```

### Alarm (Server → Receivers)

```json
{
    "type": "alarm",
    "roomId": "demo-1",
    "ts": 1234567890,
    "frameId": 456,
    "severity": "high",
    "message": "Fall detected",
    "snapshotJpegBase64": "...",
    "state": "FALL_CONFIRMED",
    "score": 0.95
}
```

## 🎯 Core AI Integrity

✅ **ZERO modifications to core AI logic**

- All core files remain untouched
- Backend uses thin adapter pattern
- Frame-by-frame processing reuses existing classes

## 📄 License

MIT
