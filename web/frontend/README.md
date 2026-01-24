# Fall Detection System - Frontend

React + TypeScript application serving as the user interface for the Fall Detection System. It includes a Camera Simulator (for sending frames) and a Receiver Dashboard (for monitoring alerts).

## 🛠️ Tech Stack

- **Framework**: React 18
- **Build Tool**: Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Shadcn/ui (radix-ui based)
- **Icons**: Lucide React
- **Integration**: WebSockets (native)

## 📂 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/            # Reusable UI components (Button, Card, etc.)
│   │   └── ...
│   ├── pages/
│   │   ├── CameraPage.tsx   # Camera Simulator & Visualizer
│   │   ├── ReceiverPage.tsx # Monitoring Dashboard
│   │   └── HomePage.tsx     # Landing page
│   ├── App.tsx            # Main router
│   └── main.tsx           # Entry point
├── public/                # Static assets
└── vite.config.ts         # Vite configuration
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18.x or higher
- npm or yarn

### Installation

```bash
# Navigate to frontend directory
cd web/frontend

# Install dependencies
npm install
```

### Running Locally

```bash
# Start development server
npm run dev
```
The application will operate at `http://localhost:3000`.

## 🖥️ Key Features

### Camera Page (`/camera`)
- **Webcam Access**: Captures video from local webcam.
- **WebSocket Streaming**: Sends video frames to the backend (`ws://localhost:4611/ws`).
- **Telemetry Overlay**: Receives inference results (bounding boxes, keypoints, state) and renders them on top of the video feed.
- **Visuals**:
  - **Green**: Normal State
  - **Yellow**: Candidate State
  - **Red**: Fall Confirmed
  - **Skeleton**: Visualizes pose estimation keypoints and connections.

### Receiver Page (`/receiver`)
- **Dashboard**: Monitors alarms from connected cameras.
- **Real-time Alerts**: Shows immediate notifications when a fall is detected.

## 🤝 Backend Connection

Ensure the FastAPI backend is running on port `4611` before using the frontend. The WebSocket connection string is hardcoded to `ws://localhost:4611/ws` in this demo version.
