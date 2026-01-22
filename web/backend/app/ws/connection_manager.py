"""
WebSocket Hub - Room-based broadcast system
Connects cameras with receivers in the same room
"""
import asyncio
import base64
import time
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
import cv2
import numpy as np
from datetime import datetime

from app.adapters.realtime_adapter import RealtimeSessionManager
from app.core.config import get_settings

settings = get_settings()

class ConnectionManager:
    """Manages WebSocket connections with room-based routing"""
    
    def __init__(self):
        # room_id -> {"cameras": Set[WebSocket], "receivers": Set[WebSocket]}
        self.rooms: Dict[str, Dict[str, Set[WebSocket]]] = {}
        
        # websocket -> metadata
        self.connections: Dict[WebSocket, Dict] = {}
        
        # Session manager for real-time processing
        self.session_manager = RealtimeSessionManager(settings.core_config_path)
        
        # Frame rate limiting
        self.frame_timestamps: Dict[str, list] = {}  # session_id -> [timestamps]
    
    async def connect(self, websocket: WebSocket, role: str, room_id: str, device_id: str):
        """Register new connection (WebSocket already accepted by caller)"""
        # NOTE: WebSocket should already be accepted by the router before calling this
        
        # Create room if not exists
        if room_id not in self.rooms:
            self.rooms[room_id] = {"cameras": set(), "receivers": set()}
        
        # Add to appropriate role set
        if role == "camera":
            self.rooms[room_id]["cameras"].add(websocket)
        elif role == "receiver":
            self.rooms[room_id]["receivers"].add(websocket)
        
        # Store metadata
        self.connections[websocket] = {
            "role": role,
            "room_id": room_id,
            "device_id": device_id,
            "connected_at": datetime.utcnow().isoformat()
        }
        
        print(f"[CONNECTED] {role.upper()} connected: {device_id} in room {room_id}")
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "role": role,
            "roomId": room_id,
            "deviceId": device_id,
            "serverTime": datetime.utcnow().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        if websocket not in self.connections:
            return
        
        metadata = self.connections[websocket]
        room_id = metadata["room_id"]
        role = metadata["role"]
        
        # Remove from room
        if room_id in self.rooms:
            if role == "camera":
                self.rooms[room_id]["cameras"].discard(websocket)
                # Cleanup session
                session_id = f"{room_id}_{metadata['device_id']}"
                self.session_manager.remove_session(session_id)
            elif role == "receiver":
                self.rooms[room_id]["receivers"].discard(websocket)
            
            # Remove empty rooms
            if not self.rooms[room_id]["cameras"] and not self.rooms[room_id]["receivers"]:
                del self.rooms[room_id]
        
        # Remove metadata
        del self.connections[websocket]
        
        print(f"[DISCONNECTED] {role.upper()} disconnected: {metadata['device_id']} from room {room_id}")
    
    def _check_frame_rate(self, session_id: str) -> bool:
        """Check if frame rate is within limits"""
        now = time.time()
        
        if session_id not in self.frame_timestamps:
            self.frame_timestamps[session_id] = []
        
        timestamps = self.frame_timestamps[session_id]
        
        # Remove old timestamps (older than 1 second)
        timestamps[:] = [ts for ts in timestamps if now - ts < 1.0]
        
        # Check limit
        if len(timestamps) >= settings.frame_rate_limit:
            return False
        
        timestamps.append(now)
        return True
    
    async def process_frame(
        self,
        websocket: WebSocket,
        room_id: str,
        device_id: str,
        frame_id: int,
        timestamp: float,
        frame_data_b64: str
    ):
        """Process frame from camera"""
        session_id = f"{room_id}_{device_id}"
        
        # Rate limiting
        if not self._check_frame_rate(session_id):
            await websocket.send_json({
                "type": "throttle",
                "suggestedFps": settings.frame_rate_limit - 5,
                "message": "Frame rate too high, please reduce"
            })
            return
        
        start_time = time.time()
        
        try:
            # Decode frame
            frame_bytes = base64.b64decode(frame_data_b64)
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to decode frame",
                    "frameId": frame_id
                })
                return
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get or create session
            session = self.session_manager.get_or_create_session(session_id)
            
            # Process frame through core AI
            result = session.process_frame(frame)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Send telemetry back to camera
            telemetry_data = {
                "type": "telemetry",
                "roomId": room_id,
                "frameId": frame_id,
                "state": result["state"],
                "score": result["score"],
                "isCandidate": result["isCandidate"],
                "isLying": result["isLying"],
                "bbox": result["bbox"],
                "keypoints": result.get("keypoints"),
                "latency": latency_ms
            }
            
            print(f"[TELEMETRY] Sending telemetry: frameId={frame_id}, state={result['state']}, score={result['score']:.2f}, bbox={result['bbox']}, keypoints_count={len(result.get('keypoints', [])) if result.get('keypoints') else 0}")
            
            await websocket.send_json(telemetry_data)
            
            # If alarm, broadcast to receivers
            if result["alarm"]:
                await self.broadcast_alarm(
                    room_id=room_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    state=result["state"],
                    score=result["score"],
                    snapshot_frame=result["snapshotFrame"]
                )
        
        except Exception as e:
            import traceback
            print(f"[ERROR] Error processing frame: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "frameId": frame_id
            })
    
    async def broadcast_alarm(
        self,
        room_id: str,
        frame_id: int,
        timestamp: float,
        state: str,
        score: float,
        snapshot_frame: Optional[np.ndarray]
    ):
        """Broadcast alarm to all receivers in room"""
        if room_id not in self.rooms:
            return
        
        # Encode snapshot
        snapshot_b64 = None
        if snapshot_frame is not None:
            # Convert RGB back to BGR for encoding
            snapshot_bgr = cv2.cvtColor(snapshot_frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', snapshot_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            snapshot_b64 = base64.b64encode(buffer).decode('utf-8')
        
        alarm_msg = {
            "type": "alarm",
            "roomId": room_id,
            "ts": timestamp,
            "frameId": frame_id,
            "severity": "high",
            "message": "Fall detected!",
            "snapshotJpegBase64": snapshot_b64,
            "state": state,
            "score": score,
            "detectedAt": datetime.utcnow().isoformat()
        }
        
        # Send to all receivers in room
        receivers = self.rooms[room_id]["receivers"]
        disconnected = []
        
        for receiver in receivers:
            try:
                await receiver.send_json(alarm_msg)
            except:
                disconnected.append(receiver)
        
        # Cleanup disconnected receivers
        for ws in disconnected:
            self.disconnect(ws)
        
        print(f"🚨 ALARM broadcast to {len(receivers) - len(disconnected)} receivers in room {room_id}")
    
    def get_room_stats(self, room_id: str) -> Dict:
        """Get statistics for a room"""
        if room_id not in self.rooms:
            return {"error": "Room not found"}
        
        return {
            "roomId": room_id,
            "cameras": len(self.rooms[room_id]["cameras"]),
            "receivers": len(self.rooms[room_id]["receivers"])
        }

# Global manager instance
manager = ConnectionManager()
