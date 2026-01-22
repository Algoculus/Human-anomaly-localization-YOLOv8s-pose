"""
WebSocket Router
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.connection_manager import manager

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for camera and receiver connections.
    
    Protocol:
    1. Client connects
    2. Client sends registration: {"type": "register", "role": "camera"|"receiver", "deviceId": "...", "roomId": "..."}
    3. Camera sends frames: {"type": "frame", "roomId": "...", "frameId": 123, "ts": ..., "data": "<base64>"}
    4. Server sends telemetry to camera and alarms to receivers
    """
    
    registered = False
    role = None
    room_id = None
    device_id = None
    
    try:
        # Wait for registration
        await websocket.accept()
        
        # First message must be registration
        data = await websocket.receive_json()
        
        if data.get("type") != "register":
            await websocket.send_json({
                "type": "error",
                "message": "First message must be registration"
            })
            await websocket.close()
            return
        
        role = data.get("role")
        room_id = data.get("roomId")
        device_id = data.get("deviceId")
        
        if not all([role, room_id, device_id]):
            await websocket.send_json({
                "type": "error",
                "message": "Missing required fields: role, roomId, deviceId"
            })
            await websocket.close()
            return
        
        if role not in ["camera", "receiver"]:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid role. Must be 'camera' or 'receiver'"
            })
            await websocket.close()
            return
        
        # Register connection
        await manager.connect(websocket, role, room_id, device_id)
        registered = True
        
        # Main message loop
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "frame" and role == "camera":
                # Process frame
                await manager.process_frame(
                    websocket=websocket,
                    room_id=data.get("roomId", room_id),
                    device_id=device_id,
                    frame_id=data.get("frameId", 0),
                    timestamp=data.get("ts", 0),
                    frame_data_b64=data.get("data", "")
                )
            
            elif msg_type == "ping":
                # Heartbeat
                await websocket.send_json({"type": "pong"})
            
            elif msg_type == "stats":
                # Get room stats
                stats = manager.get_room_stats(room_id)
                await websocket.send_json({
                    "type": "stats",
                    **stats
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        if registered:
            manager.disconnect(websocket)
