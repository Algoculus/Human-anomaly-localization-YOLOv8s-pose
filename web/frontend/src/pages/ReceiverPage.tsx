import { useState, useEffect, useRef } from "react";
// useNavigate is unused if we remove the back button from here, but Layout uses it internally.
// Wait, Layout handles the back button. So we don't need useNavigate in ReceiverPage unless we use it elsewhere.
// Looking at ReceiverPage code, navigate is ONLY used for the back button.
// So we can remove useNavigate too.
import {
  Users,
  Bell,
  Image as Wifi,
  AlertTriangle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";
import Layout from "@/components/Layout";

interface AlarmData {
  roomId: string;
  deviceId: string;
  frameId: number;
  ts: number;
  snapshot: string; // base64 JPEG
  state: string;
  score: number;
}

const FIXED_ROOM_ID = "fall-detection-room";

export default function ReceiverPage() {
  const { toast } = useToast();
  const [isConnected, setIsConnected] = useState(false);
  const [deviceId] = useState(`receiver-${Date.now()}`);
  const [alarms, setAlarms] = useState<AlarmData[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    // Create alarm audio
    audioRef.current = new Audio("/alarm.mp3");

    return () => {
      disconnect();
    };
  }, []);

  const connect = () => {
    const ws = new WebSocket(`ws://localhost:8000/ws`);

    ws.onopen = () => {
      console.log("WebSocket connected");

      // Register as receiver
      ws.send(
        JSON.stringify({
          type: "register",
          role: "receiver",
          roomId: FIXED_ROOM_ID,
          deviceId: deviceId,
        }),
      );

      setIsConnected(true);

      toast({
        title: "✅ Connected",
        description: `Monitoring room: ${FIXED_ROOM_ID}`,
      });
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "alarm") {
        handleAlarm(data);
      } else if (data.type === "error") {
        toast({
          title: "Error",
          description: data.message,
          variant: "destructive",
        });
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      toast({
        title: "Connection Error",
        description: "Failed to connect to server",
        variant: "destructive",
      });
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
      setIsConnected(false);
    };

    wsRef.current = ws;
  };

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  };

  const handleAlarm = (data: AlarmData) => {
    // Add to alarms list
    setAlarms((prev) => [data, ...prev].slice(0, 20)); // Keep last 20

    // Play alarm sound
    if (audioRef.current) {
      audioRef.current
        .play()
        .catch((e) => console.error("Audio play failed:", e));
    }

    // Show toast notification
    toast({
      title: "🚨 FALL DETECTED!",
      description: `Camera: ${data.deviceId} | Frame: ${data.frameId}`,
      variant: "destructive",
    });

    // Request notification permission and send browser notification
    if (Notification.permission === "granted") {
      new Notification("Fall Detected!", {
        body: `Camera ${data.deviceId} detected a fall`,
        icon: "/fall-icon.png",
        badge: "/badge.png",
      });
    } else if (Notification.permission !== "denied") {
      Notification.requestPermission();
    }
  };

  const clearAlarms = () => {
    setAlarms([]);
    toast({
      title: "Cleared",
      description: "All alarms cleared",
    });
  };

  const formatTime = (ts: number) => {
    return new Date(ts).toLocaleTimeString();
  };

  return (
    <Layout
      themeColor="red"
      title="Alert Receiver"
      subtitle="Monitor fall alerts in real-time"
      icon={Bell}
      actions={
        <div className="flex items-center gap-3">
          {alarms.length > 0 && (
            <Button
              variant="outline"
              onClick={clearAlarms}
              className="border-2 border-border hover:border-accent hover:bg-accent hover:text-accent-foreground"
            >
              Clear All
            </Button>
          )}
          {isConnected ? (
            <Button
              variant="destructive"
              onClick={disconnect}
              className="border-2 border-red-500 hover:shadow-[0_0_20px_rgba(239,68,68,0.5)] transition-all duration-300 text-foreground"
            >
              Disconnect
            </Button>
          ) : (
            <Button
              onClick={connect}
              className="border-2 border-red-500 bg-red-500/10 hover:bg-red-500/20 hover:shadow-[0_0_20px_rgba(239,68,68,0.5)] transition-all duration-300 text-foreground"
            >
              <Users className="w-5 h-5 mr-2" />
              Connect
            </Button>
          )}
        </div>
      }
    >
      <div className="grid lg:grid-cols-3 gap-4 md:gap-6">
        {/* Configuration Panel */}
        <div>
          <Card className="border-2 border-red-500/30 bg-card">
            <CardHeader className="border-b-2 border-red-500/20 bg-gradient-to-b from-red-500/10 to-transparent">
              <CardTitle className="text-lg text-card-foreground font-bold uppercase">
                Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 pt-6">
              <div className="p-4 bg-red-500/10 border-2 border-red-500/30">
                <Label className="text-sm font-bold text-red-400 uppercase">
                  Room ID (Fixed)
                </Label>
                <p className="text-sm text-red-300 mt-1 font-mono">
                  {FIXED_ROOM_ID}
                </p>
                <p className="text-xs text-red-400/70 mt-2">
                  All devices use the same room
                </p>
              </div>
              <div>
                <Label className="text-muted-foreground font-medium">Device ID</Label>
                <Input
                  value={deviceId}
                  disabled
                  className="font-mono text-sm bg-muted border-2 border-border text-muted-foreground mt-2"
                />
              </div>

              <div className="pt-4 border-t-2 border-border space-y-3">
                <div className="flex items-center justify-between p-4 bg-muted border-2 border-border">
                  <Label className="text-sm flex items-center gap-2 text-muted-foreground font-medium">
                    <Wifi className="w-4 h-4" />
                    Status
                  </Label>
                  {isConnected ? (
                    <Badge
                      variant="success"
                      className="animate-pulse border-2 border-green-500"
                    >
                      Connected
                    </Badge>
                  ) : (
                    <Badge
                      variant="secondary"
                      className="border-2 border-border"
                    >
                      Disconnected
                    </Badge>
                  )}
                </div>

                <div className="flex items-center justify-between p-4 bg-muted border-2 border-border">
                  <Label className="text-sm flex items-center gap-2 text-muted-foreground font-medium">
                    <AlertTriangle className="w-4 h-4" />
                    Total Alarms
                  </Label>
                  <Badge
                    variant="destructive"
                    className="text-base px-3 py-1 border-2"
                  >
                    {alarms.length}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Alarms List */}
        <div className="lg:col-span-2">
          <Card className="border-2 border-red-500/30 bg-card">
            <CardHeader className="border-b-2 border-red-500/20 bg-gradient-to-b from-red-500/10 to-transparent">
              <CardTitle className="flex items-center gap-2 text-card-foreground font-bold">
                <Bell className="w-5 h-5 text-red-400" />
                Recent Alarms
                {alarms.length > 0 && (
                  <Badge variant="destructive" className="ml-auto border-2">
                    {alarms.length}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              {alarms.length === 0 ? (
                <div className="text-center py-16 text-muted-foreground">
                  <div className="relative w-24 h-24 mx-auto mb-6">
                    <Bell className="w-full h-full opacity-20" />
                    {isConnected && (
                      <div className="absolute -top-2 -right-2">
                        <Wifi className="w-8 h-8 text-green-500 animate-pulse" />
                      </div>
                    )}
                  </div>
                  <p className="text-lg font-bold text-muted-foreground">
                    No alarms received yet
                  </p>
                  <p className="text-sm mt-2 text-muted-foreground">
                    {isConnected
                      ? "Listening for fall detections..."
                      : 'Click "Connect" to start monitoring'}
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {alarms.map((alarm, idx) => (
                    <div
                      key={idx}
                      className="p-4 border-2 border-red-500/50 bg-gradient-to-r from-red-500/10 to-orange-500/10 hover:border-red-500 hover:shadow-[0_0_20px_rgba(239,68,68,0.3)] transition-all duration-300 animate-in slide-in-from-bottom"
                      style={{ animationDelay: `${idx * 50}ms` }}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Badge
                            variant="destructive"
                            className="animate-pulse border-2"
                          >
                            [FALL DETECTED]
                          </Badge>
                          <span className="text-xs text-muted-foreground font-mono">
                            Frame #{alarm.frameId}
                          </span>
                        </div>
                        <span className="text-sm text-foreground font-bold">
                          {formatTime(alarm.ts)}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                        <div className="p-3 bg-muted border-2 border-border">
                          <span className="text-muted-foreground text-xs block mb-1 uppercase font-medium">
                            Camera ID
                          </span>
                          <span className="font-mono text-foreground font-bold">
                            {alarm.deviceId
                              ? alarm.deviceId.split("-").pop()
                              : "unknown"}
                          </span>
                        </div>

                        <div className="p-3 bg-muted border-2 border-border">
                          <span className="text-muted-foreground text-xs block mb-1 uppercase font-medium">
                            State
                          </span>
                          <span className="font-black text-red-400">
                            {alarm.state}
                          </span>
                        </div>

                        <div className="p-3 bg-muted border-2 border-border">
                          <span className="text-muted-foreground text-xs block mb-1 uppercase font-medium">
                            Confidence
                          </span>
                          <span className="font-black text-red-400">
                            {(alarm.score * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div className="p-3 bg-muted border-2 border-border">
                          <span className="text-muted-foreground text-xs block mb-1 uppercase font-medium">
                            Timestamp
                          </span>
                          <span className="text-foreground font-bold">
                            {new Date(alarm.ts).toLocaleTimeString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
