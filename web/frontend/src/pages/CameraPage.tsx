import { useState, useRef, useEffect } from 'react';
import { Video, Play, Pause, Activity, Wifi } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/components/ui/use-toast';
import Layout from '@/components/Layout';

interface TrackData {
  id: number;
  state: string;
  score: number;
  label?: string; // FALL, NORMAL, UNCERTAIN
  bbox: number[] | null;
  keypoints: number[][] | null;
}

interface Telemetry {
  frameId: number;
  tracks: TrackData[];
  alarm: boolean;
  latency?: number;
}

const FIXED_ROOM_ID = 'fall-detection-room';

export default function CameraPage() {
  const { toast } = useToast();
  const [isStreaming, setIsStreaming] = useState(false);
  const [deviceId] = useState(`camera-${Date.now()}`);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const fps = 15; // Fixed FPS by default

  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(
    document.createElement('canvas'),
  );
  const streamIntervalRef = useRef<number | null>(null);
  const frameCountRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const latestTelemetryRef = useRef<Telemetry | null>(null);

  useEffect(() => {
    return () => {
      stopStreaming();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Loop for continuous overlay rendering
  useEffect(() => {
    if (!isStreaming) return;

    const renderLoop = () => {
      drawOverlay();
      animationFrameRef.current = requestAnimationFrame(renderLoop);
    };

    renderLoop();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isStreaming]);

  const startStreaming = async () => {
    try {
      // Get webcam stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Connect WebSocket
      const ws = new WebSocket(`ws://localhost:9000/ws`);

      ws.onopen = () => {
        console.log('WebSocket connected');

        toast({
          title: 'Connected',
          description: 'WebSocket connected successfully',
        });

        // Register as camera
        ws.send(
          JSON.stringify({
            type: 'register',
            role: 'camera',
            roomId: FIXED_ROOM_ID,
            deviceId: deviceId,
          }),
        );

        setIsStreaming(true);

        // Start sending frames
        streamIntervalRef.current = window.setInterval(() => {
          sendFrame(ws);
        }, 1000 / fps);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // console.log('Received:', data)

        if (data.type === 'telemetry') {
          latestTelemetryRef.current = data;
          setTelemetry(data);

          if (data.alarm) {
            toast({
              title: 'FALL DETECTED!',
              description: 'Alarm triggered.',
              variant: 'destructive',
            });
          }
        } else if (data.type === 'error') {
          console.error('WebSocket error:', data.message);
          toast({
            title: 'Error',
            description: data.message,
            variant: 'destructive',
          });
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        toast({
          title: 'Connection Error',
          description: 'Failed to connect to server',
          variant: 'destructive',
        });
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        setIsStreaming(false);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to start streaming:', error);
      toast({
        title: 'Camera Error',
        description: 'Failed to access webcam',
        variant: 'destructive',
      });
    }
  };

  const stopStreaming = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsStreaming(false);
    setTelemetry(null);
  };

  const sendFrame = (ws: WebSocket) => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Draw video frame to capture canvas (NO MIRROR - backend needs original)
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;

        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = (reader.result as string).split(',')[1];

          const frameMsg = {
            type: 'frame',
            roomId: FIXED_ROOM_ID,
            frameId: frameCountRef.current,
            ts: Date.now(),
            data: base64,
          };

          // console.log(`Sending frame ${frameCountRef.current}`)
          ws.send(JSON.stringify(frameMsg));
          frameCountRef.current++;
        };
        reader.readAsDataURL(blob);
      },
      'image/jpeg',
      0.8,
    );
  };

  // COCO pose skeleton connections
  const SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
  ];

  const drawOverlay = () => {
    if (!canvasRef.current || !videoRef.current) return;

    const data = latestTelemetryRef.current;
    if (!data || !data.tracks) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match display size
    if (
      canvas.width !== video.clientWidth ||
      canvas.height !== video.clientHeight
    ) {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    // Mirror for display
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);

    // Draw all tracks
    data.tracks.forEach((track) => {
      // Color based on state - handles all backend states
      let color = 'rgb(0, 255, 0)'; // NORMAL - Green
      if (track.state === 'FALL' || track.state === 'FALL_CONFIRMED') {
        color = 'rgb(255, 0, 0)'; // Red for fall
      } else if (
        track.state === 'CANDIDATE' ||
        track.state === 'HYPOTHESIS' ||
        track.state === 'VERIFYING'
      ) {
        color = 'rgb(255, 165, 0)'; // Orange for candidate/uncertain
      }

      // Draw Keypoints and Skeleton
      if (track.keypoints) {
        // Draw skeleton connections
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;

        SKELETON.forEach(([idx1, idx2]) => {
          // indices are 1-based in COCO definition array above, convert to 0-based
          const i1 = idx1 - 1;
          const i2 = idx2 - 1;

          if (track.keypoints && track.keypoints[i1] && track.keypoints[i2]) {
            const kp1 = track.keypoints[i1];
            const kp2 = track.keypoints[i2];

            // kp format: [x, y, conf]
            if (kp1[2] > 0.5 && kp2[2] > 0.5) {
              ctx.beginPath();
              ctx.moveTo(kp1[0] * scaleX, kp1[1] * scaleY);
              ctx.lineTo(kp2[0] * scaleX, kp2[1] * scaleY);
              ctx.stroke();
            }
          }
        });

        // Draw keypoint dots
        track.keypoints.forEach(([kx, ky, conf]: number[]) => {
          if (conf > 0.5) {
            const skx = kx * scaleX;
            const sky = ky * scaleY;
            ctx.beginPath();
            ctx.arc(skx, sky, 3, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
          }
        });
      }

      if (!track.bbox) return;

      const [x1, y1, x2, y2] = track.bbox;
      const x = x1 * scaleX;
      const y = y1 * scaleY;
      const w = (x2 - x1) * scaleX;
      const h = (y2 - y1) * scaleY;

      // Bbox
      ctx.shadowBlur = 0;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Label
      const labelHeight = 22;
      const labelText = `ID:${track.id} ${track.state} ${track.score.toFixed(2)}`;

      ctx.font = 'bold 14px sans-serif';
      const textMetrics = ctx.measureText(labelText);
      const labelWidth = textMetrics.width + 10;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - labelHeight, labelWidth, labelHeight);

      // Text (Unmirror for readable text inside mirrored context? No, just draw simple)
      // Actually, we are continuously in mirrored context (scale -1, 1).
      // To draw text that reads correctly, we need to flip the context back temporarily or draw it mirrored-reversed relative to the mirror?
      // Easiest is to save/restore or just scale negative on width.

      ctx.save();
      ctx.translate(x + labelWidth / 2, y - labelHeight / 2); // center of label
      ctx.scale(-1, 1); // flip back
      ctx.fillStyle = '#ffffff';
      if (color === 'rgb(0, 255, 0)' || color === '#22c55e') {
        ctx.fillStyle = '#000000'; // Black text on green
      }
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelText, 0, 0);
      ctx.restore();
    });

    ctx.restore();
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case 'FALL':
      case 'FALL_CONFIRMED':
        return 'destructive';
      case 'CANDIDATE':
      case 'FALL_CANDIDATE':
      case 'HYPOTHESIS':
      case 'VERIFYING':
        return 'warning';
      case 'NORMAL':
      case 'STANDING':
        return 'success';
      default:
        return 'default';
    }
  };

  // Get highest priority track for side panel
  const getPriorityTrack = () => {
    if (!telemetry || !telemetry.tracks || telemetry.tracks.length === 0)
      return null;
    // Priority: FALL > CANDIDATE > NORMAL, then score
    return telemetry.tracks.reduce((prev, current) => {
      const fallStates = ['FALL', 'FALL_CONFIRMED'];
      const candidateStates = ['CANDIDATE', 'HYPOTHESIS', 'VERIFYING'];

      if (fallStates.includes(current.state)) return current;
      if (fallStates.includes(prev.state)) return prev;
      if (
        candidateStates.includes(current.state) &&
        !fallStates.includes(prev.state)
      )
        return current;
      if (current.score > prev.score) return current;
      return prev;
    });
  };

  const priorityTrack = getPriorityTrack();

  return (
    <Layout
      themeColor="blue"
      title="Camera Simulator"
      subtitle="Stream frames for real-time fall detection"
      icon={Video}
      actions={
        isStreaming ? (
          <Button
            variant="destructive"
            onClick={stopStreaming}
            className="w-full md:w-auto border-2 border-red-500 hover:shadow-[0_0_20px_rgba(239,68,68,0.5)] transition-all duration-300 text-foreground"
          >
            <Pause className="w-5 h-5 mr-2" />
            Stop Streaming
          </Button>
        ) : (
          <Button
            onClick={startStreaming}
            className="w-full md:w-auto border-2 border-blue-500 bg-blue-500/10 hover:bg-blue-500/20 hover:shadow-[0_0_20px_rgba(59,130,246,0.5)] transition-all duration-300 text-foreground"
          >
            <Play className="w-5 h-5 mr-2" />
            Start Streaming
          </Button>
        )
      }
    >
      <div className="grid lg:grid-cols-3 gap-4 md:gap-6">
        <div className="lg:col-span-2">
          <Card className="overflow-hidden border-2 border-blue-500/30 bg-card hover:border-blue-500/50 transition-all duration-300">
            <CardHeader className="border-b-2 border-blue-500/20 bg-gradient-to-b from-blue-500/10 to-transparent">
              <CardTitle className="flex items-center gap-2 text-card-foreground font-bold">
                <Video className="w-5 h-5 text-blue-400" />
                LIVE FEED
                {isStreaming && (
                  <Badge
                    variant="success"
                    className="ml-auto animate-pulse border-2 border-green-500"
                  >
                    <Wifi className="w-3 h-3 mr-1" />
                    LIVE
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="relative bg-black aspect-video">
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  style={{ transform: 'scaleX(-1)' }} // Mirror video
                  muted
                />
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full"
                />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4 md:space-y-6">
          {!isStreaming && (
            <Card className="border-2 border-border bg-card">
              <CardHeader className="border-b-2 border-border">
                <CardTitle className="text-lg text-card-foreground font-bold uppercase">
                  Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 pt-6">
                <div className="p-4 bg-blue-500/10 border-2 border-blue-500/30">
                  <Label className="text-sm font-bold text-blue-400 uppercase">
                    Room ID
                  </Label>
                  <p className="text-sm text-blue-300 mt-1 font-mono">
                    {FIXED_ROOM_ID}
                  </p>
                </div>
                <div>
                  <Label className="text-muted-foreground font-medium">
                    Device ID
                  </Label>
                  <Input
                    value={deviceId}
                    disabled
                    className="font-mono text-sm bg-muted border-2 border-border text-muted-foreground mt-2"
                  />
                </div>
              </CardContent>
            </Card>
          )}

          {isStreaming && priorityTrack && (
            <Card className="border-2 border-blue-500/30 bg-card animate-in slide-in-from-right duration-500">
              <CardHeader className="border-b-2 border-blue-500/20 bg-gradient-to-b from-blue-500/10 to-transparent">
                <CardTitle className="text-lg flex items-center gap-2 text-card-foreground font-bold">
                  <Activity className="w-5 h-5 text-blue-400" />
                  Live Telemetry
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 pt-6">
                <div>
                  <Label className="text-sm text-muted-foreground uppercase font-medium">
                    Status (ID: {priorityTrack.id})
                  </Label>
                  <div className="mt-2 text-4xl font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                    {(priorityTrack.score * 100).toFixed(1)}%
                  </div>
                  <Badge
                    variant={getStateColor(priorityTrack.state)}
                    className="mt-3 text-base px-4 py-2 border-2"
                  >
                    {priorityTrack.state}
                  </Badge>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-4 bg-muted border-2 border-border">
                    <Label className="text-xs text-muted-foreground uppercase font-medium">
                      Active Tracks
                    </Label>
                    <p className="text-2xl font-bold mt-2 text-foreground">
                      {telemetry?.tracks.length || 0}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </Layout>
  );
}
