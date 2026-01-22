import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Video, Play, Pause, AlertCircle, Activity, Wifi } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useToast } from '@/components/ui/use-toast'

interface Telemetry {
  frameId: number
  state: string
  score: number
  bbox: number[] | null
  keypoints: number[][] | null
  latency: number
}

const FIXED_ROOM_ID = 'fall-detection-room'

export default function CameraPage() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [isStreaming, setIsStreaming] = useState(false)
  const [deviceId] = useState(`camera-${Date.now()}`)
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const fps = 15 // Fixed FPS for consistent performance
  
  const wsRef = useRef<WebSocket | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const captureCanvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'))
  const streamIntervalRef = useRef<number | null>(null)
  const frameCountRef = useRef(0)
  const animationFrameRef = useRef<number | null>(null)
  const latestTelemetryRef = useRef<Telemetry | null>(null)

  useEffect(() => {
    return () => {
      stopStreaming()
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])

  // Continuous overlay rendering loop
  useEffect(() => {
    if (!isStreaming) return

    const renderLoop = () => {
      drawOverlay()
      animationFrameRef.current = requestAnimationFrame(renderLoop)
    }

    renderLoop()

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [isStreaming])

  const startStreaming = async () => {
    try {
      // Get webcam stream
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      // Connect WebSocket
      const ws = new WebSocket(`ws://localhost:9000/ws`)
      
      ws.onopen = () => {
        console.log('WebSocket connected')
        
        toast({
          title: '✅ Connected',
          description: 'WebSocket connected successfully',
        })
        
        // Register as camera
        ws.send(JSON.stringify({
          type: 'register',
          role: 'camera',
          roomId: FIXED_ROOM_ID,
          deviceId: deviceId
        }))

        setIsStreaming(true)
        
        // Start sending frames
        streamIntervalRef.current = window.setInterval(() => {
          sendFrame(ws)
        }, 1000 / fps)
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        console.log('📥 Received WebSocket message:', data)
        
        if (data.type === 'telemetry') {
          console.log('📊 Telemetry data:', {
            frameId: data.frameId,
            state: data.state,
            score: data.score,
            bbox: data.bbox,
            keypoints: data.keypoints ? `${data.keypoints.length} points` : 'none',
            latency: data.latency
          })
          latestTelemetryRef.current = data
          setTelemetry(data)
        } else if (data.type === 'error') {
          console.error('❌ WebSocket error:', data.message)
          toast({
            title: 'Error',
            description: data.message,
            variant: 'destructive'
          })
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        toast({
          title: 'Connection Error',
          description: 'Failed to connect to server',
          variant: 'destructive'
        })
      }

      ws.onclose = () => {
        console.log('WebSocket closed')
        setIsStreaming(false)
      }

      wsRef.current = ws

    } catch (error) {
      console.error('Failed to start streaming:', error)
      toast({
        title: 'Camera Error',
        description: 'Failed to access webcam',
        variant: 'destructive'
      })
    }
  }

  const stopStreaming = () => {
    // Stop WebSocket
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    // Stop frame sending
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current)
      streamIntervalRef.current = null
    }

    // Stop video stream
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }

    setIsStreaming(false)
    setTelemetry(null)
  }

  const sendFrame = (ws: WebSocket) => {
    if (!videoRef.current) return

    const video = videoRef.current
    const canvas = captureCanvasRef.current
    const ctx = canvas.getContext('2d')
    
    if (!ctx) return

    // Draw video frame to capture canvas (NO MIRROR - backend needs original)
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    
    // Draw original frame (not mirrored) for YOLO processing
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert to base64
    canvas.toBlob((blob) => {
      if (!blob) return
      
      const reader = new FileReader()
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(',')[1]
        
        const frameMsg = {
          type: 'frame',
          roomId: FIXED_ROOM_ID,
          frameId: frameCountRef.current,
          ts: Date.now(),
          data: base64
        }
        
        console.log(`📤 Sending frame ${frameCountRef.current} (${base64.length} bytes)`)
        ws.send(JSON.stringify(frameMsg))
        frameCountRef.current++
      }
      reader.readAsDataURL(blob)
    }, 'image/jpeg', 0.8)
  }

  const drawOverlay = () => {
    if (!canvasRef.current || !videoRef.current) return

    const data = latestTelemetryRef.current
    if (!data) return

    const canvas = canvasRef.current
    const video = videoRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Match canvas size to video display size
    if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
      canvas.width = video.clientWidth
      canvas.height = video.clientHeight
    }

    // Clear previous overlay
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Calculate scale factors
    const scaleX = canvas.width / video.videoWidth
    const scaleY = canvas.height / video.videoHeight

    // Save context and flip horizontally (mirror entire canvas to match video)
    ctx.save()
    ctx.scale(-1, 1)
    ctx.translate(-canvas.width, 0)

    // Draw bounding box (using original coordinates, canvas is already mirrored)
    if (data.bbox) {
      const [x1, y1, x2, y2] = data.bbox
      const x = x1 * scaleX
      const y = y1 * scaleY
      const w = (x2 - x1) * scaleX
      const h = (y2 - y1) * scaleY
      
      // Determine color based on state
      let color = '#22c55e' // Default green for NORMAL/STANDING
      if (data.state === 'FALL' || data.state === 'FALL_CONFIRMED') {
        color = '#ef4444' // Red for FALL
      } else if (data.state === 'CANDIDATE' || data.state === 'FALL_CANDIDATE') {
        color = '#eab308' // Yellow for CANDIDATE
      }
      
      // Draw bounding box with glow effect
      ctx.shadowBlur = 10
      ctx.shadowColor = color
      ctx.strokeStyle = color
      ctx.lineWidth = 3
      ctx.strokeRect(x, y, w, h)
      ctx.shadowBlur = 0

      // Draw label background with rounded corners
      const labelHeight = 35
      const labelWidth = 160
      ctx.fillStyle = color
      ctx.beginPath()
      
      // Manual rounded rect (for browser compatibility)
      const radius = 5
      ctx.moveTo(x + radius, y - labelHeight - 5)
      ctx.lineTo(x + labelWidth - radius, y - labelHeight - 5)
      ctx.quadraticCurveTo(x + labelWidth, y - labelHeight - 5, x + labelWidth, y - labelHeight - 5 + radius)
      ctx.lineTo(x + labelWidth, y - 5 - radius)
      ctx.quadraticCurveTo(x + labelWidth, y - 5, x + labelWidth - radius, y - 5)
      ctx.lineTo(x + radius, y - 5)
      ctx.quadraticCurveTo(x, y - 5, x, y - 5 - radius)
      ctx.lineTo(x, y - labelHeight - 5 + radius)
      ctx.quadraticCurveTo(x, y - labelHeight - 5, x + radius, y - labelHeight - 5)
      ctx.closePath()
      ctx.fill()
      
      // Draw label text (unmirror text so it reads correctly)
      ctx.save()
      ctx.scale(-1, 1) // Unmirror text
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 16px sans-serif'
      ctx.fillText(`${data.state}`, -(x + labelWidth - 8), y - 18)
      ctx.font = '14px sans-serif'
      ctx.fillText(`Score: ${(data.score * 100).toFixed(1)}%`, -(x + labelWidth - 8), y - 3)
      ctx.restore()
    }

    // Draw keypoints (using original coordinates, canvas is already mirrored)
    if (data.keypoints) {
      data.keypoints.forEach(([x, y, conf]: number[]) => {
        if (conf > 0.5) {
          const scaledX = x * scaleX
          const scaledY = y * scaleY
          
          // Draw keypoint with glow
          ctx.shadowBlur = 5
          ctx.shadowColor = '#3b82f6'
          ctx.beginPath()
          ctx.arc(scaledX, scaledY, 4, 0, 2 * Math.PI)
          ctx.fillStyle = '#3b82f6'
          ctx.fill()
          
          // Draw keypoint border
          ctx.strokeStyle = '#ffffff'
          ctx.lineWidth = 2
          ctx.stroke()
          ctx.shadowBlur = 0
        }
      })
      
      // Draw skeleton connections (if available)
      const connections = [
        [0, 1], [0, 2], [1, 3], [2, 4], // Head
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
        [5, 11], [6, 12], [11, 12], // Torso
        [11, 13], [13, 15], [12, 14], [14, 16] // Legs
      ]
      
      ctx.strokeStyle = '#60a5fa'
      ctx.lineWidth = 2
      connections.forEach(([idx1, idx2]) => {
        if (data.keypoints && data.keypoints[idx1] && data.keypoints[idx2]) {
          const [x1, y1, conf1] = data.keypoints[idx1]
          const [x2, y2, conf2] = data.keypoints[idx2]
          if (conf1 > 0.5 && conf2 > 0.5) {
            const scaledX1 = x1 * scaleX
            const scaledY1 = y1 * scaleY
            const scaledX2 = x2 * scaleX
            const scaledY2 = y2 * scaleY
            ctx.beginPath()
            ctx.moveTo(scaledX1, scaledY1)
            ctx.lineTo(scaledX2, scaledY2)
            ctx.stroke()
          }
        }
      })
    }

    // Restore context (undo mirror)
    ctx.restore()
  }

  const getStateColor = (state: string) => {
    switch (state) {
      case 'FALL':
      case 'FALL_CONFIRMED': return 'destructive' // Red
      case 'CANDIDATE':
      case 'FALL_CANDIDATE': return 'warning' // Yellow
      case 'NORMAL':
      case 'STANDING': return 'success' // Green
      case 'LYING': return 'secondary'
      default: return 'default'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-800 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 md:mb-8 gap-4">
          <div className="flex items-center gap-3 md:gap-4">
            <Button variant="ghost" size="icon" onClick={() => navigate('/')} className="hover:bg-gray-700 text-gray-300 hover:text-white">
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Camera Simulator
              </h1>
              <p className="text-sm text-gray-400">Stream frames for real-time fall detection</p>
            </div>
          </div>
          
          {isStreaming ? (
            <Button 
              variant="destructive" 
              onClick={stopStreaming}
              className="w-full md:w-auto shadow-lg hover:shadow-xl transition-all duration-300"
            >
              <Pause className="w-5 h-5 mr-2" />
              Stop Streaming
            </Button>
          ) : (
            <Button 
              onClick={startStreaming}
              className="w-full md:w-auto bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all duration-300"
            >
              <Play className="w-5 h-5 mr-2" />
              Start Streaming
            </Button>
          )}
        </div>

        <div className="grid lg:grid-cols-3 gap-4 md:gap-6">
          {/* Video Feed */}
          <div className="lg:col-span-2">
            <Card className="overflow-hidden shadow-2xl border-2 border-gray-700 bg-gray-800/90 backdrop-blur-sm hover:shadow-2xl transition-shadow duration-300">
              <CardHeader className="bg-gradient-to-r from-gray-800 to-slate-800 border-b border-gray-700">
                <CardTitle className="flex items-center gap-2 text-gray-200">
                  <Video className="w-5 h-5 text-blue-400" />
                  Live Feed
                  {isStreaming && (
                    <Badge variant="default" className="ml-auto bg-green-500 animate-pulse shadow-lg shadow-green-500/50">
                      <Wifi className="w-3 h-3 mr-1" />
                      LIVE
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="relative bg-black rounded-b-lg overflow-hidden aspect-video">
                  <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    style={{ transform: 'scaleX(-1)' }}
                    muted
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full"
                  />
                  
                  {!isStreaming && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-gray-900/90 to-gray-800/90 backdrop-blur-sm">
                      <Play className="w-16 h-16 text-white mb-4 opacity-50" />
                      <p className="text-white text-lg font-medium">Click "Start Streaming" to begin</p>
                      <p className="text-gray-400 text-sm mt-2">Your camera will be accessed for fall detection</p>
                    </div>
                  )}
                  
                  {/* Frame Counter Overlay */}
                  {isStreaming && telemetry && (
                    <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-md px-3 py-2 rounded-lg border border-white/20">
                      <div className="flex items-center gap-2 text-white text-sm">
                        <Activity className="w-4 h-4" />
                        <span>Frame #{telemetry.frameId}</span>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Telemetry Panel */}
          <div className="space-y-4 md:space-y-6">
            {/* Configuration */}
            {!isStreaming && (
              <Card className="shadow-2xl border-2 border-gray-700 bg-gray-800/90 backdrop-blur-sm animate-in slide-in-from-right duration-500">
                <CardHeader className="bg-gradient-to-r from-gray-800 to-slate-800 border-b border-gray-700">
                  <CardTitle className="text-lg text-gray-200">Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 pt-6">
                  <div className="p-4 bg-blue-950/30 rounded-lg border border-blue-800">
                    <Label className="text-sm font-medium text-blue-300">Room ID (Fixed)</Label>
                    <p className="text-sm text-blue-400 mt-1 font-mono">{FIXED_ROOM_ID}</p>
                    <p className="text-xs text-blue-500 mt-2">All devices use the same room</p>
                  </div>
                  <div>
                    <Label className="text-gray-300">Device ID</Label>
                    <Input 
                      value={deviceId} 
                      disabled
                      className="font-mono text-sm bg-gray-900/50 border-gray-700 text-gray-300"
                    />
                  </div>
                  <div>
                    <Label className="text-gray-300">Frame Rate</Label>
                    <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700 mt-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-400">Fixed FPS</span>
                        <Badge variant="outline" className="border-green-600 text-green-400 bg-green-950/30">{fps} fps</Badge>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">Optimized for real-time detection</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Live Telemetry */}
            {isStreaming && telemetry && (
              <Card className="shadow-2xl border-2 border-gray-700 bg-gray-800/90 backdrop-blur-sm animate-in slide-in-from-right duration-500">
                <CardHeader className="bg-gradient-to-r from-gray-800 to-slate-800 border-b border-gray-700">
                  <CardTitle className="text-lg flex items-center gap-2 text-gray-200">
                    <Activity className="w-5 h-5 text-blue-400" />
                    Live Telemetry
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 pt-6">
                  <div>
                    <Label className="text-sm text-gray-400">Detection State</Label>
                    <div className="mt-2">
                      <Badge 
                        variant={getStateColor(telemetry.state)}
                        className="text-base px-4 py-2 animate-in fade-in duration-300 shadow-lg"
                      >
                        {telemetry.state}
                      </Badge>
                    </div>
                  </div>

                  <div className="p-4 bg-gradient-to-r from-blue-950/30 to-purple-950/30 rounded-lg border border-blue-800">
                    <Label className="text-sm text-gray-400">Confidence Score</Label>
                    <div className="mt-2 text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                      {(telemetry.score * 100).toFixed(1)}%
                    </div>
                    <div className="mt-2 w-full bg-gray-900 rounded-full h-2 overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-300 shadow-lg shadow-blue-500/50"
                        style={{ width: `${telemetry.score * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700">
                      <Label className="text-xs text-gray-400">Frame ID</Label>
                      <p className="text-lg font-semibold mt-1 text-gray-200">{telemetry.frameId}</p>
                    </div>
                    <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700">
                      <Label className="text-xs text-gray-400">Latency</Label>
                      <p className="text-lg font-semibold mt-1 text-gray-200">{telemetry.latency}ms</p>
                    </div>
                  </div>

                  {(telemetry.state === 'FALL' || telemetry.state === 'FALL_CONFIRMED') && (
                    <div className="p-4 bg-gradient-to-r from-red-950/40 to-orange-950/40 rounded-lg border-2 border-red-700 animate-pulse shadow-2xl shadow-red-500/30">
                      <div className="flex items-center gap-3">
                        <AlertCircle className="w-6 h-6 text-red-400" />
                        <div>
                          <span className="font-bold text-red-300 text-lg">FALL DETECTED!</span>
                          <p className="text-sm mt-1 text-red-400">
                            Alarm broadcast to all receivers
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
