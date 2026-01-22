import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Video, Play, Pause, AlertCircle, Activity, Wifi } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useToast } from '@/components/ui/use-toast'

interface TrackData {
  id: number
  state: string
  score: number
  bbox: number[] | null
  keypoints: number[][] | null
}

interface Telemetry {
  frameId: number
  tracks: TrackData[]
  alarm: boolean
  latency?: number
}

const FIXED_ROOM_ID = 'fall-detection-room'

export default function CameraPage() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [isStreaming, setIsStreaming] = useState(false)
  const [deviceId] = useState(`camera-${Date.now()}`)
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const fps = 15 // Fixed FPS by default

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

  // Loop for continuous overlay rendering
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
      const ws = new WebSocket(`ws://localhost:4611/ws`)

      ws.onopen = () => {
        console.log('WebSocket connected')

        toast({
          title: 'Connected',
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
        // console.log('Received:', data)

        if (data.type === 'telemetry') {
          latestTelemetryRef.current = data
          setTelemetry(data)

          if (data.alarm) {
            toast({
              title: 'FALL DETECTED!',
              description: 'Alarm triggered.',
              variant: 'destructive'
            })
          }
        } else if (data.type === 'error') {
          console.error('WebSocket error:', data.message)
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
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current)
      streamIntervalRef.current = null
    }

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

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

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

        // console.log(`Sending frame ${frameCountRef.current}`)
        ws.send(JSON.stringify(frameMsg))
        frameCountRef.current++
      }
      reader.readAsDataURL(blob)
    }, 'image/jpeg', 0.8)
  }

  const drawOverlay = () => {
    if (!canvasRef.current || !videoRef.current) return

    const data = latestTelemetryRef.current
    if (!data || !data.tracks) return

    const canvas = canvasRef.current
    const video = videoRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Match display size
    if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
      canvas.width = video.clientWidth
      canvas.height = video.clientHeight
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scaleX = canvas.width / video.videoWidth
    const scaleY = canvas.height / video.videoHeight

    // Mirror for display
    ctx.save()
    ctx.scale(-1, 1)
    ctx.translate(-canvas.width, 0)

    // Draw all tracks
    data.tracks.forEach(track => {
      if (!track.bbox) return

      const [x1, y1, x2, y2] = track.bbox
      const x = x1 * scaleX
      const y = y1 * scaleY
      const w = (x2 - x1) * scaleX
      const h = (y2 - y1) * scaleY

      // Color based on state
      let color = '#22c55e' // NORMAL/STANDING
      if (track.state === 'FALL' || track.state === 'FALL_CONFIRMED') {
        color = '#ef4444' // Red
      } else if (track.state === 'CANDIDATE' || track.state === 'FALL_CANDIDATE') {
        color = '#eab308' // Yellow
      }

      // Bbox
      ctx.shadowBlur = 10
      ctx.shadowColor = color
      ctx.strokeStyle = color
      ctx.lineWidth = 3
      ctx.strokeRect(x, y, w, h)
      ctx.shadowBlur = 0

      // Label
      const labelHeight = 35
      const labelWidth = 180
      ctx.fillStyle = color
      ctx.beginPath()

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

      // Text (Unmirror)
      ctx.save()
      ctx.scale(-1, 1)
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 16px sans-serif'

      ctx.fillText(`ID:${track.id} ${track.state}`, -(x + labelWidth - 8), y - 18)
      ctx.font = '14px sans-serif'
      ctx.fillText(`Score: ${(track.score * 100).toFixed(1)}%`, -(x + labelWidth - 8), y - 3)
      ctx.restore()

      // Keypoints
      if (track.keypoints) {
        track.keypoints.forEach(([kx, ky, conf]: number[]) => {
          if (conf > 0.5) {
            const skx = kx * scaleX
            const sky = ky * scaleY
            ctx.beginPath()
            ctx.arc(skx, sky, 4, 0, 2 * Math.PI)
            ctx.fillStyle = '#3b82f6'
            ctx.fill()
          }
        })
        // Simple skeleton TODO: Add connections loop if needed, but bbox+state is enough for now
      }
    })

    ctx.restore()
  }

  const getStateColor = (state: string) => {
    switch (state) {
      case 'FALL':
      case 'FALL_CONFIRMED': return 'destructive'
      case 'CANDIDATE':
      case 'FALL_CANDIDATE': return 'warning'
      case 'NORMAL':
      case 'STANDING': return 'success'
      default: return 'default'
    }
  }

  // Get highest priority track for side panel
  const getPriorityTrack = () => {
    if (!telemetry || !telemetry.tracks || telemetry.tracks.length === 0) return null
    // Priority: FALL > CANDIDATE > NORMAL, then score
    return telemetry.tracks.reduce((prev, current) => {
      if (current.state === 'FALL_CONFIRMED') return current
      if (prev.state === 'FALL_CONFIRMED') return prev
      if (current.state === 'CANDIDATE' && prev.state !== 'FALL_CONFIRMED') return current
      if (current.score > prev.score) return current
      return prev
    })
  }

  const priorityTrack = getPriorityTrack()

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-800 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
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
          <div className="lg:col-span-2">
            <Card className="overflow-hidden shadow-2xl border-2 border-gray-700 bg-gray-800/90 backdrop-blur-sm">
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
              <Card className="shadow-2xl border-2 border-gray-700 bg-gray-800/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-lg text-gray-200">Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 pt-6">
                  <div className="p-4 bg-blue-950/30 rounded-lg border border-blue-800">
                    <Label className="text-sm font-medium text-blue-300">Room ID</Label>
                    <p className="text-sm text-blue-400 mt-1 font-mono">{FIXED_ROOM_ID}</p>
                  </div>
                  <div>
                    <Label className="text-gray-300">Device ID</Label>
                    <Input value={deviceId} disabled className="font-mono text-sm bg-gray-900/50 border-gray-700 text-gray-300" />
                  </div>
                </CardContent>
              </Card>
            )}

            {isStreaming && priorityTrack && (
              <Card className="shadow-2xl border-2 border-gray-700 bg-gray-800/90 backdrop-blur-sm animate-in slide-in-from-right duration-500">
                <CardHeader className="bg-gradient-to-r from-gray-800 to-slate-800 border-b border-gray-700">
                  <CardTitle className="text-lg flex items-center gap-2 text-gray-200">
                    <Activity className="w-5 h-5 text-blue-400" />
                    Live Telemetry (Highest Priority)
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 pt-6">
                  <div>
                    <Label className="text-sm text-gray-400">Status (ID: {priorityTrack.id})</Label>
                    <div className="mt-2 text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                      {(priorityTrack.score * 100).toFixed(1)}%
                    </div>
                    <Badge
                      variant={getStateColor(priorityTrack.state)}
                      className="mt-2 text-base px-4 py-2"
                    >
                      {priorityTrack.state}
                    </Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700">
                      <Label className="text-xs text-gray-400">Active Tracks</Label>
                      <p className="text-lg font-semibold mt-1 text-gray-200">{telemetry?.tracks.length || 0}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
