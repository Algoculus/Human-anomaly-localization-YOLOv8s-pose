import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Users, Bell, Image as ImageIcon, Wifi, AlertTriangle, Video } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useToast } from '@/components/ui/use-toast'

interface AlarmData {
  roomId: string
  deviceId: string
  frameId: number
  ts: number
  snapshot: string // base64 JPEG
  state: string
  score: number
}

const FIXED_ROOM_ID = 'fall-detection-room'

export default function ReceiverPage() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [isConnected, setIsConnected] = useState(false)
  const [deviceId] = useState(`receiver-${Date.now()}`)
  const [alarms, setAlarms] = useState<AlarmData[]>([])
  
  const wsRef = useRef<WebSocket | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    // Create alarm audio
    audioRef.current = new Audio('/alarm.mp3')
    
    return () => {
      disconnect()
    }
  }, [])

  const connect = () => {
    const ws = new WebSocket(`ws://localhost:9000/ws`)
    
    ws.onopen = () => {
      console.log('WebSocket connected')
      
      // Register as receiver
      ws.send(JSON.stringify({
        type: 'register',
        role: 'receiver',
        roomId: FIXED_ROOM_ID,
        deviceId: deviceId
      }))

      setIsConnected(true)
      
      toast({
        title: '✅ Connected',
        description: `Monitoring room: ${FIXED_ROOM_ID}`,
      })
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'alarm') {
        handleAlarm(data)
      } else if (data.type === 'error') {
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
      setIsConnected(false)
    }

    wsRef.current = ws
  }

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
  }

  const handleAlarm = (data: AlarmData) => {
    // Add to alarms list
    setAlarms(prev => [data, ...prev].slice(0, 20)) // Keep last 20
    
    // Play alarm sound
    if (audioRef.current) {
      audioRef.current.play().catch(e => console.error('Audio play failed:', e))
    }
    
    // Show toast notification
    toast({
      title: '🚨 FALL DETECTED!',
      description: `Camera: ${data.deviceId} | Frame: ${data.frameId}`,
      variant: 'destructive'
    })

    // Request notification permission and send browser notification
    if (Notification.permission === 'granted') {
      new Notification('Fall Detected!', {
        body: `Camera ${data.deviceId} detected a fall`,
        icon: '/fall-icon.png',
        badge: '/badge.png'
      })
    } else if (Notification.permission !== 'denied') {
      Notification.requestPermission()
    }
  }

  const clearAlarms = () => {
    setAlarms([])
    toast({
      title: 'Cleared',
      description: 'All alarms cleared',
    })
  }

  const formatTime = (ts: number) => {
    return new Date(ts).toLocaleTimeString()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 md:mb-8 gap-4">
          <div className="flex items-center gap-3 md:gap-4">
            <Button variant="ghost" size="icon" onClick={() => navigate('/')} className="hover:bg-white/50 dark:hover:bg-gray-800/50">
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent">
                Alert Receiver
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">Monitor fall alerts in real-time</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 w-full md:w-auto">
            {alarms.length > 0 && (
              <Button 
                variant="outline" 
                onClick={clearAlarms}
                className="flex-1 md:flex-none"
              >
                Clear All
              </Button>
            )}
            {isConnected ? (
              <Button 
                variant="destructive" 
                onClick={disconnect}
                className="flex-1 md:flex-none shadow-lg hover:shadow-xl transition-all duration-300"
              >
                Disconnect
              </Button>
            ) : (
              <Button 
                onClick={connect}
                className="flex-1 md:flex-none bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700 shadow-lg hover:shadow-xl transition-all duration-300"
              >
                <Users className="w-5 h-5 mr-2" />
                Connect
              </Button>
            )}
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-4 md:gap-6">
          {/* Configuration Panel */}
          <div>
            <Card className="shadow-lg border-2 border-gray-200 dark:border-gray-700">
              <CardHeader className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-gray-800 dark:to-gray-700">
                <CardTitle className="text-lg">Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 pt-6">
                <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                  <Label className="text-sm font-medium text-red-900 dark:text-red-300">Room ID (Fixed)</Label>
                  <p className="text-sm text-red-700 dark:text-red-400 mt-1 font-mono">{FIXED_ROOM_ID}</p>
                  <p className="text-xs text-red-600 dark:text-red-500 mt-2">All devices use the same room</p>
                </div>
                <div>
                  <Label>Device ID</Label>
                  <Input 
                    value={deviceId} 
                    disabled
                    className="font-mono text-sm bg-gray-50 dark:bg-gray-800"
                  />
                </div>

                <div className="pt-4 border-t space-y-3">
                  <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <Label className="text-sm flex items-center gap-2">
                      <Wifi className="w-4 h-4" />
                      Status
                    </Label>
                    {isConnected ? (
                      <Badge className="bg-green-500 animate-pulse">
                        Connected
                      </Badge>
                    ) : (
                      <Badge variant="secondary">
                        Disconnected
                      </Badge>
                    )}
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <Label className="text-sm flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" />
                      Total Alarms
                    </Label>
                    <Badge variant="destructive" className="text-base px-3 py-1">
                      {alarms.length}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Alarms Grid */}
          <div className="lg:col-span-2">
            <Card className="shadow-xl border-2 border-gray-200 dark:border-gray-700">
              <CardHeader className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-gray-800 dark:to-gray-700">
                <CardTitle className="flex items-center gap-2">
                  <Bell className="w-5 h-5 text-red-600" />
                  Recent Alarms
                  {alarms.length > 0 && (
                    <Badge variant="destructive" className="ml-auto">
                      {alarms.length}
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                {alarms.length === 0 ? (
                  <div className="text-center py-16 text-gray-500">
                    <div className="relative w-24 h-24 mx-auto mb-6">
                      <ImageIcon className="w-full h-full opacity-20" />
                      {isConnected && (
                        <div className="absolute -top-2 -right-2">
                          <Wifi className="w-8 h-8 text-green-500 animate-pulse" />
                        </div>
                      )}
                    </div>
                    <p className="text-lg font-medium">No alarms received yet</p>
                    <p className="text-sm mt-2">
                      {isConnected 
                        ? 'Listening for fall detections...' 
                        : 'Click "Connect" to start monitoring'}
                    </p>
                  </div>
                ) : (
                  <div className="grid md:grid-cols-2 gap-4">
                    {alarms.map((alarm, idx) => (
                      <Card 
                        key={idx} 
                        className="border-2 border-red-300 dark:border-red-800 overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300 animate-in slide-in-from-bottom"
                        style={{ animationDelay: `${idx * 50}ms` }}
                      >
                        <CardContent className="p-0">
                          {/* Snapshot */}
                          <div className="relative">
                            <img
                              src={`data:image/jpeg;base64,${alarm.snapshot}`}
                              alt="Fall snapshot"
                              className="w-full aspect-video object-cover"
                            />
                            <div className="absolute top-2 right-2">
                              <Badge variant="destructive" className="animate-pulse">
                                🚨 FALL
                              </Badge>
                            </div>
                          </div>
                          
                          {/* Details */}
                          <div className="p-4 space-y-2 text-sm bg-gradient-to-br from-red-50/50 to-orange-50/50 dark:from-red-900/20 dark:to-orange-900/20">
                            <div className="flex items-center justify-between">
                              <span className="text-gray-600 dark:text-gray-400 flex items-center gap-1">
                                <Video className="w-3 h-3" />
                                Camera
                              </span>
                              <span className="font-mono text-xs bg-white dark:bg-gray-800 px-2 py-1 rounded">
                                {alarm.deviceId ? alarm.deviceId.split('-').pop() : 'unknown'}
                              </span>
                            </div>
                            
                            <div className="flex items-center justify-between">
                              <span className="text-gray-600 dark:text-gray-400">Time</span>
                              <span className="font-medium">{formatTime(alarm.ts)}</span>
                            </div>
                            
                            <div className="flex items-center justify-between">
                              <span className="text-gray-600 dark:text-gray-400">Confidence</span>
                              <Badge variant="destructive" className="font-bold">
                                {(alarm.score * 100).toFixed(1)}%
                              </Badge>
                            </div>
                            
                            <div className="flex items-center justify-between">
                              <span className="text-gray-600 dark:text-gray-400">Frame</span>
                              <span className="font-mono text-xs bg-white dark:bg-gray-800 px-2 py-1 rounded">
                                #{alarm.frameId}
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
