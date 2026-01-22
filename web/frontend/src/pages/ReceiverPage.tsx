import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Users, Bell, Image as Wifi, AlertTriangle } from 'lucide-react'
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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-800 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 md:mb-8 gap-4">
          <div className="flex items-center gap-3 md:gap-4">
            <Button variant="ghost" size="icon" onClick={() => navigate('/')} className="hover:bg-gray-700 text-gray-300 hover:text-white">
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent">
                Alert Receiver
              </h1>
              <p className="text-sm text-gray-400">Monitor fall alerts in real-time</p>
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
            <Card className="shadow-2xl border-2 border-gray-700 bg-gray-800/90 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-gray-800 to-slate-800 border-b border-gray-700">
                <CardTitle className="text-lg text-gray-200">Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 pt-6">
                <div className="p-4 bg-red-950/30 rounded-lg border border-red-800">
                  <Label className="text-sm font-medium text-red-300">Room ID (Fixed)</Label>
                  <p className="text-sm text-red-400 mt-1 font-mono">{FIXED_ROOM_ID}</p>
                  <p className="text-xs text-red-500 mt-2">All devices use the same room</p>
                </div>
                <div>
                  <Label className="text-gray-300">Device ID</Label>
                  <Input 
                    value={deviceId} 
                    disabled
                    className="font-mono text-sm bg-gray-900/50 border-gray-700 text-gray-300"
                  />
                </div>

                <div className="pt-4 border-t border-gray-700 space-y-3">
                  <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg border border-gray-700">
                    <Label className="text-sm flex items-center gap-2 text-gray-300">
                      <Wifi className="w-4 h-4" />
                      Status
                    </Label>
                    {isConnected ? (
                      <Badge className="bg-green-500 animate-pulse shadow-lg shadow-green-500/50">
                        Connected
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="bg-gray-700 text-gray-300">
                        Disconnected
                      </Badge>
                    )}
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg border border-gray-700">
                    <Label className="text-sm flex items-center gap-2 text-gray-300">
                      <AlertTriangle className="w-4 h-4" />
                      Total Alarms
                    </Label>
                    <Badge variant="destructive" className="text-base px-3 py-1 shadow-lg">
                      {alarms.length}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Alarms List */}
          <div className="lg:col-span-2">
            <Card className="shadow-2xl border-2 border-gray-700 bg-gray-800/90 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-gray-800 to-slate-800 border-b border-gray-700">
                <CardTitle className="flex items-center gap-2 text-gray-200">
                  <Bell className="w-5 h-5 text-red-400" />
                  Recent Alarms
                  {alarms.length > 0 && (
                    <Badge variant="destructive" className="ml-auto shadow-lg">
                      {alarms.length}
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                {alarms.length === 0 ? (
                  <div className="text-center py-16 text-gray-400">
                    <div className="relative w-24 h-24 mx-auto mb-6">
                      <Bell className="w-full h-full opacity-20" />
                      {isConnected && (
                        <div className="absolute -top-2 -right-2">
                          <Wifi className="w-8 h-8 text-green-500 animate-pulse" />
                        </div>
                      )}
                    </div>
                    <p className="text-lg font-medium text-gray-300">No alarms received yet</p>
                    <p className="text-sm mt-2 text-gray-500">
                      {isConnected 
                        ? 'Listening for fall detections...' 
                        : 'Click "Connect" to start monitoring'}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {alarms.map((alarm, idx) => (
                      <div 
                        key={idx} 
                        className="p-4 border-2 border-red-700 rounded-lg bg-gradient-to-r from-red-950/40 to-orange-950/40 hover:border-red-600 transition-all duration-300 animate-in slide-in-from-bottom shadow-lg hover:shadow-xl"
                        style={{ animationDelay: `${idx * 50}ms` }}
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <Badge variant="destructive" className="animate-pulse shadow-lg">
                              [FALL DETECTED]
                            </Badge>
                            <span className="text-xs text-gray-400 font-mono">
                              Frame #{alarm.frameId}
                            </span>
                          </div>
                          <span className="text-sm text-gray-300 font-medium">
                            {formatTime(alarm.ts)}
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                          <div className="p-2 bg-gray-900/50 rounded border border-gray-700">
                            <span className="text-gray-400 text-xs block mb-1">Camera ID</span>
                            <span className="font-mono text-gray-200">{alarm.deviceId ? alarm.deviceId.split('-').pop() : 'unknown'}</span>
                          </div>
                          
                          <div className="p-2 bg-gray-900/50 rounded border border-gray-700">
                            <span className="text-gray-400 text-xs block mb-1">State</span>
                            <span className="font-bold text-red-400">{alarm.state}</span>
                          </div>
                          
                          <div className="p-2 bg-gray-900/50 rounded border border-gray-700">
                            <span className="text-gray-400 text-xs block mb-1">Confidence</span>
                            <span className="font-bold text-red-400">{(alarm.score * 100).toFixed(1)}%</span>
                          </div>
                          
                          <div className="p-2 bg-gray-900/50 rounded border border-gray-700">
                            <span className="text-gray-400 text-xs block mb-1">Timestamp</span>
                            <span className="text-gray-200">{new Date(alarm.ts).toLocaleTimeString()}</span>
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
      </div>
    </div>
  )
}
