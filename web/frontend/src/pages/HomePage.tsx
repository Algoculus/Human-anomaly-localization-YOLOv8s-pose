import { Link } from 'react-router-dom'
import { Camera, Users, Upload, Activity } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="flex justify-center mb-6">
            <Activity className="w-16 h-16 text-primary" />
          </div>
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
            Fall Detection System
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Real-time fall detection powered by YOLOv8s-pose with WebSocket streaming
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {/* Camera Mode */}
          <Card className="hover:shadow-xl transition-shadow">
            <CardHeader>
              <div className="flex justify-center mb-4">
                <Camera className="w-12 h-12 text-blue-500" />
              </div>
              <CardTitle className="text-center">Camera Simulator</CardTitle>
              <CardDescription className="text-center">
                Stream frames from webcam or video file for real-time detection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link to="/camera">
                <Button className="w-full" size="lg">
                  Launch Camera
                </Button>
              </Link>
              <ul className="mt-4 space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li>✓ Real-time pose estimation</li>
                <li>✓ Live telemetry display</li>
                <li>✓ Instant alarm broadcast</li>
              </ul>
            </CardContent>
          </Card>

          {/* Receiver Mode */}
          <Card className="hover:shadow-xl transition-shadow">
            <CardHeader>
              <div className="flex justify-center mb-4">
                <Users className="w-12 h-12 text-green-500" />
              </div>
              <CardTitle className="text-center">Alert Receiver</CardTitle>
              <CardDescription className="text-center">
                Monitor fall alerts from multiple cameras in real-time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link to="/receiver">
                <Button className="w-full" variant="secondary" size="lg">
                  Open Receiver
                </Button>
              </Link>
              <ul className="mt-4 space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li>✓ Multi-camera monitoring</li>
                <li>✓ Snapshot gallery</li>
                <li>✓ Audio/visual notifications</li>
              </ul>
            </CardContent>
          </Card>

          {/* Upload Mode */}
          <Card className="hover:shadow-xl transition-shadow">
            <CardHeader>
              <div className="flex justify-center mb-4">
                <Upload className="w-12 h-12 text-purple-500" />
              </div>
              <CardTitle className="text-center">Video Upload</CardTitle>
              <CardDescription className="text-center">
                Process pre-recorded videos and generate overlay + metrics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link to="/upload">
                <Button className="w-full" variant="outline" size="lg">
                  Upload Video
                </Button>
              </Link>
              <ul className="mt-4 space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li>✓ Offline processing</li>
                <li>✓ Annotated video output</li>
                <li>✓ Performance metrics</li>
              </ul>
            </CardContent>
          </Card>
        </div>

        {/* System Info */}
        <div className="mt-16 text-center">
          <div className="inline-block bg-white dark:bg-gray-800 rounded-lg shadow px-8 py-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-semibold">Core AI:</span> YOLOv8s-pose + State Machine
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              <span className="font-semibold">Backend:</span> FastAPI + WebSocket
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              <span className="font-semibold">Frontend:</span> React + TypeScript + shadcn/ui
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
