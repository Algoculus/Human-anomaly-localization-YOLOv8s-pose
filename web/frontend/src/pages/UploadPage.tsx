import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Upload as UploadIcon, FileVideo, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { useToast } from '@/components/ui/use-toast'

interface JobStatus {
  jobId: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  progress: number
  outputUrl?: string
  metricsUrl?: string
  error?: string
  metrics?: {
    accuracy?: number
    precision?: number
    recall?: number
    f1?: number
  }
  duration?: number
  fps?: number
  frameCount?: number
}

export default function UploadPage() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const pollIntervalRef = useRef<number | null>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      // Validate file type
      const validTypes = ['video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska']
      if (!validTypes.includes(file.type)) {
        toast({
          title: 'Invalid File Type',
          description: 'Please upload a video file (MP4, AVI, MOV, MKV)',
          variant: 'destructive'
        })
        return
      }

      // Validate file size (max 100MB)
      if (file.size > 100 * 1024 * 1024) {
        toast({
          title: 'File Too Large',
          description: 'Maximum file size is 100MB',
          variant: 'destructive'
        })
        return
      }

      setSelectedFile(file)
      setJobStatus(null)
    }
  }

  const uploadVideo = async () => {
    if (!selectedFile) return

    setIsUploading(true)
    
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('/api/upload/', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const data = await response.json()
      
      setJobStatus({
        jobId: data.jobId,
        status: 'queued',
        progress: 0
      })

      toast({
        title: 'Upload Successful',
        description: 'Video processing started',
      })

      // Start polling for status
      startPolling(data.jobId)

    } catch (error) {
      console.error('Upload error:', error)
      toast({
        title: 'Upload Failed',
        description: 'Failed to upload video',
        variant: 'destructive'
      })
    } finally {
      setIsUploading(false)
    }
  }

  const startPolling = (jobId: string) => {
    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/upload/${jobId}`)
        
        if (!response.ok) {
          throw new Error('Failed to fetch status')
        }

        const data = await response.json()
        setJobStatus(data)

        // Stop polling if completed or failed
        if (data.status === 'completed' || data.status === 'failed') {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
            pollIntervalRef.current = null
          }

          if (data.status === 'completed') {
            toast({
              title: 'Processing Complete',
              description: 'Video is ready for download',
            })
          } else {
            toast({
              title: 'Processing Failed',
              description: data.error || 'An error occurred',
              variant: 'destructive'
            })
          }
        }
      } catch (error) {
        console.error('Polling error:', error)
      }
    }, 2000) // Poll every 2 seconds
  }

  const downloadOutput = () => {
    if (jobStatus?.outputUrl) {
      window.open(jobStatus.outputUrl, '_blank')
    }
  }

  const downloadMetrics = () => {
    if (jobStatus?.metricsUrl) {
      window.open(jobStatus.metricsUrl, '_blank')
    }
  }

  const resetUpload = () => {
    setSelectedFile(null)
    setJobStatus(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <Button variant="ghost" onClick={() => navigate('/')}>
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h1 className="text-3xl font-bold">Video Upload</h1>
            <p className="text-gray-600 dark:text-gray-400">Process pre-recorded videos offline</p>
            <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">📁 Results saved to: web/backend/artifacts/videos/&lt;job-id&gt;/</p>
          </div>
        </div>

        <div className="space-y-6">
          {/* Upload Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UploadIcon className="w-5 h-5" />
                Upload Video
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!selectedFile ? (
                <div 
                  className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-12 text-center cursor-pointer hover:border-primary transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <FileVideo className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                  <p className="text-lg font-medium mb-2">Click to upload or drag and drop</p>
                  <p className="text-sm text-gray-500">MP4, AVI, MOV, MKV (max 100MB)</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept="video/*"
                    onChange={handleFileSelect}
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  {/* File Info */}
                  <div className="flex items-center justify-between p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                    <div className="flex items-center gap-3">
                      <FileVideo className="w-8 h-8 text-primary" />
                      <div>
                        <p className="font-medium">{selectedFile.name}</p>
                        <p className="text-sm text-gray-500">{formatFileSize(selectedFile.size)}</p>
                      </div>
                    </div>
                    <Button variant="ghost" size="sm" onClick={resetUpload}>
                      Remove
                    </Button>
                  </div>

                  {/* Upload Button */}
                  {!jobStatus && (
                    <Button 
                      className="w-full" 
                      size="lg"
                      onClick={uploadVideo}
                      disabled={isUploading}
                    >
                      {isUploading ? (
                        <>
                          <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                          Uploading...
                        </>
                      ) : (
                        <>
                          <UploadIcon className="w-5 h-5 mr-2" />
                          Start Processing
                        </>
                      )}
                    </Button>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Processing Status */}
          {jobStatus && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {jobStatus.status === 'completed' && <CheckCircle className="w-5 h-5 text-green-500" />}
                  {jobStatus.status === 'failed' && <XCircle className="w-5 h-5 text-red-500" />}
                  {(jobStatus.status === 'queued' || jobStatus.status === 'processing') && (
                    <Loader2 className="w-5 h-5 animate-spin text-primary" />
                  )}
                  Processing Status
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Progress Bar */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium capitalize">{jobStatus.status}</span>
                    <span className="text-sm text-gray-500">{jobStatus.progress}%</span>
                  </div>
                  <Progress value={jobStatus.progress} />
                </div>

                {/* Error Message */}
                {jobStatus.error && (
                  <div className="p-4 bg-red-100 dark:bg-red-900 rounded-lg">
                    <p className="text-red-700 dark:text-red-300">{jobStatus.error}</p>
                  </div>
                )}

                {/* Completed Results */}
                {jobStatus.status === 'completed' && (
                  <div className="space-y-4">
                    {/* Video Info */}
                    <div className="grid grid-cols-3 gap-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                      <div>
                        <p className="text-sm text-gray-500">Duration</p>
                        <p className="font-medium">{formatDuration(jobStatus.duration || 0)}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">FPS</p>
                        <p className="font-medium">{jobStatus.fps?.toFixed(1)}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Frames</p>
                        <p className="font-medium">{jobStatus.frameCount}</p>
                      </div>
                    </div>

                    {/* Metrics */}
                    {jobStatus.metrics && (
                      <div className="grid grid-cols-2 gap-4">
                        <Card>
                          <CardContent className="p-4">
                            <p className="text-sm text-gray-500">Accuracy</p>
                            <p className="text-2xl font-bold">{(jobStatus.metrics.accuracy! * 100).toFixed(1)}%</p>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4">
                            <p className="text-sm text-gray-500">Precision</p>
                            <p className="text-2xl font-bold">{(jobStatus.metrics.precision! * 100).toFixed(1)}%</p>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4">
                            <p className="text-sm text-gray-500">Recall</p>
                            <p className="text-2xl font-bold">{(jobStatus.metrics.recall! * 100).toFixed(1)}%</p>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4">
                            <p className="text-sm text-gray-500">F1-Score</p>
                            <p className="text-2xl font-bold">{(jobStatus.metrics.f1! * 100).toFixed(1)}%</p>
                          </CardContent>
                        </Card>
                      </div>
                    )}

                    {/* Download Buttons */}
                    <div className="flex gap-4">
                      <Button className="flex-1" onClick={downloadOutput}>
                        Download Annotated Video
                      </Button>
                      <Button variant="outline" className="flex-1" onClick={downloadMetrics}>
                        Download Metrics JSON
                      </Button>
                    </div>

                    <Button variant="ghost" className="w-full" onClick={resetUpload}>
                      Upload Another Video
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
