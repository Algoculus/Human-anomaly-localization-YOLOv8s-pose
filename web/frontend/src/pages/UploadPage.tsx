import { useState, useRef } from "react";
import {
  Upload as UploadIcon,
  FileVideo,
  CheckCircle,
  XCircle,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/components/ui/use-toast";
import Layout from "@/components/Layout";

interface JobStatus {
  jobId: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  outputUrl?: string;
  metricsUrl?: string;
  error?: string;
  metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
  };
  duration?: number;
  fps?: number;
  frameCount?: number;
}

export default function UploadPage() {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<number | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type
      const validTypes = [
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
      ];
      if (!validTypes.includes(file.type)) {
        toast({
          title: "Invalid File Type",
          description: "Please upload a video file (MP4, AVI, MOV, MKV)",
          variant: "destructive",
        });
        return;
      }

      // Validate file size (max 100MB)
      if (file.size > 100 * 1024 * 1024) {
        toast({
          title: "File Too Large",
          description: "Maximum file size is 100MB",
          variant: "destructive",
        });
        return;
      }

      setSelectedFile(file);
      setJobStatus(null);
    }
  };

  const uploadVideo = async () => {
    if (!selectedFile) return;

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("/api/upload/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();

      setJobStatus({
        jobId: data.jobId,
        status: "queued",
        progress: 0,
      });

      toast({
        title: "Upload Successful",
        description: "Video processing started",
      });

      // Start polling for status
      startPolling(data.jobId);
    } catch (error) {
      console.error("Upload error:", error);
      toast({
        title: "Upload Failed",
        description: "Failed to upload video",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const startPolling = (jobId: string) => {
    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/upload/${jobId}`);

        if (!response.ok) {
          throw new Error("Failed to fetch status");
        }

        const data = await response.json();
        setJobStatus(data);

        // Stop polling if completed or failed
        if (data.status === "completed" || data.status === "failed") {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }

          if (data.status === "completed") {
            toast({
              title: "Processing Complete",
              description: "Video is ready for download",
            });
          } else {
            toast({
              title: "Processing Failed",
              description: data.error || "An error occurred",
              variant: "destructive",
            });
          }
        }
      } catch (error) {
        console.error("Polling error:", error);
      }
    }, 2000); // Poll every 2 seconds
  };

  const downloadOutput = () => {
    if (jobStatus?.outputUrl) {
      window.open(jobStatus.outputUrl, "_blank");
    }
  };

  const downloadMetrics = () => {
    if (jobStatus?.metricsUrl) {
      window.open(jobStatus.metricsUrl, "_blank");
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setJobStatus(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <Layout
      themeColor="purple"
      title="Video Upload"
      subtitle="Process pre-recorded videos offline"
      icon={UploadIcon}
    >
      <div className="space-y-6">
        {/* Helper Badge */}
        <div className="flex justify-center">
          <div className="p-3 bg-muted border-2 border-border inline-block rounded-md">
            <p className="text-xs text-muted-foreground font-mono">
              📁 Results saved to: web/backend/artifacts/videos/&lt;job-id&gt;/
            </p>
          </div>
        </div>

        {/* Upload Card */}
        <Card className="border-2 border-purple-500/30 bg-card">
          <CardHeader className="border-b-2 border-purple-500/20 bg-gradient-to-b from-purple-500/10 to-transparent">
            <CardTitle className="flex items-center gap-2 text-card-foreground font-bold uppercase">
              <UploadIcon className="w-5 h-5 text-purple-400" />
              Upload Video
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            {!selectedFile ? (
              <div
                className="border-2 border-dashed border-purple-500/30 p-12 text-center cursor-pointer hover:border-purple-500 hover:bg-purple-500/5 transition-all duration-300"
                onClick={() => fileInputRef.current?.click()}
              >
                <FileVideo className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                <p className="text-lg font-bold mb-2 text-card-foreground">
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-muted-foreground">
                  MP4, AVI, MOV, MKV (max 100MB)
                </p>
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
                <div className="flex items-center justify-between p-4 bg-muted border-2 border-purple-500/30">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-500/10 border-2 border-purple-500/30">
                      <FileVideo className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                      <p className="font-bold text-foreground">
                        {selectedFile.name}
                      </p>
                      <p className="text-sm text-purple-300 font-medium">
                        {formatFileSize(selectedFile.size)}
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={resetUpload}
                    className="border-2 border-purple-500/30 hover:border-purple-500/50 hover:bg-purple-500/10"
                  >
                    Remove
                  </Button>
                </div>

                {/* Upload Button */}
                {!jobStatus && (
                  <Button
                    className="w-full border-2 border-purple-500 bg-purple-500/10 hover:bg-purple-500/20 hover:shadow-[0_0_20px_rgba(168,85,247,0.5)] text-foreground"
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
          <Card className="border-2 border-purple-500/30 bg-card">
            <CardHeader className="border-b-2 border-purple-500/20 bg-gradient-to-b from-purple-500/10 to-transparent">
              <CardTitle className="flex items-center gap-2 text-card-foreground font-bold uppercase">
                {jobStatus.status === "completed" && (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                )}
                {jobStatus.status === "failed" && (
                  <XCircle className="w-5 h-5 text-red-500" />
                )}
                {(jobStatus.status === "queued" ||
                  jobStatus.status === "processing") && (
                  <Loader2 className="w-5 h-5 animate-spin text-purple-400" />
                )}
                Processing Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-6">
              {/* Progress Bar */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-bold uppercase text-foreground">
                    {jobStatus.status}
                  </span>
                  <span className="text-sm text-muted-foreground font-bold">
                    {jobStatus.progress}%
                  </span>
                </div>
                <Progress value={jobStatus.progress} />
              </div>

              {/* Error Message */}
              {jobStatus.error && (
                <div className="p-4 bg-red-500/10 border-2 border-red-500/30">
                  <p className="text-red-400 font-medium">
                    {jobStatus.error}
                  </p>
                </div>
              )}

              {/* Completed Results */}
              {jobStatus.status === "completed" && (
                <div className="space-y-4">
                  {/* Video Info */}
                  <div className="grid grid-cols-3 gap-4 p-4 bg-muted border-2 border-purple-500/30">
                    <div>
                      <p className="text-sm text-muted-foreground uppercase font-medium">
                        Duration
                      </p>
                      <p className="font-black text-purple-400 text-xl mt-1">
                        {formatDuration(jobStatus.duration || 0)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground uppercase font-medium">
                        FPS
                      </p>
                      <p className="font-black text-purple-400 text-xl mt-1">
                        {jobStatus.fps?.toFixed(1)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground uppercase font-medium">
                        Frames
                      </p>
                      <p className="font-black text-purple-400 text-xl mt-1">
                        {jobStatus.frameCount}
                      </p>
                    </div>
                  </div>

                  {/* Metrics */}
                  {jobStatus.metrics && (
                    <div className="grid grid-cols-2 gap-4">
                      <Card className="border-2 border-purple-500/30 bg-muted">
                        <CardContent className="p-4">
                          <p className="text-sm text-muted-foreground uppercase font-medium">
                            Accuracy
                          </p>
                          <p className="text-3xl font-black text-purple-400 mt-2">
                            {(jobStatus.metrics.accuracy! * 100).toFixed(1)}%
                          </p>
                        </CardContent>
                      </Card>
                      <Card className="border-2 border-purple-500/30 bg-muted">
                        <CardContent className="p-4">
                          <p className="text-sm text-muted-foreground uppercase font-medium">
                            Precision
                          </p>
                          <p className="text-3xl font-black text-purple-400 mt-2">
                            {(jobStatus.metrics.precision! * 100).toFixed(1)}%
                          </p>
                        </CardContent>
                      </Card>
                      <Card className="border-2 border-purple-500/30 bg-muted">
                        <CardContent className="p-4">
                          <p className="text-sm text-muted-foreground uppercase font-medium">
                            Recall
                          </p>
                          <p className="text-3xl font-black text-purple-400 mt-2">
                            {(jobStatus.metrics.recall! * 100).toFixed(1)}%
                          </p>
                        </CardContent>
                      </Card>
                      <Card className="border-2 border-purple-500/30 bg-muted">
                        <CardContent className="p-4">
                          <p className="text-sm text-muted-foreground uppercase font-medium">
                            F1-Score
                          </p>
                          <p className="text-3xl font-black text-purple-400 mt-2">
                            {(jobStatus.metrics.f1! * 100).toFixed(1)}%
                          </p>
                        </CardContent>
                      </Card>
                    </div>
                  )}

                  {/* Download Buttons */}
                  <div className="flex gap-4">
                    <Button
                      className="flex-1 border-2 border-purple-500 bg-purple-500/10 hover:bg-purple-500/20 hover:shadow-[0_0_20px_rgba(168,85,247,0.5)] text-foreground"
                      onClick={downloadOutput}
                    >
                      Download Annotated Video
                    </Button>
                    <Button
                      variant="outline"
                      className="flex-1 border-2 border-purple-500/30 hover:border-purple-500/50 hover:bg-purple-500/10"
                      onClick={downloadMetrics}
                    >
                      Download Metrics JSON
                    </Button>
                  </div>

                  <Button
                    variant="ghost"
                    className="w-full border-2 border-purple-500/20 hover:border-purple-500/40 hover:bg-purple-500/5"
                    onClick={resetUpload}
                  >
                    Upload Another Video
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}
