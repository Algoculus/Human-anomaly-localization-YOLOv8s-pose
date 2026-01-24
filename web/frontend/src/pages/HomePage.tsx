import { Link } from "react-router-dom";
import { Camera, Users, Upload, ArrowRight, Cpu, Server } from "lucide-react";
import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Layout from "@/components/Layout";

export default function HomePage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        stiffness: 100,
        damping: 10,
      },
    },
  };

  return (
    <Layout themeColor="cyan" backButton={false}>
      <motion.div
        className="max-w-6xl mx-auto space-y-16"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Hero Section */}
        <motion.div variants={itemVariants} className="text-center space-y-6 pt-8">
          <div className="relative inline-block">
            <div className="absolute inset-0 blur-3xl bg-gradient-to-r from-cyan-400/20 via-blue-500/20 to-purple-500/20 rounded-full" />
            <h1 className="relative text-5xl md:text-7xl font-black bg-gradient-to-br from-white via-gray-200 to-gray-500 bg-clip-text text-transparent tracking-tighter shadow-sm">
              ALGO<span className="text-cyan-500">CULUS</span>
            </h1>
          </div>
          
          <div className="space-y-4">
            <h2 className="text-2xl md:text-3xl font-bold text-foreground/80 tracking-tight">
              Advanced <span className="text-cyan-500">Fall Detection</span> System
            </h2>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto leading-relaxed">
              Real-time anomaly localization powered by state-of-the-art vision models.
              Secure monitoring for safety and peace of mind.
            </p>
          </div>

          <div className="flex justify-center gap-2">
            <Badge variant="outline" className="px-4 py-1.5 border-cyan-500/30 bg-cyan-500/10 text-cyan-400">
              <Cpu className="w-3 h-3 mr-2" /> YOLOv8s-pose
            </Badge>
            <Badge variant="outline" className="px-4 py-1.5 border-blue-500/30 bg-blue-500/10 text-blue-400">
              <Server className="w-3 h-3 mr-2" /> WebSocket
            </Badge>
          </div>
        </motion.div>

        {/* Feature Cards */}
        <motion.div variants={containerVariants} className="grid md:grid-cols-3 gap-8">
          {/* Camera Mode */}
          <motion.div variants={itemVariants} whileHover={{ y: -5 }} className="h-full">
            <Card className="h-full flex flex-col overflow-hidden border-2 border-border/50 bg-card/50 backdrop-blur-sm hover:border-cyan-500/50 hover:shadow-[0_0_40px_-10px_rgba(6,182,212,0.3)] transition-all duration-300">
              <CardHeader className="text-center pb-2">
                <div className="mx-auto w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 flex items-center justify-center mb-4 border border-cyan-500/20 group-hover:border-cyan-500/50 transition-colors">
                  <Camera className="w-8 h-8 text-cyan-500" />
                </div>
                <CardTitle className="text-xl font-bold">Camera Feed</CardTitle>
                <CardDescription>Real-time webcam analysis</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col justify-between pt-4">
                <p className="text-sm text-muted-foreground text-center mb-6">
                  Stream live video from your device's camera with instant pose estimation and fall detection.
                </p>
                <Link to="/camera">
                  <Button className="w-full bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-500 hover:text-cyan-400 border border-cyan-500/20">
                    Launch Camera <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </motion.div>

          {/* Receiver Mode */}
          <motion.div variants={itemVariants} whileHover={{ y: -5 }} className="h-full">
            <Card className="h-full flex flex-col overflow-hidden border-2 border-border/50 bg-card/50 backdrop-blur-sm hover:border-red-500/50 hover:shadow-[0_0_40px_-10px_rgba(239,68,68,0.3)] transition-all duration-300">
              <CardHeader className="text-center pb-2">
                <div className="mx-auto w-16 h-16 rounded-2xl bg-gradient-to-br from-red-500/20 to-orange-600/20 flex items-center justify-center mb-4 border border-red-500/20 group-hover:border-red-500/50 transition-colors">
                  <Users className="w-8 h-8 text-red-500" />
                </div>
                <CardTitle className="text-xl font-bold">Alert Receiver</CardTitle>
                <CardDescription>Centralized monitoring dashboard</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col justify-between pt-4">
                <p className="text-sm text-muted-foreground text-center mb-6">
                  Monitor multiple camera feeds simultaneously and receive instant audio-visual alerts.
                </p>
                <Link to="/receiver">
                  <Button variant="outline" className="w-full border-red-500/20 text-red-500 hover:text-red-400 hover:bg-red-500/10 hover:border-red-500/30">
                    Open Receiver <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </motion.div>

          {/* Upload Mode */}
          <motion.div variants={itemVariants} whileHover={{ y: -5 }} className="h-full">
            <Card className="h-full flex flex-col overflow-hidden border-2 border-border/50 bg-card/50 backdrop-blur-sm hover:border-purple-500/50 hover:shadow-[0_0_40px_-10px_rgba(168,85,247,0.3)] transition-all duration-300">
              <CardHeader className="text-center pb-2">
                <div className="mx-auto w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500/20 to-pink-600/20 flex items-center justify-center mb-4 border border-purple-500/20 group-hover:border-purple-500/50 transition-colors">
                  <Upload className="w-8 h-8 text-purple-500" />
                </div>
                <CardTitle className="text-xl font-bold">Video Upload</CardTitle>
                <CardDescription>Offline batch processing</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col justify-between pt-4">
                <p className="text-sm text-muted-foreground text-center mb-6">
                  Upload pre-recorded video files for detailed analysis and performance metrics generation.
                </p>
                <Link to="/upload">
                  <Button variant="outline" className="w-full border-purple-500/20 text-purple-500 hover:text-purple-400 hover:bg-purple-500/10 hover:border-purple-500/30">
                    Upload File <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>

        {/* System Architecture */}
        <motion.div variants={itemVariants} className="pt-8 border-t border-border/50">
           <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div className="p-4 rounded-xl bg-card/30 border border-border/30 backdrop-blur-sm">
                <span className="block text-xs uppercase tracking-wider text-muted-foreground font-bold mb-1">Architecture</span>
                <span className="font-semibold text-foreground">Client-Server</span>
              </div>
              <div className="p-4 rounded-xl bg-card/30 border border-border/30 backdrop-blur-sm">
                 <span className="block text-xs uppercase tracking-wider text-muted-foreground font-bold mb-1">Latency</span>
                 <span className="font-semibold text-green-500">&lt; 100ms</span>
              </div>
              <div className="p-4 rounded-xl bg-card/30 border border-border/30 backdrop-blur-sm">
                 <span className="block text-xs uppercase tracking-wider text-muted-foreground font-bold mb-1">Frontend</span>
                 <span className="font-semibold text-blue-500">React + Vite</span>
              </div>
              <div className="p-4 rounded-xl bg-card/30 border border-border/30 backdrop-blur-sm">
                 <span className="block text-xs uppercase tracking-wider text-muted-foreground font-bold mb-1">Backend</span>
                 <span className="font-semibold text-purple-500">FastAPI</span>
              </div>
           </div>
        </motion.div>
      </motion.div>
    </Layout>
  );
}
