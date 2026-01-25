import { useNavigate, useLocation } from "react-router-dom";
import { Activity, Video, Upload, Bell } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ModeToggle } from "./mode-toggle";

export default function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  const links = [
    { name: "Dashboard", path: "/", icon: Activity },
    { name: "Camera", path: "/camera", icon: Video },
    { name: "Receiver", path: "/receiver", icon: Bell },
    { name: "Upload", path: "/upload", icon: Upload },
  ];

  return (
    <nav className="sticky top-0 left-0 right-0 z-50 border-b border-border bg-background/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 md:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-gradient-to-tr from-cyan-400 to-blue-600 flex items-center justify-center">
              <Activity className="h-5 w-5 text-white" />
            </div>
            <span className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent hidden md:inline-block">
              AlgoCulus
            </span>
          </div>

          <div className="flex items-center gap-1">
            {links.map((link) => (
              <Button
                key={link.path}
                variant={isActive(link.path) ? "secondary" : "ghost"}
                size="sm"
                onClick={() => navigate(link.path)}
                className={`gap-2 transition-all ${
                  isActive(link.path)
                    ? "bg-secondary text-secondary-foreground hover:bg-secondary/80"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
              >
                <link.icon className="h-4 w-4" />
                <span className="hidden sm:inline-block">{link.name}</span>
              </Button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <ModeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
}
