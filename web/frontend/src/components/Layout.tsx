import { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "./Navbar";

interface LayoutProps {
  title?: ReactNode;
  subtitle?: string;
  icon?: LucideIcon;
  themeColor?: "blue" | "red" | "purple" | "cyan" | "green";
  actions?: ReactNode;
  children: ReactNode;
  maxWidth?: string;
  backButton?: boolean;
}

export default function Layout({
  title,
  subtitle,
  icon: Icon,
  themeColor = "blue",
  actions,
  children,
  maxWidth = "max-w-7xl",
  backButton = true,
}: LayoutProps) {
  const navigate = useNavigate();

  const getThemeColors = (color: string) => {
    switch (color) {
      case "blue":
        return {
          bgGradient: "from-cyan-500/5 via-transparent to-blue-500/5",
          headerGradient: "from-cyan-400 to-blue-500",
          iconBg: "bg-blue-500/10",
          iconBorder: "border-blue-500/30",
          iconColor: "text-blue-400",
          separator: "from-blue-500/50 to-blue-500/50",
        };
      case "red":
        return {
          bgGradient: "from-red-500/5 via-transparent to-orange-500/5",
          headerGradient: "from-red-400 via-orange-400 to-red-400",
          iconBg: "bg-red-500/10",
          iconBorder: "border-red-500/30",
          iconColor: "text-red-400",
          separator: "from-red-500/50 via-orange-500/50 to-red-500/50",
        };
      case "purple":
        return {
          bgGradient: "from-purple-500/5 via-transparent to-purple-500/5",
          headerGradient: "from-purple-400 via-purple-500 to-purple-400",
          iconBg: "bg-purple-500/10",
          iconBorder: "border-purple-500/30",
          iconColor: "text-purple-400",
          separator: "from-purple-500/50 via-purple-500 to-purple-500/50",
        };
      default:
        // Assume green or fallback
        return {
          bgGradient: "from-green-500/5 via-transparent to-emerald-500/5",
          headerGradient: "from-green-400 to-emerald-500",
          iconBg: "bg-green-500/10",
          iconBorder: "border-green-500/30",
          iconColor: "text-green-400",
          separator: "from-green-500/50 to-green-500/50",
        };
    }
  };

  const theme = getThemeColors(themeColor);

  const shouldRenderHeader = title || subtitle || Icon || actions || backButton;

  return (
    <div className="min-h-screen bg-background relative font-mono text-foreground">
      <Navbar />
      
      {/* Background grid */}
      <div className="fixed inset-0 bg-[linear-gradient(to_right,hsl(var(--border))_1px,transparent_1px),linear-gradient(to_bottom,hsl(var(--border))_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20 pointer-events-none -z-10"></div>
      <div
        className={`fixed top-0 left-0 w-full h-full bg-gradient-to-br ${theme.bgGradient} pointer-events-none -z-10`}
      ></div>

      <div className={`pt-24 ${maxWidth} mx-auto p-4 md:p-8 relative z-10`}>
        {/* Header */}
        {shouldRenderHeader && (
          <div className="mb-6 md:mb-8">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-4">
              <div className="flex items-start gap-4">
                {backButton && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => navigate("/")}
                    className="hover:bg-accent border-2 border-border text-muted-foreground hover:text-foreground hover:border-accent mt-1"
                  >
                    <ArrowLeft className="w-5 h-5" />
                  </Button>
                )}
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    {Icon && (
                      <div
                        className={`p-2 md:p-3 ${theme.iconBg} border-2 ${theme.iconBorder} rounded-lg`}
                      >
                        <Icon className={`w-6 h-6 md:w-8 md:h-8 ${theme.iconColor}`} />
                      </div>
                    )}
                    <div>
                      {title && (
                        <h1
                          className={`text-2xl md:text-3xl font-black bg-gradient-to-r ${theme.headerGradient} bg-clip-text text-transparent uppercase tracking-tight`}
                        >
                          {title}
                        </h1>
                      )}
                    </div>
                  </div>
                  {subtitle && (
                     <p className="text-muted-foreground font-medium text-sm md:text-lg">
                       {subtitle}
                     </p>
                  )}
                </div>
              </div>

              {/* Actions Area */}
              {actions && (
                <div className="flex items-center gap-3 w-full md:w-auto mt-2 md:mt-0">
                  {actions}
                </div>
              )}
            </div>
            
            {/* Separator Line */}
             <div className={`w-full h-px bg-gradient-to-r ${theme.separator}`}></div>
          </div>
        )}

        {/* Content */}
        {children}
      </div>
    </div>
  );
}
