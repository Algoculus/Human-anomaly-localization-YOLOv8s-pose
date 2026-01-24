import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center border-2 px-3 py-1 text-xs font-bold uppercase tracking-wider transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-primary bg-primary text-primary-foreground hover:bg-primary/80 hover:shadow-[0_0_10px_rgba(59,130,246,0.5)]",
        secondary:
          "border-secondary bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-destructive bg-destructive text-destructive-foreground hover:bg-destructive/80 hover:shadow-[0_0_10px_rgba(239,68,68,0.5)]",
        outline: "border-border text-foreground bg-transparent",
        success:
          "border-green-500 bg-green-500 text-white hover:bg-green-600 hover:shadow-[0_0_10px_rgba(34,197,94,0.5)]",
        warning:
          "border-yellow-500 bg-yellow-500 text-white hover:bg-yellow-600 hover:shadow-[0_0_10px_rgba(234,179,8,0.5)]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
