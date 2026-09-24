import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost";

const VARIANT_CLASSES: Record<Variant, string> = {
  // Solid accent - the one color reserved for the primary action on a page.
  primary:
    "bg-accent text-accent-foreground hover:bg-accent-hover disabled:hover:bg-accent",
  // Outline - secondary/optional actions (e.g. "Clean Up with AI").
  secondary:
    "bg-surface text-primary border border-border hover:bg-background disabled:hover:bg-surface",
  // No border/fill - low-emphasis actions (e.g. "keep as-is").
  ghost: "bg-transparent text-primary hover:bg-black/5",
};

export default function Button({
  variant = "primary",
  className = "",
  disabled,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      type="button"
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium shadow-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none ${VARIANT_CLASSES[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
