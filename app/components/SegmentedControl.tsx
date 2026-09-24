"use client";

import { motion } from "motion/react";

type Option<T extends string> = { value: T; label: string };

// A small, animated 2-4 way switch (Day/Night, Quick View/Real View,
// Before/After) - one shared component so every such control in the app
// looks and moves identically. The active pill slides between options via
// a shared layoutId rather than a hard cut, which is most of what makes a
// segmented control feel "expensive" rather than just a row of buttons.
export default function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
  layoutId,
  className = "",
}: {
  value: T;
  onChange: (value: T) => void;
  options: Option<T>[];
  layoutId: string;
  className?: string;
}) {
  return (
    <div
      role="tablist"
      className={`inline-flex rounded-full border border-border bg-surface p-1 ${className}`}
    >
      {options.map((option) => {
        const isActive = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(option.value)}
            className="relative flex-1 rounded-full px-4 py-1.5 text-sm font-medium transition-colors sm:flex-none"
          >
            {isActive && (
              <motion.span
                layoutId={layoutId}
                className="absolute inset-0 rounded-full bg-primary"
                transition={{ type: "spring", stiffness: 500, damping: 40 }}
              />
            )}
            <span className={`relative z-10 ${isActive ? "text-primary-foreground" : "text-muted-foreground"}`}>
              {option.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}
