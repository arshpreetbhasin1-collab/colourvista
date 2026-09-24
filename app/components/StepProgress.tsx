"use client";

import { motion } from "motion/react";

const STEPS = ["Choose photo", "Confirm", "Space & lighting", "Clean up", "Paint"];

export default function StepProgress({ current }: { current: 1 | 2 | 3 | 4 | 5 }) {
  const progress = current / STEPS.length;

  return (
    <div className="w-full max-w-md">
      <div className="h-px w-full bg-border">
        <motion.div
          className="h-px bg-primary"
          initial={false}
          animate={{ width: `${progress * 100}%` }}
          transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
      <p className="mt-3 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {String(current).padStart(2, "0")} / {String(STEPS.length).padStart(2, "0")} &middot; {STEPS[current - 1]}
      </p>
    </div>
  );
}
