"use client";

import { motion } from "motion/react";

// An elegant "real work is happening" state for the two genuinely slow
// backend steps (AI cleanup, SAM2/SAM3 wall detection) - a title plus the
// actual real steps involved, not a fabricated percentage. The scanning
// bar is a continuous, indeterminate motion cue (it never claims to
// represent actual progress), and each step line stagger-fades in once,
// on mount, rather than looping - restrained, not busy.
export default function AnalyzingIndicator({ title, steps }: { title: string; steps: string[] }) {
  return (
    <div className="flex flex-col items-center gap-5 py-1 text-center">
      <div className="relative h-px w-44 overflow-hidden rounded-full bg-border">
        <motion.span
          className="absolute inset-y-0 w-1/3 rounded-full bg-primary"
          animate={{ x: ["-100%", "300%"] }}
          transition={{ duration: 1.7, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>
      <div>
        <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-muted-foreground">{title}</p>
        <ul className="mt-2.5 flex flex-col gap-1">
          {steps.map((step, index) => (
            <motion.li
              key={step}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 * index, duration: 0.4, ease: "easeOut" }}
              className="text-sm text-foreground/70"
            >
              {step}
            </motion.li>
          ))}
        </ul>
      </div>
    </div>
  );
}
