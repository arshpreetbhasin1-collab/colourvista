"use client";

import { useState, type PointerEvent } from "react";

// A draggable before/after reveal for the final generated image. One
// implementation for both mouse drag (desktop) and touch drag (mobile) via
// the Pointer Events API, rather than separate mouse/touch handlers.
// Pure CSS clip-path for the reveal - no pixel measurement needed, so it
// stays correct across any photo aspect ratio and window resize.
export default function CompareSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "Before",
  afterLabel = "After",
}: {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
}) {
  const [position, setPosition] = useState(50); // percent from the left

  function updateFromClientX(clientX: number, target: HTMLElement) {
    const rect = target.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.min(100, Math.max(0, pct)));
  }

  return (
    <div
      className="relative w-full touch-none overflow-hidden rounded-2xl border border-border select-none"
      onPointerDown={(event: PointerEvent<HTMLDivElement>) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        updateFromClientX(event.clientX, event.currentTarget);
      }}
      onPointerMove={(event: PointerEvent<HTMLDivElement>) => {
        if (event.buttons !== 1 && event.pointerType !== "touch") return;
        updateFromClientX(event.clientX, event.currentTarget);
      }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={afterSrc} alt={afterLabel} className="block w-full" draggable={false} />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={beforeSrc}
        alt={beforeLabel}
        draggable={false}
        className="absolute inset-0 h-full w-full object-cover"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      />

      <div
        className="pointer-events-none absolute inset-y-0 z-10 w-px -translate-x-1/2 bg-white/90"
        style={{ left: `${position}%` }}
      >
        <div className="absolute top-1/2 left-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-black/10 bg-white shadow-md">
          <div className="flex gap-0.5">
            <div className="h-3 w-0.5 rounded-full bg-foreground/50" />
            <div className="h-3 w-0.5 rounded-full bg-foreground/50" />
          </div>
        </div>
      </div>

      <span className="pointer-events-none absolute top-3 left-3 rounded-full bg-black/55 px-2.5 py-1 text-[11px] font-medium text-white">
        {beforeLabel}
      </span>
      <span className="pointer-events-none absolute top-3 right-3 rounded-full bg-black/55 px-2.5 py-1 text-[11px] font-medium text-white">
        {afterLabel}
      </span>
    </div>
  );
}
