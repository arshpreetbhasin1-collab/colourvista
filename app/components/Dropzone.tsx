"use client";

import { Camera, ImageUp } from "lucide-react";
import { motion } from "motion/react";
import { useRef, useState, type ChangeEvent, type DragEvent, type MouseEvent } from "react";

// The upload entry point: a large, image-first area rather than a plain
// file-input button. Desktop: drag-and-drop, or click anywhere to browse.
// Mobile: two explicit buttons (native camera vs. photo library) rather
// than relying on a single input's OS-dependent default chooser behavior -
// tapping the panel itself also opens the library picker as a fallback.
export default function Dropzone({ onFileSelected }: { onFileSelected: (file: File) => void }) {
  const [isDragging, setIsDragging] = useState(false);
  const libraryInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (file) onFileSelected(file);
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      onDragOver={(event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsDragging(false);
        handleFiles(event.dataTransfer.files);
      }}
      onClick={() => libraryInputRef.current?.click()}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          libraryInputRef.current?.click();
        }
      }}
      role="button"
      tabIndex={0}
      className={`flex aspect-[4/3] w-full cursor-pointer flex-col items-center justify-center gap-5 rounded-2xl border transition-colors sm:aspect-[21/9] ${
        isDragging
          ? "border-primary bg-primary/[0.04]"
          : "border-border hover:border-primary/35 hover:bg-secondary/50"
      }`}
    >
      <input
        ref={libraryInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event: ChangeEvent<HTMLInputElement>) => handleFiles(event.target.files)}
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(event: ChangeEvent<HTMLInputElement>) => handleFiles(event.target.files)}
      />

      <ImageUp className="h-7 w-7 text-muted-foreground" strokeWidth={1.4} aria-hidden />
      <div className="px-6 text-center">
        <p className="font-heading text-xl text-foreground">Add a photo of your house</p>
        <p className="mt-1.5 hidden text-sm text-muted-foreground sm:block">
          Drop a photo here, or click to browse &middot; a front-facing daylight photo works best
        </p>
        <p className="mt-1.5 text-sm text-muted-foreground sm:hidden">A front-facing daylight photo works best</p>
      </div>

      <div className="flex gap-3 sm:hidden">
        <button
          type="button"
          onClick={(event: MouseEvent) => {
            event.stopPropagation();
            cameraInputRef.current?.click();
          }}
          className="inline-flex items-center gap-2 rounded-full bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground"
        >
          <Camera className="h-4 w-4" strokeWidth={1.8} aria-hidden />
          Take Photo
        </button>
        <button
          type="button"
          onClick={(event: MouseEvent) => {
            event.stopPropagation();
            libraryInputRef.current?.click();
          }}
          className="inline-flex items-center gap-2 rounded-full border border-border px-5 py-2.5 text-sm font-medium"
        >
          Choose Photo
        </button>
      </div>
    </motion.div>
  );
}
