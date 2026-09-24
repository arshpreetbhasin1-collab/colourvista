import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import ffmpegPath from "ffmpeg-static";
import { NextRequest, NextResponse } from "next/server";
import { incrementCallCounter } from "../call-counters";
import { createAndAwaitPrediction, fetchWithRetry, friendlyReplicateErrorMessage, type ReplicatePrediction } from "../replicate";
import { segmentPhoto, type SegmentationResult } from "./segment";

const execFileAsync = promisify(execFile);

// meta/sam-2, version fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83
// https://replicate.com/meta/sam-2/versions/fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83
//
// Automatic mask generation, still used only to find discrete objects
// (windows, railings, doors, plants) so they can be subtracted out of the
// paintable building surface below - a semantic "wall" prompt can still
// include the area behind/around them.
const SAM2_MODEL_VERSION =
  "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83";

// lucataco/sam3-video, version 408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269
// https://replicate.com/lucataco/sam3-video
//
// There is no official meta/sam-3 model on Replicate (confirmed against
// Replicate's model API and its ai-detect-objects collection). This is the
// only SAM3 offering there, a third-party wrapper, not a Meta-published
// model. Its schema requires a `video` input with no `image` field despite
// the model's own description claiming image support - empirically (not
// per the docs) it does accept a single still photo directly and treats it
// as a 1-frame video. Its `return_zip` option is documented as returning
// individual frame PNGs but for a single-image input actually returns a
// zipped 1-frame video, so a still frame has to be pulled out of the video
// output ourselves (see extractFirstFramePng below) rather than consumed
// directly as an image.
const SAM3_MODEL_VERSION =
  "408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269";

// SAM2 automatic generation, SAM3 wall prompting, and OpenCV line splitting
// together routinely take longer than Vercel's 10s serverless default.
export const maxDuration = 60;

type Sam2Output = { combined_mask?: string; individual_masks?: string[] };
type Sam3Output = string; // a single video URI

type AnalysisOutcome = { ok: true; result: SegmentationResult } | { ok: false; status: number; error: string };

// Keyed by the client-generated analysisId (one per uploaded photo - see
// app/paint/page.tsx). Two requests for the same analysisId - e.g. React
// Strict Mode's double effect invocation in dev, or any other accidental
// re-trigger - converge on the SAME in-flight/completed pipeline run
// instead of firing SAM2/SAM3 a second time. The promise is stored before
// it's awaited so a second, near-simultaneous caller sees it immediately
// rather than racing a fresh run. Failed outcomes are evicted once
// resolved (see below) so a genuine retry after an error still retries.
const analysisCache = new Map<string, Promise<AnalysisOutcome>>();
const MAX_CACHED_ANALYSES = 50;

function rememberAnalysis(analysisId: string, outcome: Promise<AnalysisOutcome>) {
  if (analysisCache.size >= MAX_CACHED_ANALYSES) {
    const oldestKey = analysisCache.keys().next().value;
    if (oldestKey !== undefined) analysisCache.delete(oldestKey);
  }
  analysisCache.set(analysisId, outcome);
  outcome.then((result) => {
    if (!result.ok) analysisCache.delete(analysisId);
  });
}

export async function POST(request: NextRequest) {
  const apiToken = process.env.REPLICATE_API_TOKEN;
  if (!apiToken) {
    return NextResponse.json(
      { error: "Server misconfiguration: REPLICATE_API_TOKEN is not set." },
      { status: 500 }
    );
  }

  let imageBytes: Uint8Array;
  let imageDataUrl: string;
  let analysisId: string | null;
  try {
    const formData = await request.formData();
    const file = formData.get("image");

    if (!(file instanceof File)) {
      return NextResponse.json(
        {
          error:
            "No image found. Send it as multipart/form-data with the file under the field name 'image'.",
        },
        { status: 400 }
      );
    }

    const bytes = await file.arrayBuffer();
    imageBytes = new Uint8Array(bytes);
    imageDataUrl = `data:${file.type || "image/jpeg"};base64,${Buffer.from(bytes).toString("base64")}`;

    const analysisIdField = formData.get("analysisId");
    analysisId = typeof analysisIdField === "string" && analysisIdField ? analysisIdField : null;
  } catch {
    return NextResponse.json(
      { error: "Could not read the uploaded image from the request body." },
      { status: 400 }
    );
  }

  let outcome: AnalysisOutcome;
  const cached = analysisId ? analysisCache.get(analysisId) : undefined;
  if (cached) {
    outcome = await cached;
  } else {
    const runPromise = runAnalysis(imageBytes, imageDataUrl, apiToken);
    if (analysisId) rememberAnalysis(analysisId, runPromise);
    outcome = await runPromise;
  }

  if (!outcome.ok) {
    return NextResponse.json({ error: outcome.error }, { status: outcome.status });
  }
  return NextResponse.json(outcome.result);
}

// The actual SAM2 + SAM3 + local segmentation pipeline, run exactly once
// per analysisId (see analysisCache above). Never throws - failures are
// reported through the returned AnalysisOutcome so a failed run doesn't
// leave the cache holding an unhandled rejection.
async function runAnalysis(
  imageBytes: Uint8Array,
  imageDataUrl: string,
  apiToken: string
): Promise<AnalysisOutcome> {
  let sam2Result: ReplicatePrediction<Sam2Output>;
  let sam3Result: ReplicatePrediction<Sam3Output>;
  try {
    incrementCallCounter("sam2");
    incrementCallCounter("sam3");
    // SAM2 object detection and SAM3 wall detection don't depend on each
    // other, so run them concurrently.
    [sam2Result, sam3Result] = await Promise.all([
      createAndAwaitPrediction<Sam2Output>(
        SAM2_MODEL_VERSION,
        {
          image: imageDataUrl,
          points_per_side: 32,
          pred_iou_thresh: 0.88,
          stability_score_thresh: 0.95,
        },
        apiToken,
        "SAM2 object detection"
      ),
      createAndAwaitPrediction<Sam3Output>(
        SAM3_MODEL_VERSION,
        {
          video: imageDataUrl,
          prompt: "wall",
          mask_only: true,
        },
        apiToken,
        "SAM3 wall detection"
      ),
    ]);
  } catch (err) {
    return { ok: false, status: 502, error: friendlyReplicateErrorMessage(err, "wall detection") };
  }

  if (sam2Result.status !== "succeeded") {
    return { ok: false, status: 502, error: sam2Result.error || `SAM2 object detection ${sam2Result.status}.` };
  }
  if (sam3Result.status !== "succeeded" || !sam3Result.output) {
    return { ok: false, status: 502, error: sam3Result.error || `SAM3 wall detection ${sam3Result.status}.` };
  }

  const objectMaskUrls = sam2Result.output?.individual_masks ?? [];

  try {
    const [objectMaskBuffers, wallMaskVideoBytes] = await Promise.all([
      Promise.all(
        objectMaskUrls.map(async (url) => {
          const response = await fetchWithRetry(url);
          if (!response.ok) {
            throw new Error(`Failed to download an object mask from Replicate (HTTP ${response.status}).`);
          }
          return new Uint8Array(await response.arrayBuffer());
        })
      ),
      (async () => {
        const response = await fetchWithRetry(sam3Result.output as Sam3Output);
        if (!response.ok) {
          throw new Error(`Failed to download the SAM3 wall mask from Replicate (HTTP ${response.status}).`);
        }
        return new Uint8Array(await response.arrayBuffer());
      })(),
    ]);

    const wallMaskBytes = await extractFirstFramePng(wallMaskVideoBytes);

    const result = await segmentPhoto(imageBytes, objectMaskBuffers, wallMaskBytes);
    return { ok: true, result };
  } catch (err) {
    console.error("segmentPhoto failed:", err);
    return { ok: false, status: 500, error: friendlyReplicateErrorMessage(err, "wall detection") };
  }
}

// lucataco/sam3-video always returns a video, even for a single-image
// input (a 1-frame video). Pull that one frame out as a PNG so segment.ts
// can treat it like any other mask image.
async function extractFirstFramePng(videoBytes: Uint8Array): Promise<Uint8Array> {
  if (!ffmpegPath) {
    throw new Error("ffmpeg binary not available.");
  }

  const dir = await mkdtemp(join(tmpdir(), "sam3-frame-"));
  const inputPath = join(dir, "input.mp4");
  const outputPath = join(dir, "frame.png");

  try {
    await writeFile(inputPath, videoBytes);
    await execFileAsync(ffmpegPath, ["-y", "-i", inputPath, "-frames:v", "1", "-update", "1", outputPath]);
    return new Uint8Array(await readFile(outputPath));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}
