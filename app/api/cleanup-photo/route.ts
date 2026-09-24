import sharp from "sharp";
import { NextRequest, NextResponse } from "next/server";
import { incrementCallCounter } from "../call-counters";
import { createAndAwaitPrediction, fetchWithRetry, friendlyReplicateErrorMessage } from "../replicate";

// bytedance/seedream-5-lite, version
// eeb2857d94c49a5bcbc9d6c6057416e1d3b1a2735a16e08e4def9bf7ee22ec71
// https://replicate.com/bytedance/seedream-5-lite
const SEEDREAM_5_LITE_VERSION = "eeb2857d94c49a5bcbc9d6c6057416e1d3b1a2735a16e08e4def9bf7ee22ec71";

const CLEANUP_PROMPT =
  "Remove power lines, wires, cables, poles, and people from this photo. " +
  "Keep the building, walls, and architecture completely unchanged. " +
  "Slightly brighten shadowed areas for even lighting.";

// Confirmed against Replicate's own billing config for this exact model
// (fetched from replicate.com/bytedance/seedream-5-lite: prices: [{metric:
// "image_output_count", metric_display: "output image", price: "$0.035",
// title: "per output image", type: "per-unit"}]) - billing is per OUTPUT
// IMAGE, not per prediction/request. Asking for two separate day/night
// outputs - whether as two predictions or one prediction with
// sequential_image_generation producing 2 images - would be billed as 2
// output images, i.e. exactly double cost, no savings. The only way to get
// both day and night information for the price of ONE output image is a
// single generated image that visually contains both, split locally with
// zero additional Replicate calls - hence the stacked composite below.
const EXTERIOR_DAY_NIGHT_PROMPT =
  "Create ONE image containing two versions of this exact same building, stacked vertically with a hard " +
  "horizontal seam at the midpoint - no gap, no border, no caption text. TOP HALF: this building in " +
  "realistic natural daylight - clear, well-lit, true-to-life daytime colors. BOTTOM HALF: the exact same " +
  "building at night, with realistic nighttime ambient lighting and believable artificial illumination " +
  "(porch lights, window glow, ambient street lighting). Both halves must show the exact same building: " +
  "same architecture, same camera angle, same framing, same perspective, same windows, doors, balconies, " +
  "roof, and wall structure. Only the lighting and time of day should differ between the two halves. Also " +
  "remove power lines, wires, cables, poles, and people from both halves. Do not redesign the building, " +
  "do not add or remove architectural features, do not change the camera angle, and do not create a " +
  "different house.";

// There is no "match input, but twice as tall" option in this model's
// aspect_ratio enum, so a fixed tall canvas is used as an approximation
// that fits two stacked halves without either being absurdly squeezed for
// a typical (portrait-ish, phone-shot) house photo - not derived per-photo.
const DAY_NIGHT_COMPOSITE_ASPECT_RATIO = "9:16";

// Model call plus polling routinely takes longer than Vercel's 10s
// serverless default.
export const maxDuration = 60;

type Seedream5LiteOutput = string[]; // one or more image URIs

type SpaceType = "interior" | "exterior";

function parseSpaceType(value: FormDataEntryValue | null): SpaceType {
  return value === "exterior" ? "exterior" : "interior";
}

async function downloadBytes(url: string): Promise<Buffer> {
  const response = await fetchWithRetry(url);
  if (!response.ok) {
    throw new Error(`Failed to download the generated photo from Replicate (HTTP ${response.status}).`);
  }
  return Buffer.from(await response.arrayBuffer());
}

export async function POST(request: NextRequest) {
  const apiToken = process.env.REPLICATE_API_TOKEN;
  if (!apiToken) {
    return NextResponse.json(
      { error: "Server misconfiguration: REPLICATE_API_TOKEN is not set." },
      { status: 500 }
    );
  }

  let imageDataUrl: string;
  let spaceType: SpaceType;
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
    imageDataUrl = `data:${file.type || "image/jpeg"};base64,${Buffer.from(bytes).toString("base64")}`;
    spaceType = parseSpaceType(formData.get("spaceType"));
  } catch {
    return NextResponse.json(
      { error: "Could not read the uploaded image from the request body." },
      { status: 400 }
    );
  }

  const isExterior = spaceType === "exterior";

  let prediction;
  try {
    incrementCallCounter("seedream");
    prediction = await createAndAwaitPrediction<Seedream5LiteOutput>(
      SEEDREAM_5_LITE_VERSION,
      isExterior
        ? {
            prompt: EXTERIOR_DAY_NIGHT_PROMPT,
            image_input: [imageDataUrl],
            aspect_ratio: DAY_NIGHT_COMPOSITE_ASPECT_RATIO,
            output_format: "jpeg",
          }
        : {
            prompt: CLEANUP_PROMPT,
            image_input: [imageDataUrl],
            aspect_ratio: "match_input_image",
            output_format: "jpeg",
          },
      apiToken,
      "AI photo cleanup"
    );
  } catch (err) {
    const message = friendlyReplicateErrorMessage(err, "AI cleanup");
    return NextResponse.json({ error: message }, { status: 502 });
  }

  const outputUrl = prediction.output?.[0];
  if (prediction.status !== "succeeded" || !outputUrl) {
    return NextResponse.json(
      { error: prediction.error || `AI photo cleanup ${prediction.status}.` },
      { status: 502 }
    );
  }

  try {
    const bytes = await downloadBytes(outputUrl);

    if (!isExterior) {
      const cleanedImageDataUrl = `data:image/jpeg;base64,${bytes.toString("base64")}`;
      return NextResponse.json({ cleanedImageDataUrl });
    }

    // Split the day/night composite locally - deterministic, based on the
    // ACTUAL returned image's dimensions (read after the fact via sharp,
    // not assumed in advance from the requested aspect ratio), and costs
    // no additional Replicate call. The day half becomes the same
    // `cleanedImageDataUrl` the existing "compare original vs cleaned, use
    // this" UI already expects, so that flow needs no changes.
    const metadata = await sharp(bytes).metadata();
    const width = metadata.width ?? 0;
    const height = metadata.height ?? 0;
    if (!width || !height) {
      throw new Error("Could not read the generated composite's dimensions.");
    }
    const halfHeight = Math.floor(height / 2);

    const [dayBytes, nightBytes] = await Promise.all([
      sharp(bytes).extract({ left: 0, top: 0, width, height: halfHeight }).jpeg().toBuffer(),
      sharp(bytes)
        .extract({ left: 0, top: halfHeight, width, height: height - halfHeight })
        .jpeg()
        .toBuffer(),
    ]);

    const cleanedImageDataUrl = `data:image/jpeg;base64,${dayBytes.toString("base64")}`;
    const nightImageDataUrl = `data:image/jpeg;base64,${nightBytes.toString("base64")}`;

    return NextResponse.json({ cleanedImageDataUrl, nightImageDataUrl });
  } catch (err) {
    const message = friendlyReplicateErrorMessage(err, "AI cleanup");
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
