import cvReady from "@techstark/opencv-js";
import type { Mat } from "@techstark/opencv-js";
import sharp from "sharp";

type CV = Awaited<typeof cvReady>;
type Rect = InstanceType<CV["Rect"]>;

let cvPromise: Promise<CV> | null = null;
function getCv(): Promise<CV> {
  if (!cvPromise) {
    cvPromise = cvReady as unknown as Promise<CV>;
  }
  return cvPromise;
}

// SAM2 automatic mode returns masks for everything it finds - both small
// discrete objects (windows, railings, plants) and large surfaces
// (walls, floor, ceiling). It has no semantic labels, so "object" vs.
// "surface" is a size cutoff: anything covering more than this fraction of
// the photo is assumed to be a surface and is left as part of the
// candidate wall area rather than subtracted out of it.
const OBJECT_AREA_FRACTION = 0.2;

// Any candidate region (whole area or a post-split piece) smaller than
// this fraction of the photo is dropped as noise.
//
// Measured against a real leftover mask (post-exclusion, post-closing): its
// 20 raw connected components split cleanly into two clusters with a ~45x
// gap between them - 7 components from 0.33% to 6.2% of the frame (real
// surfaces: two main walls, the rooftop terrace split into two pieces by a
// cable shadow, a window-pillar fragment, and two edge-of-frame slivers of
// neighboring buildings), then a hard drop to <=0.007% (specks of a few
// pixels or less - segmentation noise, nothing in between). The previous
// 0.02 (2%) cutoff sat inside the real cluster and dropped the rooftop
// terrace entirely. 0.003 sits in the gap: it keeps everything in the real
// cluster and nothing from the noise cluster.
const MIN_REGION_AREA_FRACTION = 0.003;

// OpenCV work happens on a downscaled copy so cost stays bounded regardless
// of the uploaded photo's resolution; polygon coordinates are rescaled back
// to the original photo's pixel space before returning.
const ANALYSIS_MAX_DIMENSION = 1024;

const CANNY_THRESHOLD_1 = 50;
const CANNY_THRESHOLD_2 = 150;
const HOUGH_VOTE_THRESHOLD = 80;
const HOUGH_MAX_LINE_GAP = 20;
const LINE_MERGE_ANGLE_TOLERANCE_DEG = 5;
const LINE_MERGE_DISTANCE_TOLERANCE_PX = 8;
const MIN_LINE_LENGTH_FRACTION = 0.15; // of the image diagonal
const MIN_CROSSING_FRACTION = 0.4; // of the area's shorter bbox side
const POLY_SIMPLIFY_EPSILON_FRACTION = 0.005; // of contour perimeter

// A leftover area this large is almost certainly several real surfaces
// (sky/wall/ground, or two separate wall faces) that failed to get split by
// the normal pass - it must not be returned as one unsplit region.
const OVERSIZED_AREA_FRACTION = 0.5;
// The forced re-scan is scoped to a crop of just that area, so thresholds
// tuned for the whole photo's diagonal would almost never fire; both are
// relaxed relative to the crop's own diagonal instead.
const SCOPED_HOUGH_VOTE_THRESHOLD = 30;
const SCOPED_MIN_LINE_LENGTH_FRACTION = 0.2; // of the crop's diagonal

// Morphological closing on SAM3's raw wall mask bridges small segmentation
// gaps/holes (noise) without bridging real ones (a balcony railing void, a
// window, an architectural break). Fixed in absolute pixels rather than
// scaled to image size: the analysis mat itself is already normalized to
// <= ANALYSIS_MAX_DIMENSION regardless of the source photo's resolution, so
// a fixed pixel count here is already resolution-independent.
//
// Measured against a real SAM3 wall mask (not a synthetic test): the
// narrowest real seam between two genuinely different wall planes (a
// building facade and the free-standing compound/boundary wall directly
// below it, physically touching but separated by a thin mortar-line gap in
// the mask) was 3px in analysis space. A 5px (radius 2) kernel bridges gaps
// up to ~4px and fused those two planes into one region - confirmed by
// diffing connected components before/after closing on that real mask. A
// 3px (radius 1) kernel only bridges gaps up to ~2px, which cleans up small
// same-surface mask fragmentation without crossing that 3px seam.
const WALL_MASK_CLOSE_KERNEL_PX = 3;

// Sky: bright, low-saturation, low local texture, connected to the top edge.
const SKY_BRIGHTNESS_MIN = 140; // HSV V channel, 0-255
const SKY_SATURATION_MAX = 70; // HSV S channel, 0-255
const SKY_TEXTURE_MAX = 12; // local Sobel-magnitude score

// Ground: distinctly different in color from the facade, textured,
// connected to the bottom edge. The color threshold is adaptive (see
// buildSkyAndGroundMasks) - this is only the floor for a perfectly flat,
// zero-variance facade.
const GROUND_COLOR_DIFF_FLOOR = 25; // Lab distance from the facade reference color
// How far above the facade's own internal color variance a pixel's Lab
// distance must be to count as "distinctly different", not just texture.
const FACADE_VARIATION_MULTIPLIER = 1.5;
const GROUND_TEXTURE_MIN = 20; // local Sobel-magnitude score
// The facade reference color is sampled from this horizontal band, assuming
// a roughly conventional eye-level house photo (sky top / facade middle /
// ground bottom). This is a simple heuristic, not framing-invariant.
const FACADE_BAND_TOP_FRACTION = 0.35;
const FACADE_BAND_BOTTOM_FRACTION = 0.65;

type Point = [number, number];

type RegionKind = "whole" | "merged" | "split";

export interface Region {
  id: string;
  areaId: string;
  kind: RegionKind;
  area: number;
  points: Point[];
  // Objects (windows, railings, doors) fully enclosed by this region's wall
  // area - e.g. a window with wall visible all around it - don't just notch
  // the outer boundary, they leave a literal hole in the mask. A single
  // outer ring can't represent that, so holes are carried separately and
  // rendered as cutouts (fill-rule="evenodd" in the UI).
  holes: Point[][];
  // A point guaranteed to sit inside the region's own mask (and away from
  // any hole) - see findInteriorPoint. A naive bbox or vertex-average
  // centroid can land outside a concave region or inside a window cutout;
  // this is the UI's dot-marker anchor.
  dot: Point;
}

export interface SegmentationDebugStats {
  paintablePixels: number;
  excludedPixels: number;
  sam3CandidateCount: number;
  recoveredCandidateCount: number;
  finalWallCount: number;
  unassignedPaintablePercent: number;
}

export interface SegmentationResult {
  imageWidth: number;
  imageHeight: number;
  regions: Region[];
  debug: SegmentationDebugStats;
}

interface Segment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

type AreaSource = "sam3" | "recovered";

interface Area {
  label: number;
  source: AreaSource;
  mask: Mat;
  area: number;
}

// Splits `candidateMask` into its connected components and keeps the ones
// clearing MIN_REGION_AREA_FRACTION, each as its own standalone mask -
// shared by both the SAM3-labeled pass and the unassigned-paintable
// recovery pass below, so a candidate area is built identically either way.
function extractAreaCandidates(
  cv: CV,
  candidateMask: Mat,
  width: number,
  height: number,
  totalArea: number,
  source: AreaSource,
  track: <T extends { delete(): void }>(mat: T) => T
): Area[] {
  const labels = track(new cv.Mat());
  const stats = track(new cv.Mat());
  const centroids = track(new cv.Mat());
  const numLabels = cv.connectedComponentsWithStats(candidateMask, labels, stats, centroids, 8, cv.CV_32S);

  const areas: Area[] = [];
  const labelData = labels.data32S;
  for (let label = 1; label < numLabels; label++) {
    const areaPixels = stats.data32S[label * 5 + 4];
    if (areaPixels / totalArea < MIN_REGION_AREA_FRACTION) continue;

    const maskData = new Uint8Array(width * height);
    for (let i = 0; i < maskData.length; i++) {
      if (labelData[i] === label) maskData[i] = 255;
    }
    const mask = new cv.Mat(height, width, cv.CV_8UC1);
    mask.data.set(maskData);
    areas.push({ label, source, mask: track(mask), area: areaPixels });
  }
  return areas;
}

export async function segmentPhoto(
  photoBytes: Uint8Array,
  objectMaskBuffers: Uint8Array[],
  wallMaskBuffer: Uint8Array
): Promise<SegmentationResult> {
  const cv = await getCv();

  const meta = await sharp(Buffer.from(photoBytes)).metadata();
  const originalWidth = meta.width ?? 0;
  const originalHeight = meta.height ?? 0;
  if (!originalWidth || !originalHeight) {
    throw new Error("Could not read the uploaded photo's dimensions.");
  }

  const scale = Math.min(1, ANALYSIS_MAX_DIMENSION / Math.max(originalWidth, originalHeight));
  const width = Math.max(1, Math.round(originalWidth * scale));
  const height = Math.max(1, Math.round(originalHeight * scale));
  const totalArea = width * height;
  const scaleX = originalWidth / width;
  const scaleY = originalHeight / height;

  const { data: photoRgba } = await sharp(Buffer.from(photoBytes))
    .resize(width, height, { fit: "fill" })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const photoMat = cv.matFromImageData({
    data: new Uint8ClampedArray(photoRgba.buffer, photoRgba.byteOffset, photoRgba.length),
    width,
    height,
  });

  const mats: { delete(): void }[] = [photoMat];
  const track = <T extends { delete(): void }>(mat: T): T => {
    mats.push(mat);
    return mat;
  };

  try {
    const unionMask = track(new cv.Mat(height, width, cv.CV_8UC1));
    unionMask.setTo(new cv.Scalar(0, 0, 0, 0));

    for (const maskBytes of objectMaskBuffers) {
      const { data: maskGray } = await sharp(Buffer.from(maskBytes))
        .resize(width, height, { fit: "fill" })
        .greyscale()
        .raw()
        .toBuffer({ resolveWithObject: true });

      const maskMat = new cv.Mat(height, width, cv.CV_8UC1);
      maskMat.data.set(maskGray);
      const binaryMask = new cv.Mat();
      cv.threshold(maskMat, binaryMask, 127, 255, cv.THRESH_BINARY);
      maskMat.delete();

      const maskArea = cv.countNonZero(binaryMask);
      if (maskArea / totalArea <= OBJECT_AREA_FRACTION) {
        cv.bitwise_or(unionMask, binaryMask, unionMask);
      }
      binaryMask.delete();
    }

    const { skyMask, groundMask } = buildSkyAndGroundMasks(cv, photoMat, width, height, track);

    const excludedMask = track(new cv.Mat());
    cv.bitwise_or(unionMask, skyMask, excludedMask);
    cv.bitwise_or(excludedMask, groundMask, excludedMask);

    // The paintable building surface is defined independently of SAM3:
    // everything that isn't a discrete SAM2 object, sky, or ground. SAM3's
    // semantic "wall" mask is used below only to PARTITION this surface
    // into individual wall planes - it must never be the reason real
    // paintable surface disappears just because SAM3 didn't label it wall.
    const paintableMask = track(new cv.Mat());
    cv.bitwise_not(excludedMask, paintableMask);
    const paintablePixels = cv.countNonZero(paintableMask);
    const excludedPixels = totalArea - paintablePixels;

    const { data: wallGray } = await sharp(Buffer.from(wallMaskBuffer))
      .resize(width, height, { fit: "fill" })
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const wallGrayMat = track(new cv.Mat(height, width, cv.CV_8UC1));
    wallGrayMat.data.set(wallGray);
    const wallMaskRaw = track(new cv.Mat());
    cv.threshold(wallGrayMat, wallMaskRaw, 127, 255, cv.THRESH_BINARY);

    const closeKernel = track(
      cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        new cv.Size(WALL_MASK_CLOSE_KERNEL_PX, WALL_MASK_CLOSE_KERNEL_PX)
      )
    );
    const wallMask = track(new cv.Mat());
    cv.morphologyEx(wallMaskRaw, wallMask, cv.MORPH_CLOSE, closeKernel);

    // finalCandidate = sam3Mask ∩ paintableMask
    const sam3CandidateMask = track(new cv.Mat());
    cv.bitwise_and(wallMask, paintableMask, sam3CandidateMask);
    const sam3Areas = extractAreaCandidates(cv, sam3CandidateMask, width, height, totalArea, "sam3", track);

    // wallUnion = union(all valid wall candidates); unassigned = paintable
    // surface SAM3 never claimed as wall. Meaningful (non-noise) connected
    // components of that leftover become additional wall candidates in
    // their own right, rather than being silently dropped.
    const wallUnion = track(new cv.Mat(height, width, cv.CV_8UC1));
    wallUnion.setTo(new cv.Scalar(0, 0, 0, 0));
    for (const sam3Area of sam3Areas) {
      cv.bitwise_or(wallUnion, sam3Area.mask, wallUnion);
    }
    const notWallUnion = track(new cv.Mat());
    cv.bitwise_not(wallUnion, notWallUnion);
    const unassignedMask = track(new cv.Mat());
    cv.bitwise_and(paintableMask, notWallUnion, unassignedMask);
    const recoveredAreas = extractAreaCandidates(cv, unassignedMask, width, height, totalArea, "recovered", track);

    const areas: Area[] = [...sam3Areas, ...recoveredAreas];

    // Sanity metric: how much of the paintable surface still isn't covered
    // by any area candidate (sam3 + recovered) once both passes' own noise
    // filter (MIN_REGION_AREA_FRACTION) has been applied. Should stay small
    // - a large value means real surface is slipping through both passes.
    const coveredMask = track(new cv.Mat(height, width, cv.CV_8UC1));
    coveredMask.setTo(new cv.Scalar(0, 0, 0, 0));
    for (const candidateArea of areas) {
      cv.bitwise_or(coveredMask, candidateArea.mask, coveredMask);
    }
    const notCovered = track(new cv.Mat());
    cv.bitwise_not(coveredMask, notCovered);
    const uncoveredPaintable = track(new cv.Mat());
    cv.bitwise_and(paintableMask, notCovered, uncoveredPaintable);
    const uncoveredPaintablePixels = cv.countNonZero(uncoveredPaintable);
    const unassignedPaintablePercent =
      paintablePixels > 0 ? (uncoveredPaintablePixels / paintablePixels) * 100 : 0;

    const lines = detectLines(cv, photoMat, width, height, track);

    const regions: Region[] = [];
    let nextId = 0;
    for (const area of areas) {
      let split = tryBestSplit(cv, area, lines, width, height, totalArea);

      if (!split && area.area / totalArea > OVERSIZED_AREA_FRACTION) {
        split = forceSplitOversizedArea(cv, area, photoMat, lines, width, height, totalArea, track);
      }

      const areaId = `${area.source}-${area.label}`;

      if (split) {
        const mergedPoly = maskToPolygon(cv, area.mask, scaleX, scaleY, totalArea, track);
        const polyA = maskToPolygon(cv, split.pieceAMask, scaleX, scaleY, totalArea, track);
        const polyB = maskToPolygon(cv, split.pieceBMask, scaleX, scaleY, totalArea, track);
        const mergedDot = findInteriorPoint(cv, area.mask, scaleX, scaleY);
        const dotA = findInteriorPoint(cv, split.pieceAMask, scaleX, scaleY);
        const dotB = findInteriorPoint(cv, split.pieceBMask, scaleX, scaleY);
        split.pieceAMask.delete();
        split.pieceBMask.delete();

        if (mergedPoly) {
          regions.push({
            id: `r${nextId++}`,
            areaId,
            kind: "merged",
            area: area.area,
            points: mergedPoly.outer,
            holes: mergedPoly.holes,
            dot: mergedDot,
          });
        }
        if (polyA) {
          regions.push({
            id: `r${nextId++}`,
            areaId,
            kind: "split",
            area: split.pieceAArea,
            points: polyA.outer,
            holes: polyA.holes,
            dot: dotA,
          });
        }
        if (polyB) {
          regions.push({
            id: `r${nextId++}`,
            areaId,
            kind: "split",
            area: split.pieceBArea,
            points: polyB.outer,
            holes: polyB.holes,
            dot: dotB,
          });
        }
      } else {
        if (area.area / totalArea > OVERSIZED_AREA_FRACTION) {
          console.warn(
            `detect-walls: area ${areaId} covers ${((area.area / totalArea) * 100).toFixed(1)}% of the image and could not be split even after a forced bisection attempt.`
          );
        }
        const poly = maskToPolygon(cv, area.mask, scaleX, scaleY, totalArea, track);
        if (poly) {
          regions.push({
            id: `r${nextId++}`,
            areaId,
            kind: "whole",
            area: area.area,
            points: poly.outer,
            holes: poly.holes,
            dot: findInteriorPoint(cv, area.mask, scaleX, scaleY),
          });
        }
      }
    }

    const finalWallCount = regions.filter((region) => region.kind !== "merged").length;

    return {
      imageWidth: originalWidth,
      imageHeight: originalHeight,
      regions,
      debug: {
        paintablePixels,
        excludedPixels,
        sam3CandidateCount: sam3Areas.length,
        recoveredCandidateCount: recoveredAreas.length,
        finalWallCount,
        unassignedPaintablePercent,
      },
    };
  } finally {
    for (const mat of mats) mat.delete();
  }
}

// Builds masks for pixels that are very likely sky (bright, desaturated,
// smooth, touching the top edge) and ground (color-distinct from the
// facade, textured, touching the bottom edge), so both can be excluded from
// the wall-candidate area before it's computed.
function buildSkyAndGroundMasks(
  cv: CV,
  photoMat: Mat,
  width: number,
  height: number,
  track: <T extends { delete(): void }>(mat: T) => T
): { skyMask: Mat; groundMask: Mat } {
  const rgb = track(new cv.Mat());
  cv.cvtColor(photoMat, rgb, cv.COLOR_RGBA2RGB);

  const gray = track(new cv.Mat());
  cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);

  // Local texture score: Sobel gradient magnitude, box-blurred so it
  // reflects a neighborhood rather than single-pixel noise.
  const sobelX = track(new cv.Mat());
  const sobelY = track(new cv.Mat());
  cv.Sobel(gray, sobelX, cv.CV_32F, 1, 0, 3);
  cv.Sobel(gray, sobelY, cv.CV_32F, 0, 1, 3);
  const gradMag = track(new cv.Mat());
  cv.magnitude(sobelX, sobelY, gradMag);
  const textureRaw = track(new cv.Mat());
  cv.blur(gradMag, textureRaw, new cv.Size(9, 9));
  const texture = track(new cv.Mat());
  cv.convertScaleAbs(textureRaw, texture);

  // --- Sky ---
  const hsv = track(new cv.Mat());
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
  const hsvChannels = track(new cv.MatVector());
  cv.split(hsv, hsvChannels);
  const satChannel = track(hsvChannels.get(1));
  const valChannel = track(hsvChannels.get(2));

  const brightMask = track(new cv.Mat());
  cv.threshold(valChannel, brightMask, SKY_BRIGHTNESS_MIN, 255, cv.THRESH_BINARY);
  const lowSatMask = track(new cv.Mat());
  cv.threshold(satChannel, lowSatMask, SKY_SATURATION_MAX, 255, cv.THRESH_BINARY_INV);
  const lowTextureMask = track(new cv.Mat());
  cv.threshold(texture, lowTextureMask, SKY_TEXTURE_MAX, 255, cv.THRESH_BINARY_INV);

  const skyCandidate = track(new cv.Mat());
  cv.bitwise_and(brightMask, lowSatMask, skyCandidate);
  cv.bitwise_and(skyCandidate, lowTextureMask, skyCandidate);

  const skyMask = track(keepComponentsTouchingEdge(cv, skyCandidate, width, height, "top"));

  // --- Ground ---
  const lab = track(new cv.Mat());
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);

  const bandTop = Math.round(height * FACADE_BAND_TOP_FRACTION);
  const bandBottom = Math.round(height * FACADE_BAND_BOTTOM_FRACTION);
  const bandRect = new cv.Rect(0, bandTop, width, Math.max(1, bandBottom - bandTop));
  const band = track(lab.roi(bandRect));
  const facade = cv.mean(band);

  const labData = lab.data;
  const colorDiff = new Uint8Array(width * height);
  // A textured facade (brick, siding, stucco) has real color variation of
  // its own, so a fixed distance-from-mean threshold can end up satisfied
  // by the facade itself, letting the ground mask bleed straight into it.
  // Track how much the reference band itself deviates from its own mean so
  // the ground threshold can be calibrated well above that baseline.
  let facadeBandDiffSum = 0;
  let facadeBandDiffCount = 0;
  for (let y = 0, i = 0, p = 0; y < height; y++) {
    const inBand = y >= bandTop && y < bandBottom;
    for (let x = 0; x < width; x++, i++, p += 3) {
      const dl = labData[p] - facade[0];
      const da = labData[p + 1] - facade[1];
      const db = labData[p + 2] - facade[2];
      const dist = Math.sqrt(dl * dl + da * da + db * db);
      colorDiff[i] = dist > 255 ? 255 : dist;
      if (inBand) {
        facadeBandDiffSum += dist;
        facadeBandDiffCount++;
      }
    }
  }
  const facadeBandMeanDiff = facadeBandDiffCount > 0 ? facadeBandDiffSum / facadeBandDiffCount : 0;
  const groundColorThreshold = Math.max(
    GROUND_COLOR_DIFF_FLOOR,
    facadeBandMeanDiff * FACADE_VARIATION_MULTIPLIER
  );

  const colorDiffMat = track(new cv.Mat(height, width, cv.CV_8UC1));
  colorDiffMat.data.set(colorDiff);

  const colorDiffMask = track(new cv.Mat());
  cv.threshold(colorDiffMat, colorDiffMask, groundColorThreshold, 255, cv.THRESH_BINARY);
  const highTextureMask = track(new cv.Mat());
  cv.threshold(texture, highTextureMask, GROUND_TEXTURE_MIN, 255, cv.THRESH_BINARY);

  const groundCandidate = track(new cv.Mat());
  cv.bitwise_and(colorDiffMask, highTextureMask, groundCandidate);

  const groundMask = track(keepComponentsTouchingEdge(cv, groundCandidate, width, height, "bottom"));

  return { skyMask, groundMask };
}

// Keeps only the connected components of `candidateMask` whose bounding box
// touches the given edge of the image - the step that turns "looks like
// sky/ground" into "actually is contiguous with the top/bottom of frame".
function keepComponentsTouchingEdge(
  cv: CV,
  candidateMask: Mat,
  width: number,
  height: number,
  edge: "top" | "bottom"
): Mat {
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const count = cv.connectedComponentsWithStats(candidateMask, labels, stats, centroids, 8, cv.CV_32S);

  const keepLabels = new Set<number>();
  for (let label = 1; label < count; label++) {
    const y = stats.data32S[label * 5 + 1];
    const h = stats.data32S[label * 5 + 3];
    if (edge === "top" && y === 0) keepLabels.add(label);
    if (edge === "bottom" && y + h === height) keepLabels.add(label);
  }

  const result = new cv.Mat(height, width, cv.CV_8UC1);
  const resultData = new Uint8Array(width * height);
  const labelData = labels.data32S;
  for (let i = 0; i < resultData.length; i++) {
    if (keepLabels.has(labelData[i])) resultData[i] = 255;
  }
  result.data.set(resultData);

  labels.delete();
  stats.delete();
  centroids.delete();

  return result;
}

function detectLines(
  cv: CV,
  photoMat: Mat,
  width: number,
  height: number,
  track: <T extends { delete(): void }>(mat: T) => T
): Segment[] {
  const gray = track(new cv.Mat());
  cv.cvtColor(photoMat, gray, cv.COLOR_RGBA2GRAY);

  const blurred = track(new cv.Mat());
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

  const edges = track(new cv.Mat());
  cv.Canny(blurred, edges, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);

  const diagonal = Math.hypot(width, height);
  const linesMat = track(new cv.Mat());
  cv.HoughLinesP(
    edges,
    linesMat,
    1,
    Math.PI / 180,
    HOUGH_VOTE_THRESHOLD,
    diagonal * 0.1,
    HOUGH_MAX_LINE_GAP
  );

  const rawSegments: Segment[] = [];
  for (let i = 0; i < linesMat.rows; i++) {
    rawSegments.push({
      x1: linesMat.data32S[i * 4],
      y1: linesMat.data32S[i * 4 + 1],
      x2: linesMat.data32S[i * 4 + 2],
      y2: linesMat.data32S[i * 4 + 3],
    });
  }

  const minMergedLength = diagonal * MIN_LINE_LENGTH_FRACTION;
  return mergeCollinearSegments(rawSegments).filter((line) => segmentLength(line) >= minMergedLength);
}

// Same as detectLines, but scoped to a crop of the photo and with
// thresholds relaxed relative to the CROP's own diagonal instead of the
// full photo's - used only for the forced re-scan of oversized areas, where
// the whole-photo thresholds would almost never be satisfiable.
function detectLinesInRect(
  cv: CV,
  photoMat: Mat,
  rect: Rect,
  track: <T extends { delete(): void }>(mat: T) => T
): Segment[] {
  const crop = track(photoMat.roi(rect));
  const gray = track(new cv.Mat());
  cv.cvtColor(crop, gray, cv.COLOR_RGBA2GRAY);

  const blurred = track(new cv.Mat());
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

  const edges = track(new cv.Mat());
  cv.Canny(blurred, edges, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);

  const cropDiagonal = Math.hypot(rect.width, rect.height);
  const linesMat = track(new cv.Mat());
  cv.HoughLinesP(
    edges,
    linesMat,
    1,
    Math.PI / 180,
    SCOPED_HOUGH_VOTE_THRESHOLD,
    cropDiagonal * SCOPED_MIN_LINE_LENGTH_FRACTION,
    HOUGH_MAX_LINE_GAP
  );

  const rawSegments: Segment[] = [];
  for (let i = 0; i < linesMat.rows; i++) {
    rawSegments.push({
      x1: linesMat.data32S[i * 4] + rect.x,
      y1: linesMat.data32S[i * 4 + 1] + rect.y,
      x2: linesMat.data32S[i * 4 + 2] + rect.x,
      y2: linesMat.data32S[i * 4 + 3] + rect.y,
    });
  }

  const minMergedLength = cropDiagonal * SCOPED_MIN_LINE_LENGTH_FRACTION;
  return mergeCollinearSegments(rawSegments).filter((line) => segmentLength(line) >= minMergedLength);
}

function segmentLength(s: Segment): number {
  return Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
}

function segmentAngleDeg(s: Segment): number {
  return (Math.atan2(s.y2 - s.y1, s.x2 - s.x1) * 180) / Math.PI;
}

function angleDiffDeg(a: number, b: number): number {
  const d = Math.abs(a - b) % 180;
  return d > 90 ? 180 - d : d;
}

// Perpendicular distance from a point to the infinite line through segment s.
function perpDistance(s: Segment, px: number, py: number): number {
  const dx = s.x2 - s.x1;
  const dy = s.y2 - s.y1;
  const len = Math.hypot(dx, dy) || 1;
  return Math.abs((px - s.x1) * dy - (py - s.y1) * dx) / len;
}

// Raw Hough output is fragmented (broken up by furniture, shadows,
// texture). Group segments that lie on nearly the same infinite line and
// collapse each group into one segment spanning its extreme endpoints.
function mergeCollinearSegments(segments: Segment[]): Segment[] {
  const used = new Array(segments.length).fill(false);
  const merged: Segment[] = [];

  const order = segments
    .map((_, i) => i)
    .sort((a, b) => segmentLength(segments[b]) - segmentLength(segments[a]));

  for (const seedIdx of order) {
    if (used[seedIdx]) continue;
    const seed = segments[seedIdx];
    used[seedIdx] = true;
    const group: Segment[] = [seed];
    const seedAngle = segmentAngleDeg(seed);

    for (let i = 0; i < segments.length; i++) {
      if (used[i]) continue;
      const candidate = segments[i];
      if (angleDiffDeg(seedAngle, segmentAngleDeg(candidate)) > LINE_MERGE_ANGLE_TOLERANCE_DEG) continue;

      const dist = Math.max(
        perpDistance(seed, candidate.x1, candidate.y1),
        perpDistance(seed, candidate.x2, candidate.y2)
      );
      if (dist > LINE_MERGE_DISTANCE_TOLERANCE_PX) continue;

      group.push(candidate);
      used[i] = true;
    }

    merged.push(collapseGroup(group));
  }

  return merged;
}

function collapseGroup(group: Segment[]): Segment {
  const base = group.reduce((a, b) => (segmentLength(a) >= segmentLength(b) ? a : b));
  const dx = base.x2 - base.x1;
  const dy = base.y2 - base.y1;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;

  let minT = Infinity;
  let maxT = -Infinity;
  let minPoint = { x: base.x1, y: base.y1 };
  let maxPoint = { x: base.x2, y: base.y2 };

  for (const seg of group) {
    for (const [px, py] of [
      [seg.x1, seg.y1],
      [seg.x2, seg.y2],
    ] as const) {
      const t = (px - base.x1) * ux + (py - base.y1) * uy;
      if (t < minT) {
        minT = t;
        minPoint = { x: px, y: py };
      }
      if (t > maxT) {
        maxT = t;
        maxPoint = { x: px, y: py };
      }
    }
  }

  return { x1: minPoint.x, y1: minPoint.y, x2: maxPoint.x, y2: maxPoint.y };
}

interface SplitResult {
  pieceAMask: Mat;
  pieceBMask: Mat;
  pieceAArea: number;
  pieceBArea: number;
}

// Finds the single line, among `lines`, that most crosses `area`'s interior
// (not just clipping a corner) - the crossing has to cover a minimum
// fraction of the area's shorter bbox side to qualify.
function findBestCrossingLine(
  cv: CV,
  area: Area,
  lines: Segment[],
  width: number,
  height: number
): { line: Segment; crossing: number } | null {
  const boundsRect = cv.boundingRect(area.mask);
  const minCrossing = MIN_CROSSING_FRACTION * Math.min(boundsRect.width, boundsRect.height);

  let best: { line: Segment; crossing: number } | null = null;

  for (const line of lines) {
    const lineMask = new cv.Mat(height, width, cv.CV_8UC1);
    lineMask.setTo(new cv.Scalar(0, 0, 0, 0));
    cv.line(lineMask, new cv.Point(line.x1, line.y1), new cv.Point(line.x2, line.y2), new cv.Scalar(255, 0, 0, 0), 1);

    const intersection = new cv.Mat();
    cv.bitwise_and(lineMask, area.mask, intersection);
    const crossing = cv.countNonZero(intersection);
    lineMask.delete();
    intersection.delete();

    if (crossing >= minCrossing && (!best || crossing > best.crossing)) {
      best = { line, crossing };
    }
  }

  return best;
}

// Cuts `area`'s mask along `line` (dilated to 3px) and, if that yields
// exactly two pieces both clearing the minimum size, returns them.
function cutAreaWithLine(
  cv: CV,
  area: Area,
  line: Segment,
  width: number,
  height: number,
  totalArea: number
): SplitResult | null {
  const cutMask = area.mask.clone();
  cv.line(cutMask, new cv.Point(line.x1, line.y1), new cv.Point(line.x2, line.y2), new cv.Scalar(0, 0, 0, 0), 3);

  const cutLabels = new cv.Mat();
  const cutStats = new cv.Mat();
  const cutCentroids = new cv.Mat();
  const cutCount = cv.connectedComponentsWithStats(cutMask, cutLabels, cutStats, cutCentroids, 8, cv.CV_32S);
  cutMask.delete();

  const pieces: { label: number; area: number }[] = [];
  for (let label = 1; label < cutCount; label++) {
    const pieceArea = cutStats.data32S[label * 5 + 4];
    if (pieceArea / totalArea >= MIN_REGION_AREA_FRACTION) {
      pieces.push({ label, area: pieceArea });
    }
  }
  pieces.sort((a, b) => b.area - a.area);

  if (pieces.length < 2) {
    cutLabels.delete();
    cutStats.delete();
    cutCentroids.delete();
    return null;
  }

  const [pieceA, pieceB] = pieces;
  const cutLabelData = cutLabels.data32S;
  const dataA = new Uint8Array(width * height);
  const dataB = new Uint8Array(width * height);
  for (let i = 0; i < dataA.length; i++) {
    if (cutLabelData[i] === pieceA.label) dataA[i] = 255;
    else if (cutLabelData[i] === pieceB.label) dataB[i] = 255;
  }

  const pieceAMask = new cv.Mat(height, width, cv.CV_8UC1);
  pieceAMask.data.set(dataA);
  const pieceBMask = new cv.Mat(height, width, cv.CV_8UC1);
  pieceBMask.data.set(dataB);

  cutLabels.delete();
  cutStats.delete();
  cutCentroids.delete();

  return { pieceAMask, pieceBMask, pieceAArea: pieceA.area, pieceBArea: pieceB.area };
}

function tryBestSplit(
  cv: CV,
  area: Area,
  lines: Segment[],
  width: number,
  height: number,
  totalArea: number
): SplitResult | null {
  const best = findBestCrossingLine(cv, area, lines, width, height);
  if (!best) return null;
  return cutAreaWithLine(cv, area, best.line, width, height, totalArea);
}

// For an area that's still unsplit and covers more than
// OVERSIZED_AREA_FRACTION of the image: re-run line detection scoped to
// just that area's (padded) bounding box with relaxed thresholds, and if
// that still finds nothing splittable, force a straight bisection through
// the centroid along the longer bbox axis. The bisection is a last resort,
// not a detected boundary - it exists so a >50% blob never survives as one
// unsplit region.
function forceSplitOversizedArea(
  cv: CV,
  area: Area,
  photoMat: Mat,
  globalLines: Segment[],
  width: number,
  height: number,
  totalArea: number,
  track: <T extends { delete(): void }>(mat: T) => T
): SplitResult | null {
  const bounds = cv.boundingRect(area.mask);
  const margin = Math.round(Math.max(bounds.width, bounds.height) * 0.1);
  const x = Math.max(0, bounds.x - margin);
  const y = Math.max(0, bounds.y - margin);
  const w = Math.min(width - x, bounds.width + margin * 2);
  const h = Math.min(height - y, bounds.height + margin * 2);
  const cropRect = new cv.Rect(x, y, w, h);

  const scopedLines = detectLinesInRect(cv, photoMat, cropRect, track);
  const best = findBestCrossingLine(cv, area, [...scopedLines, ...globalLines], width, height);
  if (best) {
    const cut = cutAreaWithLine(cv, area, best.line, width, height, totalArea);
    if (cut) return cut;
  }

  const bisection: Segment =
    bounds.width >= bounds.height
      ? {
          x1: bounds.x + bounds.width / 2,
          y1: bounds.y,
          x2: bounds.x + bounds.width / 2,
          y2: bounds.y + bounds.height,
        }
      : {
          x1: bounds.x,
          y1: bounds.y + bounds.height / 2,
          x2: bounds.x + bounds.width,
          y2: bounds.y + bounds.height / 2,
        };

  return cutAreaWithLine(cv, area, bisection, width, height, totalArea);
}

// A point guaranteed to sit inside `mask` (and away from any hole cut into
// it), for placing a wall-dot marker. A bbox or vertex-average centroid can
// land outside a concave region or inside a window/door hole; distance
// transform gives, for every foreground pixel, its distance to the nearest
// background pixel (which includes hole boundaries, since holes are
// literal 0s in these masks) - the pixel that maximizes that distance is
// the deepest interior point, the same idea as a polygon's "pole of
// inaccessibility".
function findInteriorPoint(cv: CV, mask: Mat, scaleX: number, scaleY: number): Point {
  const dist = new cv.Mat();
  cv.distanceTransform(mask, dist, cv.DIST_L2, 5);
  // opencv-js's generated types model minMaxLoc's C++ pointer-output
  // params, which don't apply to the actual embind call - the real,
  // documented JS signature is `cv.minMaxLoc(src)` returning
  // {minVal, maxVal, minLoc, maxLoc} directly.
  const minMaxLoc = cv.minMaxLoc as unknown as (src: Mat) => { maxLoc: { x: number; y: number } };
  const { maxLoc } = minMaxLoc(dist);
  dist.delete();
  return [Math.round(maxLoc.x * scaleX), Math.round(maxLoc.y * scaleY)];
}

function contourToPoints(
  cv: CV,
  contour: Mat,
  scaleX: number,
  scaleY: number,
  track: <T extends { delete(): void }>(mat: T) => T
): Point[] | null {
  const perimeter = cv.arcLength(contour, true);
  const approx = track(new cv.Mat());
  cv.approxPolyDP(contour, approx, Math.max(1, POLY_SIMPLIFY_EPSILON_FRACTION * perimeter), true);

  const points: Point[] = [];
  for (let i = 0; i < approx.rows; i++) {
    const x = approx.data32S[i * 2];
    const y = approx.data32S[i * 2 + 1];
    points.push([Math.round(x * scaleX), Math.round(y * scaleY)]);
  }

  return points.length >= 3 ? points : null;
}

// RETR_CCOMP (rather than RETR_EXTERNAL) keeps hole contours - an object
// (window, railing, door) that's fully enclosed by this mask, not just
// notching its edge, leaves a literal hole that a single outer ring can't
// represent. Without this, an excluded object sitting entirely inside a
// wall's mask gets silently re-absorbed into the outer polygon and painted
// over, even though the pixel-level mask correctly excluded it.
function maskToPolygon(
  cv: CV,
  mask: Mat,
  scaleX: number,
  scaleY: number,
  totalArea: number,
  track: <T extends { delete(): void }>(mat: T) => T
): { outer: Point[]; holes: Point[][] } | null {
  const contours = track(new cv.MatVector());
  const hierarchy = track(new cv.Mat());
  cv.findContours(mask, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);

  let largestIdx = -1;
  let largestArea = 0;
  for (let i = 0; i < contours.size(); i++) {
    const contour = track(contours.get(i));
    const contourArea = cv.contourArea(contour);
    if (contourArea > largestArea) {
      largestArea = contourArea;
      largestIdx = i;
    }
  }

  if (largestIdx === -1) return null;

  const outer = contourToPoints(cv, track(contours.get(largestIdx)), scaleX, scaleY, track);
  if (!outer) return null;

  const holes: Point[][] = [];
  const hierarchyData = hierarchy.data32S;
  for (let i = 0; i < contours.size(); i++) {
    const parent = hierarchyData[i * 4 + 3];
    if (parent !== largestIdx) continue;

    const holeContour = track(contours.get(i));
    if (cv.contourArea(holeContour) / totalArea < MIN_REGION_AREA_FRACTION) continue;

    const holePoints = contourToPoints(cv, holeContour, scaleX, scaleY, track);
    if (holePoints) holes.push(holePoints);
  }

  return { outer, holes };
}
