/**
 * Browser-side media preprocessing for google/gemma-4-E4B.
 *
 * This module deliberately has no model, tokenizer, WebGPU, or UI dependency.
 * It turns URL-backed chat content into the tensors consumed by Gemma 4's
 * vision and audio encoders, leaving prompt construction and GPU upload to the
 * caller.  All browser-only APIs are accessed lazily so importing this file
 * from a server-rendered module is safe; calling a decoder on the server is
 * reported as a clear error instead.
 */

export const GEMMA_IMAGE_PATCH_SIZE = 16;
export const GEMMA_IMAGE_POOLING_KERNEL_SIZE = 3;
export const GEMMA_IMAGE_PATCH_PIXELS =
  GEMMA_IMAGE_PATCH_SIZE * GEMMA_IMAGE_PATCH_SIZE * 3;
export const GEMMA_AUDIO_SAMPLE_RATE = 16_000;
export const GEMMA_AUDIO_FEATURE_SIZE = 128;
export const GEMMA_AUDIO_FRAME_LENGTH = 320;
export const GEMMA_AUDIO_HOP_LENGTH = 160;
export const GEMMA_AUDIO_FFT_LENGTH = 512;

const SIDE_MULTIPLE = GEMMA_IMAGE_PATCH_SIZE * GEMMA_IMAGE_POOLING_KERNEL_SIZE;
const SUPPORTED_SOFT_TOKEN_COUNTS = [70, 140, 280, 560, 1120] as const;
const MEDIA_EVENT_TIMEOUT_MS = 30_000;
const DATA_URL_PROTOCOL = "data:";

export type GemmaSoftTokenCount = (typeof SUPPORTED_SOFT_TOKEN_COUNTS)[number];

/**
 * Defensive limits for untrusted URL content. They are intentionally separate
 * from model limits so an application can make them stricter without changing
 * the Gemma processor layout.
 */
export interface GemmaMediaLimits {
  maxContentParts: number;
  maxImages: number;
  maxVideos: number;
  maxAudios: number;
  maxImageBytes: number;
  maxVideoBytes: number;
  maxAudioBytes: number;
  maxTotalBytes: number;
  maxDecodedImagePixels: number;
  maxDecodedVideoPixels: number;
  maxDecodedAudioSamples: number;
  maxVideoDurationSeconds: number;
  maxAudioDurationSeconds: number;
  maxAudioChannels: number;
  maxVideoFrames: number;
  maxTensorBytes: number;
}

export const DEFAULT_GEMMA_MEDIA_LIMITS: Readonly<GemmaMediaLimits> = Object.freeze({
  maxContentParts: 32,
  maxImages: 8,
  maxVideos: 2,
  maxAudios: 4,
  maxImageBytes: 32 * 1024 * 1024,
  maxVideoBytes: 128 * 1024 * 1024,
  maxAudioBytes: 64 * 1024 * 1024,
  maxTotalBytes: 192 * 1024 * 1024,
  maxDecodedImagePixels: 20_000_000,
  maxDecodedVideoPixels: 8_000_000,
  // Per channel, before 16 kHz resampling; prevents pathological high-rate files.
  maxDecodedAudioSamples: 8_000_000,
  maxVideoDurationSeconds: 120,
  // Gemma's processor defaults to 480,000 samples, i.e. 30 seconds at 16 kHz.
  maxAudioDurationSeconds: 30,
  maxAudioChannels: 8,
  maxVideoFrames: 32,
  maxTensorBytes: 256 * 1024 * 1024,
});

export class GemmaMediaError extends Error {
  readonly code: string;

  constructor(message: string, code = "gemma_media_error") {
    super(message);
    this.name = "GemmaMediaError";
    this.code = code;
  }
}

export interface GemmaTextContentPart {
  type: "text";
  text: string;
}

export interface GemmaImageContentPart {
  type: "image";
  url: string;
  /** OpenAI-compatible hint: `low` uses 70 Gemma soft tokens, `high` keeps 280. */
  detail?: "low" | "high";
}

export interface GemmaVideoContentPart {
  type: "video";
  url: string;
  /** Optional caller cap for sampled frames (1–32 by default). */
  frames?: number;
}

export interface GemmaAudioContentPart {
  type: "audio";
  url: string;
}

/** Normalized, URL-only content format used by this module. */
export type GemmaContentPart =
  | GemmaTextContentPart
  | GemmaImageContentPart
  | GemmaVideoContentPart
  | GemmaAudioContentPart;

export interface GemmaVisionInput {
  /** Canonical source URL; callers should not assume it remains fetchable later. */
  sourceUrl: string;
  /** [maxPatches, 16 * 16 * 3], RGB values in [0, 1], padded with zeroes. */
  pixelValues: Float32Array;
  /** [maxPatches, 2], x/y patch coordinates; padded rows are [-1, -1]. */
  positionIds: Int32Array;
  /** Number of non-padding 16x16 patches. */
  patchCount: number;
  /** patchCount / poolingKernelSize². */
  numSoftTokens: number;
  /** Resized dimensions, each divisible by 48. */
  width: number;
  height: number;
  maxPatches: number;
  patchSize: number;
}

export interface GemmaVideoInput {
  sourceUrl: string;
  /** [frameCount, maxPatches, 16 * 16 * 3], RGB values in [0, 1]. */
  pixelValues: Float32Array;
  /** [frameCount, maxPatches, 2], with [-1, -1] padding rows. */
  positionIds: Int32Array;
  /** Seconds into the decoded video for every extracted frame. */
  timestamps: Float64Array;
  frameCount: number;
  patchCountPerFrame: number;
  numSoftTokensPerFrame: number;
  width: number;
  height: number;
  maxPatches: number;
  patchSize: number;
}

export interface GemmaAudioInput {
  sourceUrl: string;
  /** [frameCount, 128] log-mel features compatible with Gemma4AudioFeatureExtractor. */
  inputFeatures: Float32Array;
  /** [frameCount], 1 for fully real-audio frames and 0 for padding. */
  inputFeaturesMask: Uint8Array;
  frameCount: number;
  featureSize: number;
  sampleRate: number;
  sampleCount: number;
  durationSeconds: number;
}

export type GemmaPreparedContentPart =
  | GemmaTextContentPart
  | { type: "image"; imageIndex: number; sourceUrl: string }
  | { type: "video"; videoIndex: number; sourceUrl: string }
  | { type: "audio"; audioIndex: number; sourceUrl: string };

export interface GemmaMediaInputs {
  /** Input ordering, suitable for inserting the matching Gemma modality token. */
  parts: GemmaPreparedContentPart[];
  images: GemmaVisionInput[];
  videos: GemmaVideoInput[];
  audios: GemmaAudioInput[];
}

export interface GemmaMediaPreprocessOptions {
  signal?: AbortSignal;
  limits?: Partial<GemmaMediaLimits>;
  /** Gemma image processor default: 280 soft tokens per still image. */
  imageSoftTokens?: GemmaSoftTokenCount;
  /** Gemma video processor default: 70 soft tokens per sampled frame. */
  videoSoftTokens?: GemmaSoftTokenCount;
  /** Gemma video processor default: 32 uniformly distributed frames. */
  videoFrames?: number;
}

interface ResolvedOptions {
  signal?: AbortSignal;
  limits: GemmaMediaLimits;
  imageSoftTokens: GemmaSoftTokenCount;
  videoSoftTokens: GemmaSoftTokenCount;
  videoFrames: number;
}

interface ByteBudget {
  used: number;
  limit: number;
}

interface TensorBudget {
  used: number;
  limit: number;
}

interface VisionLayout {
  width: number;
  height: number;
  patchWidth: number;
  patchHeight: number;
  patchCount: number;
  maxPatches: number;
  numSoftTokens: number;
}

interface DecodedImage {
  source: CanvasImageSource;
  width: number;
  height: number;
  close: () => void;
}

interface CanvasSurface {
  context: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
}

interface FftPlan {
  bitReversed: Uint16Array;
  cos: Float64Array[];
  sin: Float64Array[];
}

let cachedFftPlan: FftPlan | undefined;
let cachedMelFilters: Float64Array | undefined;
let cachedHannWindow: Float64Array | undefined;

/**
 * Validates and normalizes an OpenAI-style content array.
 *
 * Accepted media spellings are `image`/`image_url`, `video`/`video_url`, and
 * `audio`/`audio_url`. For the `_url` forms, the URL may be a string or the
 * usual `{ url: string }` object. A plain string is accepted as one text part.
 */
export function validateGemmaContentParts(
  content: unknown,
  limitOverrides: Partial<GemmaMediaLimits> = {}
): GemmaContentPart[] {
  return normalizeContentParts(content, resolveLimits(limitOverrides));
}

/** Process a mixed content list sequentially, preserving media/token order. */
export async function preprocessGemmaMedia(
  content: unknown,
  options: GemmaMediaPreprocessOptions = {}
): Promise<GemmaMediaInputs> {
  const resolved = resolveOptions(options);
  const contentParts = normalizeContentParts(content, resolved.limits);
  const byteBudget: ByteBudget = { used: 0, limit: resolved.limits.maxTotalBytes };
  const tensorBudget: TensorBudget = { used: 0, limit: resolved.limits.maxTensorBytes };
  const parts: GemmaPreparedContentPart[] = [];
  const images: GemmaVisionInput[] = [];
  const videos: GemmaVideoInput[] = [];
  const audios: GemmaAudioInput[] = [];

  for (const part of contentParts) {
    throwIfAborted(resolved.signal);
    if (part.type === "text") {
      parts.push(part);
      continue;
    }
    if (part.type === "image") {
      const image = await preprocessImageUrlInternal(
        part.url,
        part.detail === "low" ? { ...resolved, imageSoftTokens: 70 } : resolved,
        byteBudget,
        tensorBudget,
        "image"
      );
      const imageIndex = images.push(image) - 1;
      parts.push({ type: "image", imageIndex, sourceUrl: image.sourceUrl });
      continue;
    }
    if (part.type === "video") {
      const video = await preprocessVideoUrlInternal(
        part.url,
        part.frames === undefined ? resolved : { ...resolved, videoFrames: part.frames },
        byteBudget,
        tensorBudget,
        "video"
      );
      const videoIndex = videos.push(video) - 1;
      parts.push({ type: "video", videoIndex, sourceUrl: video.sourceUrl });
      continue;
    }
    const audio = await preprocessAudioUrlInternal(
      part.url,
      resolved,
      byteBudget,
      tensorBudget,
      "audio"
    );
    const audioIndex = audios.push(audio) - 1;
    parts.push({ type: "audio", audioIndex, sourceUrl: audio.sourceUrl });
  }

  return { parts, images, videos, audios };
}

/** Fetch, decode, resize, patchify, and pad one Gemma still image. */
export async function preprocessGemmaImageUrl(
  url: string,
  options: GemmaMediaPreprocessOptions = {}
): Promise<GemmaVisionInput> {
  const resolved = resolveOptions(options);
  const sourceUrl = normalizeMediaUrl(url, "image URL", resolved.limits.maxImageBytes);
  return preprocessImageUrlInternal(
    sourceUrl,
    resolved,
    { used: 0, limit: resolved.limits.maxTotalBytes },
    { used: 0, limit: resolved.limits.maxTensorBytes },
    "image"
  );
}

/** Fetch, decode, frame-sample, resize, patchify, and pad one Gemma video. */
export async function preprocessGemmaVideoUrl(
  url: string,
  options: GemmaMediaPreprocessOptions = {}
): Promise<GemmaVideoInput> {
  const resolved = resolveOptions(options);
  const sourceUrl = normalizeMediaUrl(url, "video URL", resolved.limits.maxVideoBytes);
  return preprocessVideoUrlInternal(
    sourceUrl,
    resolved,
    { used: 0, limit: resolved.limits.maxTotalBytes },
    { used: 0, limit: resolved.limits.maxTensorBytes },
    "video"
  );
}

/** Fetch, decode, resample, and calculate 16 kHz Gemma log-mel features. */
export async function preprocessGemmaAudioUrl(
  url: string,
  options: GemmaMediaPreprocessOptions = {}
): Promise<GemmaAudioInput> {
  const resolved = resolveOptions(options);
  const sourceUrl = normalizeMediaUrl(url, "audio URL", resolved.limits.maxAudioBytes);
  return preprocessAudioUrlInternal(
    sourceUrl,
    resolved,
    { used: 0, limit: resolved.limits.maxTotalBytes },
    { used: 0, limit: resolved.limits.maxTensorBytes },
    "audio"
  );
}

/**
 * Exposed for callers which want to inspect Gemma's aspect-ratio-preserving
 * resize before allocating a canvas. This mirrors the HF processor algorithm.
 */
export function getGemmaImageResize(
  sourceWidth: number,
  sourceHeight: number,
  maxSoftTokens: GemmaSoftTokenCount = 280
): { width: number; height: number; maxPatches: number } {
  assertPositiveInteger(sourceWidth, "sourceWidth");
  assertPositiveInteger(sourceHeight, "sourceHeight");
  assertSoftTokenCount(maxSoftTokens, "maxSoftTokens");
  const layout = makeVisionLayout(sourceWidth, sourceHeight, maxSoftTokens);
  return { width: layout.width, height: layout.height, maxPatches: layout.maxPatches };
}

function resolveOptions(options: GemmaMediaPreprocessOptions): ResolvedOptions {
  const limits = resolveLimits(options.limits ?? {});
  const imageSoftTokens = options.imageSoftTokens ?? 280;
  const videoSoftTokens = options.videoSoftTokens ?? 70;
  assertSoftTokenCount(imageSoftTokens, "imageSoftTokens");
  assertSoftTokenCount(videoSoftTokens, "videoSoftTokens");

  const defaultVideoFrames = Math.min(32, limits.maxVideoFrames);
  const videoFrames = options.videoFrames ?? defaultVideoFrames;
  assertPositiveInteger(videoFrames, "videoFrames");
  if (videoFrames > limits.maxVideoFrames) {
    throw new GemmaMediaError(
      `videoFrames (${videoFrames}) exceeds the configured maxVideoFrames (${limits.maxVideoFrames}).`,
      "video_frame_limit"
    );
  }
  return { signal: options.signal, limits, imageSoftTokens, videoSoftTokens, videoFrames };
}

function resolveLimits(overrides: Partial<GemmaMediaLimits>): GemmaMediaLimits {
  if (!isRecord(overrides)) {
    throw new GemmaMediaError("media limits must be an object.", "invalid_limits");
  }
  const merged: GemmaMediaLimits = { ...DEFAULT_GEMMA_MEDIA_LIMITS, ...overrides };
  const integerFields: Array<keyof GemmaMediaLimits> = [
    "maxContentParts",
    "maxImages",
    "maxVideos",
    "maxAudios",
    "maxImageBytes",
    "maxVideoBytes",
    "maxAudioBytes",
    "maxTotalBytes",
    "maxDecodedImagePixels",
    "maxDecodedVideoPixels",
    "maxDecodedAudioSamples",
    "maxAudioChannels",
    "maxVideoFrames",
    "maxTensorBytes",
  ];
  for (const field of integerFields) assertPositiveInteger(merged[field], `limits.${field}`);
  if (!Number.isFinite(merged.maxVideoDurationSeconds) || merged.maxVideoDurationSeconds <= 0) {
    throw new GemmaMediaError(
      "limits.maxVideoDurationSeconds must be a finite positive number.",
      "invalid_limits"
    );
  }
  if (!Number.isFinite(merged.maxAudioDurationSeconds) || merged.maxAudioDurationSeconds <= 0) {
    throw new GemmaMediaError(
      "limits.maxAudioDurationSeconds must be a finite positive number.",
      "invalid_limits"
    );
  }
  if (merged.maxImageBytes > merged.maxTotalBytes) {
    // This is valid: the total budget still caps a request. Do not reject it.
  }
  return merged;
}

function normalizeContentParts(content: unknown, limits: GemmaMediaLimits): GemmaContentPart[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (!Array.isArray(content) || content.length === 0) {
    throw new GemmaMediaError(
      "content must be a non-empty string or an array of text, image, video, and audio parts.",
      "invalid_content"
    );
  }
  if (content.length > limits.maxContentParts) {
    throw new GemmaMediaError(
      `content has ${content.length} parts; the configured limit is ${limits.maxContentParts}.`,
      "content_part_limit"
    );
  }

  let images = 0;
  let videos = 0;
  let audios = 0;
  let estimatedDataBytes = 0;
  const normalized: GemmaContentPart[] = [];

  for (let index = 0; index < content.length; index++) {
    const raw = content[index];
    const path = `content[${index}]`;
    if (!isRecord(raw) || typeof raw.type !== "string") {
      throw new GemmaMediaError(`${path} must be an object with a string type.`, "invalid_content");
    }
    if (raw.type === "text") {
      if (typeof raw.text !== "string") {
        throw new GemmaMediaError(`${path}.text must be a string.`, "invalid_content");
      }
      normalized.push({ type: "text", text: raw.text });
      continue;
    }

    const modality = modalityForType(raw.type);
    if (!modality) {
      throw new GemmaMediaError(
        `${path}.type=${JSON.stringify(raw.type)} is unsupported. Use text, image/image_url, video/video_url, or audio/audio_url.`,
        "unsupported_content_type"
      );
    }
    const maxBytes =
      modality === "image"
        ? limits.maxImageBytes
        : modality === "video"
          ? limits.maxVideoBytes
          : limits.maxAudioBytes;
    const url = normalizeMediaUrl(extractPartUrl(raw, modality, path), `${path} ${modality} URL`, maxBytes);
    if (url.startsWith(DATA_URL_PROTOCOL)) {
      estimatedDataBytes += estimateDataUrlBytes(url);
      if (estimatedDataBytes > limits.maxTotalBytes) {
        throw new GemmaMediaError(
          `${path} makes data-URL content exceed the configured total media budget of ${formatBytes(limits.maxTotalBytes)}.`,
          "media_byte_limit"
        );
      }
    }
    if (modality === "image") {
      images++;
      if (images > limits.maxImages) {
        throw new GemmaMediaError(
          `content has more than ${limits.maxImages} image parts.`,
          "image_count_limit"
        );
      }
      normalized.push({ type: "image", url, detail: imageDetail(raw, path) });
    } else if (modality === "video") {
      videos++;
      if (videos > limits.maxVideos) {
        throw new GemmaMediaError(
          `content has more than ${limits.maxVideos} video parts.`,
          "video_count_limit"
        );
      }
      normalized.push({ type: "video", url, frames: videoFrameLimit(raw, path, limits) });
    } else {
      audios++;
      if (audios > limits.maxAudios) {
        throw new GemmaMediaError(
          `content has more than ${limits.maxAudios} audio parts.`,
          "audio_count_limit"
        );
      }
      normalized.push({ type: "audio", url });
    }
  }
  return normalized;
}

function modalityForType(type: string): "image" | "video" | "audio" | null {
  if (type === "image" || type === "image_url") return "image";
  if (type === "video" || type === "video_url") return "video";
  if (type === "audio" || type === "audio_url") return "audio";
  return null;
}

function extractPartUrl(raw: Record<string, unknown>, modality: "image" | "video" | "audio", path: string): string {
  const keys = ["url", `${modality}_url`, modality];
  for (const key of keys) {
    const candidate = raw[key];
    if (typeof candidate === "string") return candidate;
    if (isRecord(candidate) && typeof candidate.url === "string") return candidate.url;
  }
  throw new GemmaMediaError(
    `${path} must provide a non-empty URL as .url or .${modality}_url.url.`,
    "invalid_media_url"
  );
}

function imageDetail(raw: Record<string, unknown>, path: string): "low" | "high" | undefined {
  const nested = raw.image_url;
  const value = raw.detail ?? (isRecord(nested) ? nested.detail : undefined);
  if (value === undefined || value === "auto" || value === "high") return value === "high" ? "high" : undefined;
  if (value === "low") return "low";
  throw new GemmaMediaError(`${path}.detail must be "low", "high", or "auto".`, "invalid_content");
}

function videoFrameLimit(raw: Record<string, unknown>, path: string, limits: GemmaMediaLimits): number | undefined {
  const nested = raw.video_url;
  const value = raw.frames ?? (isRecord(nested) ? nested.frames : undefined);
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1 || value > limits.maxVideoFrames) {
    throw new GemmaMediaError(
      `${path}.frames must be an integer from 1 to ${limits.maxVideoFrames}.`,
      "video_frame_limit"
    );
  }
  return value;
}

function normalizeMediaUrl(value: string, name: string, maxBytes: number): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new GemmaMediaError(`${name} must be a non-empty string.`, "invalid_media_url");
  }
  if (value.length > maxBytes * 2 && value.trimStart().startsWith(DATA_URL_PROTOCOL)) {
    throw new GemmaMediaError(
      `${name} is too large before decoding; its data URL exceeds the ${formatBytes(maxBytes)} source limit.`,
      "media_byte_limit"
    );
  }

  let parsed: URL;
  try {
    const base = typeof location !== "undefined" ? location.href : undefined;
    parsed = base ? new URL(value, base) : new URL(value);
  } catch {
    throw new GemmaMediaError(`${name} is not a valid absolute browser URL.`, "invalid_media_url");
  }
  if (!["https:", "http:", "blob:", DATA_URL_PROTOCOL].includes(parsed.protocol)) {
    throw new GemmaMediaError(
      `${name} must use https:, http:, blob:, or data:; ${parsed.protocol || "this protocol"} is not allowed.`,
      "unsupported_media_protocol"
    );
  }
  if ((parsed.protocol === "https:" || parsed.protocol === "http:") && (parsed.username || parsed.password)) {
    throw new GemmaMediaError(`${name} must not embed credentials.`, "invalid_media_url");
  }
  if (parsed.protocol === DATA_URL_PROTOCOL && estimateDataUrlBytes(parsed.href) > maxBytes) {
    throw new GemmaMediaError(
      `${name} exceeds the ${formatBytes(maxBytes)} source limit.`,
      "media_byte_limit"
    );
  }
  return parsed.href;
}

async function preprocessImageUrlInternal(
  sourceUrl: string,
  options: ResolvedOptions,
  byteBudget: ByteBudget,
  tensorBudget: TensorBudget,
  label: string
): Promise<GemmaVisionInput> {
  const blob = await fetchMediaBlob(
    sourceUrl,
    label,
    options.limits.maxImageBytes,
    byteBudget,
    options.signal
  );
  return preprocessImageBlob(sourceUrl, blob, options, tensorBudget);
}

async function preprocessImageBlob(
  sourceUrl: string,
  blob: Blob,
  options: ResolvedOptions,
  tensorBudget: TensorBudget
): Promise<GemmaVisionInput> {
  const decoded = await decodeImageBlob(blob, sourceLabel(sourceUrl), options.signal);
  try {
    ensureDecodedPixels(
      decoded.width,
      decoded.height,
      options.limits.maxDecodedImagePixels,
      "image"
    );
    const layout = makeVisionLayout(decoded.width, decoded.height, options.imageSoftTokens);
    reserveTensorBytes(
      tensorBudget,
      layout.maxPatches * (GEMMA_IMAGE_PATCH_PIXELS * Float32Array.BYTES_PER_ELEMENT + 2 * Int32Array.BYTES_PER_ELEMENT),
      "image patch tensors"
    );
    const rgba = drawSourceToRgba(decoded.source, layout.width, layout.height);
    return makeVisionInput(sourceUrl, rgba, layout);
  } finally {
    decoded.close();
  }
}

async function preprocessVideoUrlInternal(
  sourceUrl: string,
  options: ResolvedOptions,
  byteBudget: ByteBudget,
  tensorBudget: TensorBudget,
  label: string
): Promise<GemmaVideoInput> {
  const blob = await fetchMediaBlob(
    sourceUrl,
    label,
    options.limits.maxVideoBytes,
    byteBudget,
    options.signal
  );
  return preprocessVideoBlob(sourceUrl, blob, options, tensorBudget);
}

async function preprocessVideoBlob(
  sourceUrl: string,
  blob: Blob,
  options: ResolvedOptions,
  tensorBudget: TensorBudget
): Promise<GemmaVideoInput> {
  ensureBrowserDom("Video preprocessing");
  const objectUrl = URL.createObjectURL(blob);
  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = "anonymous";
  video.disablePictureInPicture = true;

  try {
    const metadata = waitForMediaEvent(
      video,
      ["loadedmetadata"],
      ["error", "abort"],
      options.signal,
      "video metadata"
    );
    video.src = objectUrl;
    video.load();
    await metadata;
    throwIfAborted(options.signal);

    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    if (!Number.isInteger(sourceWidth) || !Number.isInteger(sourceHeight) || sourceWidth < 1 || sourceHeight < 1) {
      throw new GemmaMediaError(
        `Could not read video dimensions for ${sourceLabel(sourceUrl)}.`,
        "video_decode"
      );
    }
    ensureDecodedPixels(sourceWidth, sourceHeight, options.limits.maxDecodedVideoPixels, "video");
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration < 0) {
      throw new GemmaMediaError(
        `Video ${sourceLabel(sourceUrl)} has no finite duration and cannot be sampled.`,
        "video_duration"
      );
    }
    if (duration > options.limits.maxVideoDurationSeconds) {
      throw new GemmaMediaError(
        `Video ${sourceLabel(sourceUrl)} is ${duration.toFixed(1)} seconds; the configured limit is ${options.limits.maxVideoDurationSeconds} seconds.`,
        "video_duration_limit"
      );
    }

    const layout = makeVisionLayout(sourceWidth, sourceHeight, options.videoSoftTokens);
    const perFramePixelBytes = layout.maxPatches * GEMMA_IMAGE_PATCH_PIXELS * Float32Array.BYTES_PER_ELEMENT;
    const perFramePositionBytes = layout.maxPatches * 2 * Int32Array.BYTES_PER_ELEMENT;
    reserveTensorBytes(
      tensorBudget,
      options.videoFrames * (perFramePixelBytes + perFramePositionBytes) + options.videoFrames * Float64Array.BYTES_PER_ELEMENT,
      "video frame tensors"
    );

    const pixelValues = new Float32Array(options.videoFrames * layout.maxPatches * GEMMA_IMAGE_PATCH_PIXELS);
    const positionIds = new Int32Array(options.videoFrames * layout.maxPatches * 2);
    positionIds.fill(-1);
    const timestamps = new Float64Array(options.videoFrames);
    const positions = makePatchPositions(layout);
    const canvas = createCanvasSurface(layout.width, layout.height);

    for (let frame = 0; frame < options.videoFrames; frame++) {
      throwIfAborted(options.signal);
      const timestamp = videoFrameTimestamp(frame, options.videoFrames, duration);
      await seekVideo(video, timestamp, options.signal);
      const rgba = drawVideoFrameToRgba(canvas, video, layout.width, layout.height);
      const pixelOffset = frame * layout.maxPatches * GEMMA_IMAGE_PATCH_PIXELS;
      patchifyRgba(rgba, layout, pixelValues, pixelOffset);
      positionIds.set(positions, frame * layout.maxPatches * 2);
      timestamps[frame] = timestamp;
    }

    return {
      sourceUrl,
      pixelValues,
      positionIds,
      timestamps,
      frameCount: options.videoFrames,
      patchCountPerFrame: layout.patchCount,
      numSoftTokensPerFrame: layout.numSoftTokens,
      width: layout.width,
      height: layout.height,
      maxPatches: layout.maxPatches,
      patchSize: GEMMA_IMAGE_PATCH_SIZE,
    };
  } catch (error) {
    if (error instanceof GemmaMediaError || isAbortError(error)) throw error;
    throw new GemmaMediaError(
      `Could not decode video ${sourceLabel(sourceUrl)}. Ensure the browser supports its codec and the URL is CORS-accessible.`,
      "video_decode"
    );
  } finally {
    video.removeAttribute("src");
    try {
      video.load();
    } catch {
      // Releasing a detached media element is best effort.
    }
    video.remove();
    URL.revokeObjectURL(objectUrl);
  }
}

async function preprocessAudioUrlInternal(
  sourceUrl: string,
  options: ResolvedOptions,
  byteBudget: ByteBudget,
  tensorBudget: TensorBudget,
  label: string
): Promise<GemmaAudioInput> {
  const blob = await fetchMediaBlob(
    sourceUrl,
    label,
    options.limits.maxAudioBytes,
    byteBudget,
    options.signal
  );
  return preprocessAudioBlob(sourceUrl, blob, options, tensorBudget);
}

async function preprocessAudioBlob(
  sourceUrl: string,
  blob: Blob,
  options: ResolvedOptions,
  tensorBudget: TensorBudget
): Promise<GemmaAudioInput> {
  const audioBuffer = await decodeAudioBlob(blob, sourceLabel(sourceUrl), options.signal);
  if (!Number.isFinite(audioBuffer.duration) || audioBuffer.duration <= 0) {
    throw new GemmaMediaError(`Audio ${sourceLabel(sourceUrl)} is empty.`, "audio_decode");
  }
  if (audioBuffer.duration > options.limits.maxAudioDurationSeconds) {
    throw new GemmaMediaError(
      `Audio ${sourceLabel(sourceUrl)} is ${audioBuffer.duration.toFixed(1)} seconds; the configured limit is ${options.limits.maxAudioDurationSeconds} seconds.`,
      "audio_duration_limit"
    );
  }
  if (audioBuffer.numberOfChannels < 1 || audioBuffer.numberOfChannels > options.limits.maxAudioChannels) {
    throw new GemmaMediaError(
      `Audio ${sourceLabel(sourceUrl)} has ${audioBuffer.numberOfChannels} channels; supported range is 1-${options.limits.maxAudioChannels}.`,
      "audio_channel_limit"
    );
  }
  if (audioBuffer.length > options.limits.maxDecodedAudioSamples) {
    throw new GemmaMediaError(
      `Audio ${sourceLabel(sourceUrl)} has ${formatNumber(audioBuffer.length)} decoded samples per channel; the configured limit is ${formatNumber(options.limits.maxDecodedAudioSamples)}.`,
      "decoded_audio_limit"
    );
  }

  const mono = downmixToMono(audioBuffer);
  const samples = linearResample(mono, audioBuffer.sampleRate, GEMMA_AUDIO_SAMPLE_RATE);
  const maxSamples = Math.floor(options.limits.maxAudioDurationSeconds * GEMMA_AUDIO_SAMPLE_RATE);
  if (samples.length > maxSamples) {
    throw new GemmaMediaError(
      `Audio ${sourceLabel(sourceUrl)} exceeds the ${maxSamples}-sample 16 kHz limit after resampling.`,
      "audio_duration_limit"
    );
  }
  if (samples.length <= GEMMA_AUDIO_HOP_LENGTH) {
    throw new GemmaMediaError(
      `Audio ${sourceLabel(sourceUrl)} is too short; Gemma needs more than ${GEMMA_AUDIO_HOP_LENGTH} samples at 16 kHz.`,
      "audio_too_short"
    );
  }

  const frameCount = getAudioFrameCount(samples.length);
  reserveTensorBytes(
    tensorBudget,
    frameCount * (GEMMA_AUDIO_FEATURE_SIZE * Float32Array.BYTES_PER_ELEMENT + Uint8Array.BYTES_PER_ELEMENT),
    "audio feature tensors"
  );
  const { inputFeatures, inputFeaturesMask } = extractGemmaLogMel(samples, frameCount);
  return {
    sourceUrl,
    inputFeatures,
    inputFeaturesMask,
    frameCount,
    featureSize: GEMMA_AUDIO_FEATURE_SIZE,
    sampleRate: GEMMA_AUDIO_SAMPLE_RATE,
    sampleCount: samples.length,
    durationSeconds: samples.length / GEMMA_AUDIO_SAMPLE_RATE,
  };
}

async function fetchMediaBlob(
  url: string,
  kind: string,
  maxBytes: number,
  budget: ByteBudget,
  signal?: AbortSignal
): Promise<Blob> {
  throwIfAborted(signal);
  let response: Response;
  try {
    response = await fetch(url, { credentials: "omit", signal });
  } catch (error) {
    if (isAbortError(error)) throw error;
    throw new GemmaMediaError(
      `Could not fetch ${kind} from ${sourceLabel(url)}. Ensure the URL is reachable and permits CORS requests.`,
      "media_fetch"
    );
  }
  if (!response.ok) {
    throw new GemmaMediaError(
      `Could not fetch ${kind} from ${sourceLabel(url)}: HTTP ${response.status}.`,
      "media_fetch"
    );
  }
  if (response.type === "opaque") {
    throw new GemmaMediaError(
      `Could not read ${kind} from ${sourceLabel(url)} because the response is opaque. The host must allow CORS.`,
      "media_cors"
    );
  }
  if (response.url) {
    // Redirects should not turn an approved http(s) URL into an exotic scheme.
    normalizeMediaUrl(response.url, `${kind} redirect URL`, maxBytes);
  }

  const declaredLength = response.headers.get("content-length");
  if (declaredLength !== null) {
    const bytes = Number(declaredLength);
    if (Number.isFinite(bytes) && bytes >= 0) {
      ensureSourceBytes(bytes, maxBytes, budget, kind);
    }
  }

  if (!response.body) {
    const blob = await response.blob();
    ensureSourceBytes(blob.size, maxBytes, budget, kind);
    budget.used += blob.size;
    return blob;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    for (;;) {
      throwIfAborted(signal);
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      bytes += value.byteLength;
      ensureSourceBytes(bytes, maxBytes, budget, kind);
      chunks.push(value);
    }
  } catch (error) {
    try {
      await reader.cancel();
    } catch {
      // The reader may already be closed after an aborted request.
    }
    throw error;
  }
  budget.used += bytes;
  return new Blob(chunks as unknown as BlobPart[], {
    type: response.headers.get("content-type") ?? "",
  });
}

function ensureSourceBytes(bytes: number, maxBytes: number, budget: ByteBudget, kind: string): void {
  if (bytes > maxBytes) {
    throw new GemmaMediaError(
      `${kind} source is ${formatBytes(bytes)}; the per-item limit is ${formatBytes(maxBytes)}.`,
      "media_byte_limit"
    );
  }
  if (budget.used + bytes > budget.limit) {
    throw new GemmaMediaError(
      `Media sources exceed the configured total limit of ${formatBytes(budget.limit)}.`,
      "media_byte_limit"
    );
  }
}

async function decodeImageBlob(blob: Blob, label: string, signal?: AbortSignal): Promise<DecodedImage> {
  throwIfAborted(signal);
  let bitmapFailure: unknown;
  if (typeof createImageBitmap === "function") {
    try {
      const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
      throwIfAborted(signal);
      if (bitmap.width < 1 || bitmap.height < 1) {
        bitmap.close();
        throw new GemmaMediaError(`Image ${label} has invalid dimensions.`, "image_decode");
      }
      return { source: bitmap, width: bitmap.width, height: bitmap.height, close: () => bitmap.close() };
    } catch (error) {
      if (isAbortError(error) || error instanceof GemmaMediaError) throw error;
      bitmapFailure = error;
    }
  }

  ensureBrowserDom("Image preprocessing");
  const objectUrl = URL.createObjectURL(blob);
  const image = document.createElement("img");
  image.decoding = "async";
  try {
    const loaded = waitForMediaEvent(image, ["load"], ["error", "abort"], signal, "image decode");
    image.src = objectUrl;
    await loaded;
    throwIfAborted(signal);
    if (image.naturalWidth < 1 || image.naturalHeight < 1) {
      throw new GemmaMediaError(`Image ${label} has invalid dimensions.`, "image_decode");
    }
    return {
      source: image,
      width: image.naturalWidth,
      height: image.naturalHeight,
      close: () => URL.revokeObjectURL(objectUrl),
    };
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    if (error instanceof GemmaMediaError || isAbortError(error)) throw error;
    const bitmapHint = bitmapFailure ? " createImageBitmap also rejected the source." : "";
    throw new GemmaMediaError(
      `Could not decode image ${label}.${bitmapHint} Use a browser-supported image format.`,
      "image_decode"
    );
  }
}

async function decodeAudioBlob(blob: Blob, label: string, signal?: AbortSignal): Promise<AudioBuffer> {
  throwIfAborted(signal);
  const audioContextConstructor = getAudioContextConstructor();
  if (!audioContextConstructor) {
    throw new GemmaMediaError(
      `Audio preprocessing for ${label} requires the browser AudioContext API.`,
      "audio_api_unavailable"
    );
  }
  let context: AudioContext | undefined;
  try {
    context = new audioContextConstructor();
    const encoded = await blob.arrayBuffer();
    throwIfAborted(signal);
    const decoded = await context.decodeAudioData(encoded);
    throwIfAborted(signal);
    return decoded;
  } catch (error) {
    if (error instanceof GemmaMediaError || isAbortError(error)) throw error;
    throw new GemmaMediaError(
      `Could not decode audio ${label}. Use a browser-supported audio codec.`,
      "audio_decode"
    );
  } finally {
    if (context) {
      try {
        await context.close();
      } catch {
        // Closing an already closed/suspended context is not relevant to the result.
      }
    }
  }
}

function makeVisionLayout(sourceWidth: number, sourceHeight: number, maxSoftTokens: GemmaSoftTokenCount): VisionLayout {
  const sourcePixels = sourceWidth * sourceHeight;
  if (!Number.isSafeInteger(sourcePixels) || sourcePixels < 1) {
    throw new GemmaMediaError("Gemma image source dimensions are too large.", "image_resize");
  }
  const maxPatches = maxSoftTokens * GEMMA_IMAGE_POOLING_KERNEL_SIZE ** 2;
  const targetPixels = maxPatches * GEMMA_IMAGE_PATCH_SIZE ** 2;
  const scale = Math.sqrt(targetPixels / sourcePixels);
  let height = Math.floor((scale * sourceHeight) / SIDE_MULTIPLE) * SIDE_MULTIPLE;
  let width = Math.floor((scale * sourceWidth) / SIDE_MULTIPLE) * SIDE_MULTIPLE;
  const maxSideLength = (maxPatches / GEMMA_IMAGE_POOLING_KERNEL_SIZE ** 2) * SIDE_MULTIPLE;

  // This is the same thin-image handling used by Gemma4ImageProcessor.
  if (height === 0 && width === 0) {
    throw new GemmaMediaError(
      `Cannot resize ${sourceWidth}x${sourceHeight} image to Gemma's ${SIDE_MULTIPLE}-pixel grid.`,
      "image_resize"
    );
  }
  if (height === 0) {
    height = SIDE_MULTIPLE;
    width = Math.min(Math.floor(sourceWidth / sourceHeight) * SIDE_MULTIPLE, maxSideLength);
  } else if (width === 0) {
    width = SIDE_MULTIPLE;
    height = Math.min(Math.floor(sourceHeight / sourceWidth) * SIDE_MULTIPLE, maxSideLength);
  }
  if (width < SIDE_MULTIPLE || height < SIDE_MULTIPLE || width * height > targetPixels) {
    throw new GemmaMediaError(
      `Gemma resize of ${sourceWidth}x${sourceHeight} produced an invalid ${width}x${height} grid.`,
      "image_resize"
    );
  }
  const patchWidth = width / GEMMA_IMAGE_PATCH_SIZE;
  const patchHeight = height / GEMMA_IMAGE_PATCH_SIZE;
  const patchCount = patchWidth * patchHeight;
  if (!Number.isInteger(patchCount) || patchCount > maxPatches || patchCount % (GEMMA_IMAGE_POOLING_KERNEL_SIZE ** 2) !== 0) {
    throw new GemmaMediaError("Gemma image patch layout is invalid.", "image_resize");
  }
  return {
    width,
    height,
    patchWidth,
    patchHeight,
    patchCount,
    maxPatches,
    numSoftTokens: patchCount / GEMMA_IMAGE_POOLING_KERNEL_SIZE ** 2,
  };
}

function makeVisionInput(sourceUrl: string, rgba: Uint8ClampedArray, layout: VisionLayout): GemmaVisionInput {
  const pixelValues = new Float32Array(layout.maxPatches * GEMMA_IMAGE_PATCH_PIXELS);
  const positionIds = makePatchPositions(layout);
  patchifyRgba(rgba, layout, pixelValues, 0);
  return {
    sourceUrl,
    pixelValues,
    positionIds,
    patchCount: layout.patchCount,
    numSoftTokens: layout.numSoftTokens,
    width: layout.width,
    height: layout.height,
    maxPatches: layout.maxPatches,
    patchSize: GEMMA_IMAGE_PATCH_SIZE,
  };
}

function makePatchPositions(layout: VisionLayout): Int32Array {
  const positions = new Int32Array(layout.maxPatches * 2);
  positions.fill(-1);
  let patch = 0;
  for (let y = 0; y < layout.patchHeight; y++) {
    for (let x = 0; x < layout.patchWidth; x++, patch++) {
      positions[patch * 2] = x;
      positions[patch * 2 + 1] = y;
    }
  }
  return positions;
}

export function patchifyRgba(
  rgba: Uint8ClampedArray,
  layout: VisionLayout,
  output: Float32Array,
  outputOffset: number
): void {
  const expected = layout.width * layout.height * 4;
  if (rgba.length !== expected) {
    throw new GemmaMediaError("Canvas returned an unexpected image buffer size.", "image_canvas");
  }
  let patch = 0;
  for (let patchY = 0; patchY < layout.patchHeight; patchY++) {
    for (let patchX = 0; patchX < layout.patchWidth; patchX++, patch++) {
      // llama.cpp's GGUF converter stores the patch projection in CHW order.
      // Match it here so the native matrix can be consumed without a loader-
      // side permutation or duplicate transformed cache.
      const destination = outputOffset + patch * GEMMA_IMAGE_PATCH_PIXELS;
      for (let channel = 0; channel < 3; channel++) {
        for (let y = 0; y < GEMMA_IMAGE_PATCH_SIZE; y++) {
          let source = ((patchY * GEMMA_IMAGE_PATCH_SIZE + y) * layout.width + patchX * GEMMA_IMAGE_PATCH_SIZE) * 4;
          for (let x = 0; x < GEMMA_IMAGE_PATCH_SIZE; x++) {
            output[destination + channel * GEMMA_IMAGE_PATCH_SIZE ** 2 + y * GEMMA_IMAGE_PATCH_SIZE + x] = rgba[source + channel] / 255;
            source += 4;
          }
        }
      }
    }
  }
}

function drawSourceToRgba(source: CanvasImageSource, width: number, height: number): Uint8ClampedArray {
  const canvas = createCanvasSurface(width, height);
  const context = canvas.context;
  context.clearRect(0, 0, width, height);
  context.imageSmoothingEnabled = true;
  // OffscreenCanvasRenderingContext2D implements this in browsers even where
  // its TypeScript declaration is narrower than CanvasRenderingContext2D.
  (context as CanvasRenderingContext2D).imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, width, height);
  return context.getImageData(0, 0, width, height).data;
}

function drawVideoFrameToRgba(
  canvas: CanvasSurface,
  video: HTMLVideoElement,
  width: number,
  height: number
): Uint8ClampedArray {
  const context = canvas.context;
  context.clearRect(0, 0, width, height);
  context.imageSmoothingEnabled = true;
  (context as CanvasRenderingContext2D).imageSmoothingQuality = "high";
  context.drawImage(video, 0, 0, width, height);
  return context.getImageData(0, 0, width, height).data;
}

function createCanvasSurface(width: number, height: number): CanvasSurface {
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new GemmaMediaError("Could not create an OffscreenCanvas 2D context.", "canvas_unavailable");
    return { context };
  }
  ensureBrowserDom("Image preprocessing");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new GemmaMediaError("Could not create a Canvas 2D context.", "canvas_unavailable");
  return { context };
}

function videoFrameTimestamp(index: number, frameCount: number, duration: number): number {
  if (duration === 0 || frameCount === 1) return 0;
  // Sample bin centres, avoiding a seek exactly at the exclusive end timestamp.
  return Math.max(0, Math.min(duration - 0.001, ((index + 0.5) / frameCount) * duration));
}

async function seekVideo(video: HTMLVideoElement, timestamp: number, signal?: AbortSignal): Promise<void> {
  throwIfAborted(signal);
  if (Math.abs(video.currentTime - timestamp) < 0.001) {
    if (video.readyState >= 2) return;
    await waitForMediaEvent(video, ["loadeddata", "canplay"], ["error", "abort"], signal, "video frame decode");
    return;
  }
  const seeked = waitForMediaEvent(video, ["seeked"], ["error", "abort"], signal, "video frame seek");
  video.currentTime = timestamp;
  await seeked;
  if (video.readyState < 2) {
    await waitForMediaEvent(video, ["loadeddata", "canplay"], ["error", "abort"], signal, "video frame decode");
  }
}

function getAudioFrameCount(sampleCount: number): number {
  // The upstream extractor first pads waveform length to a multiple of 128,
  // then prepends 160 semicausal samples and unfolds 321-sample windows.
  const paddedSamples = Math.ceil(sampleCount / 128) * 128;
  const frameCount = Math.floor((paddedSamples - (GEMMA_AUDIO_FRAME_LENGTH + 1 - GEMMA_AUDIO_HOP_LENGTH)) / GEMMA_AUDIO_HOP_LENGTH) + 1;
  if (!Number.isInteger(frameCount) || frameCount < 1) {
    throw new GemmaMediaError("Audio is too short to form a Gemma log-mel frame.", "audio_too_short");
  }
  return frameCount;
}

function downmixToMono(audio: AudioBuffer): Float32Array {
  const channels = audio.numberOfChannels;
  const output = new Float32Array(audio.length);
  for (let channel = 0; channel < channels; channel++) {
    const input = audio.getChannelData(channel);
    for (let sample = 0; sample < input.length; sample++) {
      const value = input[sample];
      output[sample] += Number.isFinite(value) ? value / channels : 0;
    }
  }
  return output;
}

function linearResample(input: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (!Number.isFinite(sourceRate) || sourceRate <= 0) {
    throw new GemmaMediaError("Decoded audio has an invalid sample rate.", "audio_decode");
  }
  if (sourceRate === targetRate) return new Float32Array(input);
  const outputLength = Math.max(1, Math.round((input.length * targetRate) / sourceRate));
  const output = new Float32Array(outputLength);
  const rate = sourceRate / targetRate;
  for (let index = 0; index < outputLength; index++) {
    const position = index * rate;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const mix = position - left;
    output[index] = input[left] * (1 - mix) + input[right] * mix;
  }
  return output;
}

function extractGemmaLogMel(samples: Float32Array, frameCount: number): {
  inputFeatures: Float32Array;
  inputFeaturesMask: Uint8Array;
} {
  const features = new Float32Array(frameCount * GEMMA_AUDIO_FEATURE_SIZE);
  const mask = new Uint8Array(frameCount);
  const plan = getFftPlan();
  const melFilters = getMelFilters();
  const window = getHannWindow();
  const re = new Float64Array(GEMMA_AUDIO_FFT_LENGTH);
  const im = new Float64Array(GEMMA_AUDIO_FFT_LENGTH);
  const magnitudes = new Float64Array(GEMMA_AUDIO_FFT_LENGTH / 2 + 1);
  const leftPadding = GEMMA_AUDIO_FRAME_LENGTH / 2;

  for (let frame = 0; frame < frameCount; frame++) {
    // Gemma4AudioFeatureExtractor marks a frame valid only if the last sample
    // of its 321-sample analysis window is real (not semicausal/padding zero).
    const sourceEnd = frame * GEMMA_AUDIO_HOP_LENGTH + GEMMA_AUDIO_FRAME_LENGTH - leftPadding;
    if (sourceEnd < 0 || sourceEnd >= samples.length) continue;
    mask[frame] = 1;
    re.fill(0);
    im.fill(0);
    const frameStart = frame * GEMMA_AUDIO_HOP_LENGTH - leftPadding;
    for (let sample = 0; sample < GEMMA_AUDIO_FRAME_LENGTH; sample++) {
      const sourceIndex = frameStart + sample;
      const value = sourceIndex >= 0 && sourceIndex < samples.length ? samples[sourceIndex] : 0;
      re[plan.bitReversed[sample]] = value * window[sample];
    }
    fftInPlace(re, im, plan);
    for (let bin = 0; bin < magnitudes.length; bin++) {
      magnitudes[bin] = Math.sqrt(re[bin] * re[bin] + im[bin] * im[bin]);
    }
    const outputOffset = frame * GEMMA_AUDIO_FEATURE_SIZE;
    for (let mel = 0; mel < GEMMA_AUDIO_FEATURE_SIZE; mel++) {
      let energy = 0;
      const filterOffset = mel * magnitudes.length;
      for (let bin = 0; bin < magnitudes.length; bin++) {
        energy += magnitudes[bin] * melFilters[filterOffset + bin];
      }
      // Processor config: mel_floor=0.001, no mean/std normalization.
      features[outputOffset + mel] = Math.log(energy + 0.001);
    }
  }
  return { inputFeatures: features, inputFeaturesMask: mask };
}

function getFftPlan(): FftPlan {
  if (cachedFftPlan) return cachedFftPlan;
  const size = GEMMA_AUDIO_FFT_LENGTH;
  const bits = Math.log2(size);
  const bitReversed = new Uint16Array(size);
  for (let index = 0; index < size; index++) {
    let value = index;
    let reversed = 0;
    for (let bit = 0; bit < bits; bit++) {
      reversed = (reversed << 1) | (value & 1);
      value >>>= 1;
    }
    bitReversed[index] = reversed;
  }
  const cos: Float64Array[] = [];
  const sin: Float64Array[] = [];
  for (let width = 2; width <= size; width <<= 1) {
    const half = width >>> 1;
    const stageCos = new Float64Array(half);
    const stageSin = new Float64Array(half);
    for (let index = 0; index < half; index++) {
      const angle = (-2 * Math.PI * index) / width;
      stageCos[index] = Math.cos(angle);
      stageSin[index] = Math.sin(angle);
    }
    cos.push(stageCos);
    sin.push(stageSin);
  }
  cachedFftPlan = { bitReversed, cos, sin };
  return cachedFftPlan;
}

function fftInPlace(re: Float64Array, im: Float64Array, plan: FftPlan): void {
  let stage = 0;
  for (let width = 2; width <= GEMMA_AUDIO_FFT_LENGTH; width <<= 1, stage++) {
    const half = width >>> 1;
    const cos = plan.cos[stage];
    const sin = plan.sin[stage];
    for (let start = 0; start < GEMMA_AUDIO_FFT_LENGTH; start += width) {
      for (let index = 0; index < half; index++) {
        const even = start + index;
        const odd = even + half;
        const oddRe = re[odd];
        const oddIm = im[odd];
        const transformedRe = oddRe * cos[index] - oddIm * sin[index];
        const transformedIm = oddRe * sin[index] + oddIm * cos[index];
        const evenRe = re[even];
        const evenIm = im[even];
        re[even] = evenRe + transformedRe;
        im[even] = evenIm + transformedIm;
        re[odd] = evenRe - transformedRe;
        im[odd] = evenIm - transformedIm;
      }
    }
  }
}

function getHannWindow(): Float64Array {
  if (cachedHannWindow) return cachedHannWindow;
  const window = new Float64Array(GEMMA_AUDIO_FRAME_LENGTH);
  for (let index = 0; index < window.length; index++) {
    // Periodic Hann: np.hanning(frameLength + 1)[:-1].
    window[index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / GEMMA_AUDIO_FRAME_LENGTH);
  }
  cachedHannWindow = window;
  return window;
}

function getMelFilters(): Float64Array {
  if (cachedMelFilters) return cachedMelFilters;
  const frequencyBins = GEMMA_AUDIO_FFT_LENGTH / 2 + 1;
  const filters = new Float64Array(GEMMA_AUDIO_FEATURE_SIZE * frequencyBins);
  const melMin = hertzToMel(0);
  const melMax = hertzToMel(GEMMA_AUDIO_SAMPLE_RATE / 2);
  const points = new Float64Array(GEMMA_AUDIO_FEATURE_SIZE + 2);
  for (let index = 0; index < points.length; index++) {
    points[index] = melToHertz(melMin + ((melMax - melMin) * index) / (points.length - 1));
  }
  for (let mel = 0; mel < GEMMA_AUDIO_FEATURE_SIZE; mel++) {
    const left = points[mel];
    const center = points[mel + 1];
    const right = points[mel + 2];
    for (let bin = 0; bin < frequencyBins; bin++) {
      const frequency = (GEMMA_AUDIO_SAMPLE_RATE * bin) / GEMMA_AUDIO_FFT_LENGTH;
      const down = (frequency - left) / (center - left);
      const up = (right - frequency) / (right - center);
      filters[mel * frequencyBins + bin] = Math.max(0, Math.min(down, up));
    }
  }
  cachedMelFilters = filters;
  return filters;
}

function hertzToMel(frequency: number): number {
  return 2595 * Math.log10(1 + frequency / 700);
}

function melToHertz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

function waitForMediaEvent(
  target: EventTarget,
  successEvents: readonly string[],
  failureEvents: readonly string[],
  signal: AbortSignal | undefined,
  operation: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      for (const event of successEvents) target.removeEventListener(event, onSuccess);
      for (const event of failureEvents) target.removeEventListener(event, onFailure);
      signal?.removeEventListener("abort", onAbort);
      clearTimeout(timeout);
    };
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (error) reject(error);
      else resolve();
    };
    const onSuccess = () => finish();
    const onFailure = () =>
      finish(new GemmaMediaError(`Browser failed while waiting for ${operation}.`, "media_decode"));
    const onAbort = () => finish(abortError(signal));
    const timeout = setTimeout(
      () => finish(new GemmaMediaError(`Timed out after ${MEDIA_EVENT_TIMEOUT_MS / 1000}s waiting for ${operation}.`, "media_timeout")),
      MEDIA_EVENT_TIMEOUT_MS
    );
    for (const event of successEvents) target.addEventListener(event, onSuccess, { once: true });
    for (const event of failureEvents) target.addEventListener(event, onFailure, { once: true });
    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener("abort", onAbort, { once: true });
    }
  });
}

function getAudioContextConstructor(): (new () => AudioContext) | undefined {
  const browserGlobal = globalThis as typeof globalThis & {
    webkitAudioContext?: new () => AudioContext;
  };
  return browserGlobal.AudioContext ?? browserGlobal.webkitAudioContext;
}

function reserveTensorBytes(budget: TensorBudget, bytes: number, label: string): void {
  if (!Number.isSafeInteger(bytes) || bytes < 0) {
    throw new GemmaMediaError(`Cannot allocate an invalid ${label} size.`, "tensor_limit");
  }
  if (budget.used + bytes > budget.limit) {
    throw new GemmaMediaError(
      `${label} would exceed the configured ${formatBytes(budget.limit)} tensor-memory limit.`,
      "tensor_limit"
    );
  }
  budget.used += bytes;
}

function ensureDecodedPixels(width: number, height: number, maxPixels: number, kind: string): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new GemmaMediaError(`Decoded ${kind} has invalid dimensions.`, "media_decode");
  }
  const pixels = width * height;
  if (!Number.isSafeInteger(pixels) || pixels > maxPixels) {
    throw new GemmaMediaError(
      `Decoded ${kind} is ${width}x${height} (${formatNumber(pixels)} pixels); the configured limit is ${formatNumber(maxPixels)} pixels.`,
      "decoded_pixel_limit"
    );
  }
}

function ensureBrowserDom(operation: string): void {
  if (typeof document === "undefined" || typeof URL === "undefined") {
    throw new GemmaMediaError(`${operation} requires a browser DOM.`, "browser_api_unavailable");
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortError(signal);
}

function abortError(signal?: AbortSignal): Error {
  if (signal?.reason instanceof Error) return signal.reason;
  const error = new Error("Gemma media preprocessing was aborted.");
  error.name = "AbortError";
  return error;
}

function isAbortError(error: unknown): error is Error {
  return error instanceof Error && error.name === "AbortError";
}

function assertSoftTokenCount(value: number, name: string): asserts value is GemmaSoftTokenCount {
  if (!(SUPPORTED_SOFT_TOKEN_COUNTS as readonly number[]).includes(value)) {
    throw new GemmaMediaError(
      `${name} must be one of ${SUPPORTED_SOFT_TOKEN_COUNTS.join(", ")}; got ${String(value)}.`,
      "invalid_soft_token_count"
    );
  }
}

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new GemmaMediaError(`${name} must be a positive safe integer.`, "invalid_limit");
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function estimateDataUrlBytes(url: string): number {
  const comma = url.indexOf(",");
  if (comma < 0) return url.length;
  const header = url.slice(0, comma).toLowerCase();
  const payload = url.length - comma - 1;
  // Base64 decodes to at most 3/4 of its textual payload. Percent-encoded
  // data is conservatively bounded by its source character count.
  return header.includes(";base64") ? Math.ceil((payload * 3) / 4) : payload;
}

function sourceLabel(url: string): string {
  try {
    const parsed = new URL(url);
    if (parsed.protocol === DATA_URL_PROTOCOL) return "a data URL";
    if (parsed.protocol === "blob:") return "a blob URL";
    return `${parsed.origin}${parsed.pathname}`;
  } catch {
    return "the supplied URL";
  }
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return "an unknown size";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function formatNumber(value: number): string {
  return Number.isFinite(value) ? Math.round(value).toLocaleString("en-US") : "an invalid number";
}
