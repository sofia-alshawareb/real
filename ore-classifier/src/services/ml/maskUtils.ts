import { maskWorkingSize } from '../grainModel';
import { BACKEND_PALETTE_RGB, rgbToBackendClass, remapBackendMaskToUi } from './classMap';

/** Decode paletted PNG ImageData into backend class indices (0–3). */
export function decodePalettedMaskImageData(data: ImageData): Uint8Array {
  const { width, height, data: rgba } = data;
  const labels = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < labels.length; i++, p += 4) {
    const bc = rgbToBackendClass(rgba[p]!, rgba[p + 1]!, rgba[p + 2]!);
    labels[i] = bc ?? 0;
  }
  return labels;
}

/** Load base64 PNG into backend class labels at native mask resolution. */
export async function decodePalettedPngBase64(base64: string): Promise<{
  labels: Uint8Array;
  width: number;
  height: number;
}> {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: 'image/png' });

  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const backendLabels = decodePalettedMaskImageData(imageData);
  return { labels: backendLabels, width: canvas.width, height: canvas.height };
}

/** Nearest-neighbor upscale of class index mask. */
export function upscaleMaskNearest(
  labels: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Uint8Array {
  if (srcW === dstW && srcH === dstH) return labels;
  return downscaleMaskNearest(labels, srcW, srcH, dstW, dstH);
}

/** Encode binary hint mask (0/1) as grayscale PNG blob. */
export async function encodeBinaryHintMaskPngBlob(
  hint: Uint8Array,
  width: number,
  height: number,
): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < hint.length; i++) {
    const v = hint[i]! ? 255 : 0;
    const p = i * 4;
    rgba[p] = v;
    rgba[p + 1] = v;
    rgba[p + 2] = v;
    rgba[p + 3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encode failed'))), 'image/png');
  });
}

/** Encode binary hint mask (0/1) as base64 grayscale PNG for the refine API. */
export async function encodeBinaryHintMaskPngBase64(
  hint: Uint8Array,
  width: number,
  height: number,
): Promise<string> {
  const blob = await encodeBinaryHintMaskPngBlob(hint, width, height);
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!);
  return btoa(binary);
}

/** Merge backend talc (UI class 3) globally; clear old talc where backend is not talc. */
export function mergeGlobalTalcLayer(
  local: Uint8Array,
  backendUi: Uint8Array,
  talcClassValue = 3,
): void {
  if (local.length !== backendUi.length) {
    throw new Error('mergeGlobalTalcLayer: buffer size mismatch');
  }
  for (let i = 0; i < local.length; i++) {
    if (backendUi[i] === talcClassValue) {
      local[i] = talcClassValue;
    } else if (local[i] === talcClassValue) {
      local[i] = 0;
    }
  }
}

/** Nearest-neighbor downscale of class index mask. */
export function downscaleMaskNearest(
  labels: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Uint8Array {
  if (srcW === dstW && srcH === dstH) return labels;
  const out = new Uint8Array(dstW * dstH);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.floor(y * scaleY));
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.floor(x * scaleX));
      out[y * dstW + x] = labels[sy * srcW + sx]!;
    }
  }
  return out;
}

export function prepareMaskForFrame(
  backendLabelsNative: Uint8Array,
  maskNativeW: number,
  maskNativeH: number,
  frameNativeW: number,
  frameNativeH: number,
): { data: Uint8Array; mw: number; mh: number; maskToNativeScale: number } {
  const uiNative = remapBackendMaskToUi(backendLabelsNative);
  const { mw, mh, scale } = maskWorkingSize(frameNativeW, frameNativeH);
  const data =
    maskNativeW === mw && maskNativeH === mh
      ? uiNative
      : downscaleMaskNearest(uiNative, maskNativeW, maskNativeH, mw, mh);
  return { data, mw, mh, maskToNativeScale: scale };
}

/** For tests: build RGBA buffer mimicking backend palette PNG. */
export function backendLabelsToRgba(labels: Uint8Array, width: number, height: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < labels.length; i++) {
    const bc = Math.min(3, Math.max(0, labels[i]!));
    const [r, g, b] = BACKEND_PALETTE_RGB[bc]!;
    const p = i * 4;
    rgba[p] = r;
    rgba[p + 1] = g;
    rgba[p + 2] = b;
    rgba[p + 3] = 255;
  }
  return rgba;
}
