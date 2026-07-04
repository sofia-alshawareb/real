import { MASK_CLASSES, type MaskClassKey } from '../theme/palette';
import { upscaleMaskNearest } from './ml/maskUtils';

const MASK_CLASS_KEYS = Object.keys(MASK_CLASSES) as MaskClassKey[];

function parseHexColor(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

const UI_CLASS_RGB = (() => {
  const table = new Uint8Array(5 * 3);
  for (const key of MASK_CLASS_KEYS) {
    const meta = MASK_CLASSES[key];
    const [r, g, b] = parseHexColor(meta.color);
    const o = meta.value * 3;
    table[o] = r;
    table[o + 1] = g;
    table[o + 2] = b;
  }
  return table;
})();

/** Colored PNG of UI class labels (0 = black background). */
export async function encodeUiMaskColoredPngBlob(
  labels: Uint8Array,
  width: number,
  height: number,
): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < labels.length; i++) {
    const v = Math.min(4, Math.max(0, labels[i] ?? 0));
    const o = v * 3;
    const p = i * 4;
    rgba[p] = UI_CLASS_RGB[o];
    rgba[p + 1] = UI_CLASS_RGB[o + 1];
    rgba[p + 2] = UI_CLASS_RGB[o + 2];
    rgba[p + 3] = 255;
  }
  canvas.getContext('2d')!.putImageData(new ImageData(rgba, width, height), 0, 0);
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encode failed'))), 'image/png');
  });
}

/** Grayscale PNG: class index 0–4 mapped to 0, 64, 128, 192, 255. */
export async function encodeUiMaskGrayscalePngBlob(
  labels: Uint8Array,
  width: number,
  height: number,
): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < labels.length; i++) {
    const v = Math.min(4, Math.max(0, labels[i] ?? 0));
    const gray = v * 64;
    const p = i * 4;
    rgba[p] = gray;
    rgba[p + 1] = gray;
    rgba[p + 2] = gray;
    rgba[p + 3] = 255;
  }
  canvas.getContext('2d')!.putImageData(new ImageData(rgba, width, height), 0, 0);
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encode failed'))), 'image/png');
  });
}

/** Index PNG (mode L): pixel value equals UI class index 0–4. */
export async function encodeUiMaskIndexPngBlob(
  labels: Uint8Array,
  width: number,
  height: number,
): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < labels.length; i++) {
    const v = Math.min(4, Math.max(0, labels[i] ?? 0));
    const p = i * 4;
    rgba[p] = v;
    rgba[p + 1] = v;
    rgba[p + 2] = v;
    rgba[p + 3] = 255;
  }
  canvas.getContext('2d')!.putImageData(new ImageData(rgba, width, height), 0, 0);
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encode failed'))), 'image/png');
  });
}

export async function encodeUiMaskIndexPngBase64(
  labels: Uint8Array,
  width: number,
  height: number,
): Promise<string> {
  const blob = await encodeUiMaskIndexPngBlob(labels, width, height);
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!);
  return btoa(binary);
}

export function triggerBrowserDownload(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function downloadUserDrawnMask(params: {
  labels: Uint8Array;
  workingWidth: number;
  workingHeight: number;
  nativeWidth: number;
  nativeHeight: number;
  frameName: string;
  format: 'colored' | 'grayscale';
}): Promise<void> {
  const native =
    params.workingWidth === params.nativeWidth && params.workingHeight === params.nativeHeight
      ? params.labels
      : upscaleMaskNearest(
          params.labels,
          params.workingWidth,
          params.workingHeight,
          params.nativeWidth,
          params.nativeHeight,
        );

  const suffix = params.format === 'colored' ? 'manual_colored' : 'manual_grayscale';
  const safeName = params.frameName.replace(/[^\w\-]+/g, '_').slice(0, 80) || 'frame';
  const fileName = `${safeName}_${suffix}.png`;

  const blob =
    params.format === 'colored'
      ? await encodeUiMaskColoredPngBlob(native, params.nativeWidth, params.nativeHeight)
      : await encodeUiMaskGrayscalePngBlob(native, params.nativeWidth, params.nativeHeight);

  triggerBrowserDownload(blob, fileName);
}

export function userDrawnMaskHasInk(labels: Uint8Array): boolean {
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== 0) return true;
  }
  return false;
}
