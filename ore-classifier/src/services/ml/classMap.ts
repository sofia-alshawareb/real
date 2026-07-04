/**
 * Backend segmentation taxonomy — identity mapping with UI mask class indices.
 * Backend outputs: 0=background, 1=coarse, 2=fine, 3=talc, 4=matrix
 */

export type BackendClass = 0 | 1 | 2 | 3 | 4;

/** RGB palette colors from ml/lib/constants.py CLASS_COLORS */
export const BACKEND_PALETTE_RGB: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 0],
  [46, 125, 50],
  [198, 40, 40],
  [21, 101, 192],
  [158, 158, 158],
] as const;

/** backend class index → UI mask value (identity) */
export const BACKEND_TO_UI_CLASS: Record<BackendClass, number> = {
  0: 0,
  1: 1,
  2: 2,
  3: 3,
  4: 4,
};

export function rgbToBackendClass(r: number, g: number, b: number): BackendClass | null {
  for (let i = 0; i < BACKEND_PALETTE_RGB.length; i++) {
    const [pr, pg, pb] = BACKEND_PALETTE_RGB[i]!;
    if (r === pr && g === pg && b === pb) return i as BackendClass;
  }
  return null;
}

export function remapBackendMaskToUi(nativeLabels: Uint8Array): Uint8Array {
  const out = new Uint8Array(nativeLabels.length);
  for (let i = 0; i < nativeLabels.length; i++) {
    const bc = Math.min(4, Math.max(0, nativeLabels[i] ?? 0)) as BackendClass;
    out[i] = BACKEND_TO_UI_CLASS[bc] ?? 0;
  }
  return out;
}

export type CalibClassKey = 'coarse' | 'fine' | 'talc' | 'matrix';

export const UI_CLASS_TO_CALIB_KEY: Record<number, CalibClassKey> = {
  1: 'coarse',
  2: 'fine',
  3: 'talc',
  4: 'matrix',
};
