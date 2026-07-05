import { describe, expect, it } from 'vitest';
import { BACKEND_TO_UI_CLASS, remapBackendMaskToUi, rgbToBackendClass } from './classMap';
import { decodePalettedMaskImageData, downscaleMaskNearest, prepareMaskForFrame } from './maskUtils';

describe('classMap', () => {
  it('maps backend palette RGB to class indices', () => {
    expect(rgbToBackendClass(0, 0, 0)).toBe(0);
    expect(rgbToBackendClass(46, 125, 50)).toBe(1);
    expect(rgbToBackendClass(198, 40, 40)).toBe(2);
    expect(rgbToBackendClass(21, 101, 192)).toBe(3);
    expect(rgbToBackendClass(158, 158, 158)).toBe(4);
    expect(rgbToBackendClass(255, 255, 255)).toBeNull();
  });

  it('remaps backend labels to UI classes (identity)', () => {
    const src = new Uint8Array([0, 1, 2, 3, 4]);
    const out = remapBackendMaskToUi(src);
    expect(Array.from(out)).toEqual([
      BACKEND_TO_UI_CLASS[0],
      BACKEND_TO_UI_CLASS[1],
      BACKEND_TO_UI_CLASS[2],
      BACKEND_TO_UI_CLASS[3],
      BACKEND_TO_UI_CLASS[4],
    ]);
  });
});

describe('maskUtils', () => {
  it('downscales mask with nearest neighbor', () => {
    const src = new Uint8Array([1, 2, 3, 4]);
    const out = downscaleMaskNearest(src, 2, 2, 1, 1);
    expect(out[0]).toBe(1);
  });

  it('prepareMaskForFrame matches maskWorkingSize for large frames', () => {
    const labels = new Uint8Array(4000 * 3000).fill(1);
    const { mw, mh, maskToNativeScale, data } = prepareMaskForFrame(
      labels,
      4000,
      3000,
      4000,
      3000,
    );
    expect(mw).toBeLessThanOrEqual(1536);
    expect(mh).toBeLessThanOrEqual(1536);
    expect(maskToNativeScale).toBeGreaterThan(1);
    expect(data.length).toBe(mw * mh);
  });

  it('decodes paletted ImageData to backend classes', () => {
    const rgba = new Uint8ClampedArray([
      46, 125, 50, 255,
      21, 101, 192, 255,
    ]);
    const labels = decodePalettedMaskImageData({ width: 2, height: 1, data: rgba } as ImageData);
    expect(labels[0]).toBe(1);
    expect(labels[1]).toBe(3);
  });
});
