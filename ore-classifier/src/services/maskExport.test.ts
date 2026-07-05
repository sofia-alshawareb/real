import { describe, expect, it } from 'vitest';
import { userDrawnMaskHasInk } from './maskExport';

describe('userDrawnMaskHasInk', () => {
  it('returns false for empty mask', () => {
    expect(userDrawnMaskHasInk(new Uint8Array(100))).toBe(false);
  });

  it('returns true when any pixel is painted', () => {
    const data = new Uint8Array(100);
    data[42] = 3;
    expect(userDrawnMaskHasInk(data)).toBe(true);
  });
});
