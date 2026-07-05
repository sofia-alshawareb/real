import { describe, expect, it } from 'vitest';
import { calcMetrics } from './metricsCalc';

describe('calcMetrics', () => {
  it('expresses matrix as all pixels without talc or sulfide labels', () => {
    const data = new Uint8Array([0, 1, 2, 3, 4, 0, 0, 4, 1, 2]);
    const m = calcMetrics({ width: 10, height: 1, data });

    expect(m.coarsePixels).toBe(2);
    expect(m.finePixels).toBe(2);
    expect(m.matrixPixels).toBe(5);
    expect(m.talcFraction + m.coarseFraction + m.fineFraction + m.matrixFraction).toBeCloseTo(1, 10);
  });

  it('fractions sum to 1 over the full frame', () => {
    const data = new Uint8Array([0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4]);
    const m = calcMetrics({ width: 11, height: 1, data });
    const sum = m.talcFraction + m.coarseFraction + m.fineFraction + m.matrixFraction;

    expect(sum).toBeCloseTo(1, 10);
    expect(m.talcFraction).toBeCloseTo(1 / 11, 10);
    expect(m.coarseFraction).toBeCloseTo(2 / 11, 10);
    expect(m.fineFraction).toBeCloseTo(3 / 11, 10);
    expect(m.matrixFraction).toBeCloseTo(5 / 11, 10); // bg(1) + matrix paint(4)
    expect(m.sulfideFraction).toBeCloseTo(m.coarseFraction + m.fineFraction, 10);
  });

  it('treats unmasked background as matrix', () => {
    const data = new Uint8Array([0, 0, 0, 1, 2]);
    const m = calcMetrics({ width: 5, height: 1, data });

    expect(m.matrixFraction).toBeCloseTo(3 / 5, 10);
    expect(m.matrixPixels).toBe(3);
    expect(m.classifiedShare).toBeCloseTo(2 / 5, 10);
  });
});
