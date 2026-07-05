import { describe, expect, it } from 'vitest';
import { classifyFrame } from './rulesEngine';
import type { FrameMetrics } from '../types/models';

function metrics(partial: Partial<FrameMetrics> & Pick<FrameMetrics, 'talcFraction' | 'coarseFraction' | 'fineFraction'>): FrameMetrics {
  return {
    sulfideFraction: partial.coarseFraction + partial.fineFraction,
    matrixFraction: partial.matrixFraction ?? 0,
    classifiedShare: partial.classifiedShare ?? 1,
    coarsePixels: partial.coarsePixels,
    finePixels: partial.finePixels,
    ...partial,
  };
}

describe('classifyFrame', () => {
  it('assigns talc when talc fraction exceeds threshold', () => {
    const result = classifyFrame(metrics({ talcFraction: 0.15, coarseFraction: 0.4, fineFraction: 0.45 }), 0.1);
    expect(result.oreClass).toBe('talc');
  });

  it('assigns hard when fine pixel count exceeds coarse', () => {
    const result = classifyFrame(
      metrics({
        talcFraction: 0.05,
        coarseFraction: 0.2,
        fineFraction: 0.35,
        coarsePixels: 1200,
        finePixels: 2100,
      }),
      0.1,
    );
    expect(result.oreClass).toBe('hard');
    expect(result.reason).toContain('труднообогатимая');
    expect(result.reason).toContain('2');
  });

  it('assigns routine when coarse pixel count is not less than fine', () => {
    const result = classifyFrame(
      metrics({
        talcFraction: 0.05,
        coarseFraction: 0.35,
        fineFraction: 0.2,
        coarsePixels: 3500,
        finePixels: 2000,
      }),
      0.1,
    );
    expect(result.oreClass).toBe('routine');
    expect(result.reason).toContain('рядовая');
  });

  it('assigns routine when coarse and fine pixel counts are equal', () => {
    const result = classifyFrame(
      metrics({
        talcFraction: 0.05,
        coarseFraction: 0.25,
        fineFraction: 0.25,
        coarsePixels: 1500,
        finePixels: 1500,
      }),
      0.1,
    );
    expect(result.oreClass).toBe('routine');
  });

  it('falls back to fraction comparison for legacy metrics without pixel counts', () => {
    const result = classifyFrame(
      metrics({ talcFraction: 0.02, coarseFraction: 0.1, fineFraction: 0.4 }),
      0.1,
    );
    expect(result.oreClass).toBe('hard');
  });
});
