// Расчёт метрик по индексированной маске: прямой подсчёт площади каждого класса разметки.
// Классы маски: 1=обычные срастания, 2=тонкие срастания, 3=тальк, 4=нерудная матрица.

import type { FrameMetrics } from '../types/models';

export interface MaskBuffer {
  width: number;
  height: number;
  data: Uint8Array; // 0=фон(не классифицировано), 1=обычные, 2=тонкие, 3=тальк, 4=матрица
}

export function calcMetrics(mask: MaskBuffer): FrameMetrics {
  const total = mask.data.length;
  let coarse = 0;
  let fine = 0;
  let talc = 0;
  let matrix = 0;
  let classified = 0;
  for (let i = 0; i < total; i++) {
    const v = mask.data[i];
    if (v === 1) coarse++;
    else if (v === 2) fine++;
    else if (v === 3) talc++;
    else if (v === 4) matrix++;
    if (v !== 0) classified++;
  }
  const denom = classified || 1;
  const talcFraction = talc / denom;
  const matrixFraction = matrix / denom;
  const coarseFraction = coarse / denom;
  const fineFraction = fine / denom;
  const sulfideFraction = coarseFraction + fineFraction;

  return {
    talcFraction,
    sulfideFraction,
    matrixFraction,
    coarseFraction,
    fineFraction,
    classifiedShare: classified / total,
  };
}
