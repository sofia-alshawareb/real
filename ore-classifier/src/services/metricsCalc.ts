// Расчёт метрик по индексированной маске.
// Доли считаются от всей площади кадра (width × height).
// Матрица = все пиксели без разметки талька/срастаний (фон + явно помеченная матрица).

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
  let labeled = 0;
  for (let i = 0; i < total; i++) {
    const v = mask.data[i];
    if (v === 1) coarse++;
    else if (v === 2) fine++;
    else if (v === 3) talc++;
    if (v !== 0) labeled++;
  }
  const denom = total || 1;
  const matrix = total - coarse - fine - talc;
  const talcFraction = talc / denom;
  const coarseFraction = coarse / denom;
  const fineFraction = fine / denom;
  const matrixFraction = matrix / denom;
  const sulfideFraction = coarseFraction + fineFraction;

  return {
    talcFraction,
    sulfideFraction,
    matrixFraction,
    coarseFraction,
    fineFraction,
    coarsePixels: coarse,
    finePixels: fine,
    matrixPixels: matrix,
    classifiedShare: labeled / denom,
  };
}
