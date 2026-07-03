// Расчёт метрик по индексированной маске: доли классов + coarse/fine через связные компоненты.

import { COARSE_GRAIN_UM } from '../theme/palette';
import type { FrameMetrics } from '../types/models';

export interface MaskBuffer {
  width: number;
  height: number;
  data: Uint8Array; // 0=фон(не классифицировано), 1=сульфид, 2=gangue, 3=тальк
}

const DOWNSAMPLE_LIMIT = 2048;

function downsampleForComponents(mask: MaskBuffer): { data: Uint8Array; width: number; height: number; scale: number } {
  const longSide = Math.max(mask.width, mask.height);
  if (longSide <= DOWNSAMPLE_LIMIT) {
    return { data: mask.data, width: mask.width, height: mask.height, scale: 1 };
  }
  const scale = longSide / DOWNSAMPLE_LIMIT;
  const width = Math.max(1, Math.round(mask.width / scale));
  const height = Math.max(1, Math.round(mask.height / scale));
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const sy = Math.min(mask.height - 1, Math.round(y * scale));
    for (let x = 0; x < width; x++) {
      const sx = Math.min(mask.width - 1, Math.round(x * scale));
      data[y * width + x] = mask.data[sy * mask.width + sx];
    }
  }
  return { data, width, height, scale };
}

/** BFS по 4-соседям для класса "сульфид" (значение 1), возвращает площади компонент в пикселях уменьшенной сетки. */
function sulfideComponentAreas(data: Uint8Array, width: number, height: number): number[] {
  const visited = new Uint8Array(width * height);
  const areas: number[] = [];
  const stack: number[] = [];
  for (let start = 0; start < data.length; start++) {
    if (data[start] !== 1 || visited[start]) continue;
    let area = 0;
    stack.push(start);
    visited[start] = 1;
    while (stack.length) {
      const idx = stack.pop()!;
      area++;
      const x = idx % width;
      const y = (idx - x) / width;
      const neighbors = [
        x > 0 ? idx - 1 : -1,
        x < width - 1 ? idx + 1 : -1,
        y > 0 ? idx - width : -1,
        y < height - 1 ? idx + width : -1,
      ];
      for (const n of neighbors) {
        if (n >= 0 && !visited[n] && data[n] === 1) {
          visited[n] = 1;
          stack.push(n);
        }
      }
    }
    areas.push(area);
  }
  return areas;
}

export function calcMetrics(mask: MaskBuffer, pixelSizeUm: number, maskToNativeScale: number): FrameMetrics {
  const total = mask.data.length;
  let sulfide = 0;
  let gangue = 0;
  let talc = 0;
  let classified = 0;
  for (let i = 0; i < total; i++) {
    const v = mask.data[i];
    if (v === 1) sulfide++;
    else if (v === 2) gangue++;
    else if (v === 3) talc++;
    if (v !== 0) classified++;
  }
  const denom = classified || 1;
  const talcFraction = talc / denom;
  const sulfideFraction = sulfide / denom;
  const gangueFraction = gangue / denom;

  const { data, width, height, scale } = downsampleForComponents(mask);
  const areas = sulfideComponentAreas(data, width, height);
  // микроны на пиксель уменьшенной сетки маски = pixelSizeUm(native) * maskToNativeScale(mask->native) * scale(down)
  const umPerReducedPx = pixelSizeUm * maskToNativeScale * scale;
  let coarseArea = 0;
  let fineArea = 0;
  for (const area of areas) {
    const equivDiameterUm = 2 * Math.sqrt(area / Math.PI) * umPerReducedPx;
    if (equivDiameterUm >= COARSE_GRAIN_UM) coarseArea += area;
    else fineArea += area;
  }
  const sulfideAreaTotal = coarseArea + fineArea || 1;
  const coarseFraction = coarseArea / sulfideAreaTotal;
  const fineFraction = fineArea / sulfideAreaTotal;

  return {
    talcFraction,
    sulfideFraction,
    gangueFraction,
    coarseFraction,
    fineFraction,
    classifiedShare: classified / total,
  };
}
