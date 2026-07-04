// Общая "геология" кадра: точечная классификация пикселя в один из классов маски.
// Используется и процедурным источником тайлов (текстура), и мок-ML (маска) —
// поэтому сегментация всегда идеально ложится на изображение демо-кадров.

import { createNoise2D, type NoiseFunction2D } from 'simplex-noise';
import { mulberry32 } from './rng';

export type GrainClass = 1 | 2 | 3 | 4; // 1=обычные срастания, 2=тонкие срастания, 3=тальк, 4=нерудная матрица

export interface GeologyParams {
  /** Целевая доля площади под тальком, 0..~0.35 */
  talcShare: number;
  /** Доля тонких (мелких) сульфидных зёрен среди всех, 0..1 */
  fineBias: number;
  /** Целевая плотность сульфидных зёрен, 0.1..1 */
  sulfideDensity: number;
}

export const DEFAULT_GEOLOGY: GeologyParams = {
  talcShare: 0.04,
  fineBias: 0.35,
  sulfideDensity: 0.6,
};

/** Единая детерминированная привязка seed -> параметры геологии (без отдельного хранения). */
export function paramsForSeed(seed: number): GeologyParams {
  const rnd = mulberry32(seed);
  return {
    talcShare: rnd() * 0.24,
    fineBias: rnd(),
    sulfideDensity: 0.3 + rnd() * 0.6,
  };
}

interface Grain {
  x: number;
  y: number;
  radius: number;
  angleSeed: number;
  coarse: boolean;
}

const CELL_SIZE = 0.09;
const SEARCH_RADIUS = 1;

function cellHash(seed: number, cx: number, cy: number): number {
  let h = seed >>> 0;
  h = Math.imul(h ^ (cx + 0x9e3779b9), 0x27d4eb2d);
  h = Math.imul(h ^ (cy + 0x85ebca6b), 0x165667b1);
  h ^= h >>> 15;
  return h >>> 0;
}

function grainsForCell(seed: number, cx: number, cy: number, params: GeologyParams): Grain[] {
  const rnd = mulberry32(cellHash(seed, cx, cy));
  const baseCount = 1.5 + params.sulfideDensity * 5;
  const count = Math.max(0, Math.round(baseCount + (rnd() - 0.5) * 2));
  const grains: Grain[] = [];
  for (let i = 0; i < count; i++) {
    const isFine = rnd() < params.fineBias;
    const radius = isFine ? 0.005 + rnd() * 0.009 : 0.026 + rnd() * 0.039;
    grains.push({
      x: (cx + rnd()) * CELL_SIZE,
      y: (cy + rnd()) * CELL_SIZE,
      radius,
      angleSeed: rnd() * 1000,
      coarse: !isFine,
    });
  }
  return grains;
}

const cellCache = new Map<string, Grain[]>();
function getCellGrains(seed: number, cx: number, cy: number, params: GeologyParams): Grain[] {
  const key = `${seed}|${cx}|${cy}|${params.sulfideDensity.toFixed(2)}|${params.fineBias.toFixed(2)}`;
  let g = cellCache.get(key);
  if (!g) {
    g = grainsForCell(seed, cx, cy, params);
    cellCache.set(key, g);
  }
  return g;
}

const noiseCache = new Map<number, NoiseFunction2D>();
function getNoise(seed: number): NoiseFunction2D {
  let n = noiseCache.get(seed);
  if (!n) {
    n = createNoise2D(mulberry32(seed));
    noiseCache.set(seed, n);
  }
  return n;
}

function grainBoundaryRadius(noise: NoiseFunction2D, grain: Grain, angle: number): number {
  const n = noise(Math.cos(angle) * 2.2 + grain.angleSeed, Math.sin(angle) * 2.2 + grain.angleSeed);
  return grain.radius * (1 + 0.28 * n);
}

export interface PixelClassification {
  cls: GrainClass;
}

/** xNorm, yNorm — координаты в диапазоне [0,1) относительно короткой стороны кадра. */
export function classifyPoint(
  seed: number,
  xNorm: number,
  yNorm: number,
  params: GeologyParams = DEFAULT_GEOLOGY,
): PixelClassification {
  const boundaryNoise = getNoise(seed + 1);
  const cx = Math.floor(xNorm / CELL_SIZE);
  const cy = Math.floor(yNorm / CELL_SIZE);
  for (let dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++) {
    for (let dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy++) {
      const grains = getCellGrains(seed, cx + dx, cy + dy, params);
      for (const grain of grains) {
        const ddx = xNorm - grain.x;
        const ddy = yNorm - grain.y;
        const dist = Math.hypot(ddx, ddy);
        if (dist > grain.radius * 1.45) continue;
        const angle = Math.atan2(ddy, ddx);
        const r = grainBoundaryRadius(boundaryNoise, grain, angle);
        if (dist <= r) return { cls: grain.coarse ? 1 : 2 };
      }
    }
  }
  const talcNoise = getNoise(seed + 2);
  const nv = (talcNoise(xNorm * 7, yNorm * 7) + 1) / 2;
  const threshold = 1 - params.talcShare * 2.4;
  if (nv > threshold) return { cls: 3 };
  return { cls: 4 };
}

/** Базовый цвет фона-матрицы шлифа (без учёта зёрен) — для процедурной текстуры. */
export function backgroundRockColor(seed: number, xNorm: number, yNorm: number): [number, number, number] {
  const n = getNoise(seed + 3);
  const v = (n(xNorm * 14, yNorm * 14) + 1) / 2;
  const base = 168 + v * 24;
  return [base * 0.86, base * 0.85, base * 0.88];
}

export function classColor(cls: GrainClass): [number, number, number] {
  switch (cls) {
    case 1:
      return [46, 125, 50]; // обычные срастания — зелёный
    case 2:
      return [198, 40, 40]; // тонкие срастания — красный
    case 3:
      return [21, 101, 192]; // тальк — синий
    default:
      return [158, 158, 158]; // нерудная матрица — серый
  }
}

/** Рабочее разрешение маски: длинная сторона ограничена ради производительности браузера. */
export const MASK_LONG_EDGE = 1536;

export function maskWorkingSize(nativeWidth: number, nativeHeight: number): { mw: number; mh: number; scale: number } {
  const longSide = Math.max(nativeWidth, nativeHeight);
  if (longSide <= MASK_LONG_EDGE) {
    return { mw: nativeWidth, mh: nativeHeight, scale: 1 };
  }
  const scale = longSide / MASK_LONG_EDGE;
  const mw = Math.max(1, Math.round(nativeWidth / scale));
  const mh = Math.max(1, Math.round(nativeHeight / scale));
  return { mw, mh, scale };
}

/**
 * Синхронная генерация индексированной маски working-разрешения.
 * Для больших масок предпочитайте generateMaskDataChunked, чтобы не блокировать UI.
 */
export function generateMaskData(
  seed: number,
  width: number,
  height: number,
  params: GeologyParams = DEFAULT_GEOLOGY,
): Uint8Array {
  const longSide = Math.max(width, height);
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const yNorm = y / longSide;
    for (let x = 0; x < width; x++) {
      const xNorm = x / longSide;
      data[y * width + x] = classifyPoint(seed, xNorm, yNorm, params).cls;
    }
  }
  return data;
}

/** Построчная асинхронная генерация — не блокирует основной поток надолго. */
export async function generateMaskDataChunked(
  seed: number,
  width: number,
  height: number,
  params: GeologyParams = DEFAULT_GEOLOGY,
  onProgress?: (share: number) => void,
): Promise<Uint8Array> {
  const longSide = Math.max(width, height);
  const data = new Uint8Array(width * height);
  const rowsPerChunk = Math.max(1, Math.floor(40000 / width));
  for (let yStart = 0; yStart < height; yStart += rowsPerChunk) {
    const yEnd = Math.min(height, yStart + rowsPerChunk);
    for (let y = yStart; y < yEnd; y++) {
      const yNorm = y / longSide;
      const rowOffset = y * width;
      for (let x = 0; x < width; x++) {
        data[rowOffset + x] = classifyPoint(seed, x / longSide, yNorm, params).cls;
      }
    }
    onProgress?.(yEnd / height);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return data;
}

/**
 * Рендер прямоугольной текстурной плитки шлифа (для процедурного deep-zoom).
 * originX/originY/regionW/regionH — регион в нормализованных координатах OSD (x,y,width,height из tile.bounds).
 * tilePxW/tilePxH — фактический размер плитки в пикселях (из tile.sourceBounds).
 */
export function renderTextureTile(
  ctx: CanvasRenderingContext2D,
  seed: number,
  originX: number,
  originY: number,
  regionW: number,
  regionH: number,
  tilePxW: number,
  tilePxH: number,
  params: GeologyParams = DEFAULT_GEOLOGY,
): void {
  const sampleW = Math.max(8, Math.min(128, Math.round(tilePxW / 4)));
  const sampleH = Math.max(8, Math.min(128, Math.round(tilePxH / 4)));
  const sampleCanvas = document.createElement('canvas');
  sampleCanvas.width = sampleW;
  sampleCanvas.height = sampleH;
  const sampleCtx = sampleCanvas.getContext('2d')!;
  const imgData = sampleCtx.createImageData(sampleW, sampleH);
  const stepX = regionW / sampleW;
  const stepY = regionH / sampleH;
  for (let j = 0; j < sampleH; j++) {
    const yNorm = originY + j * stepY;
    for (let i = 0; i < sampleW; i++) {
      const xNorm = originX + i * stepX;
      const { cls } = classifyPoint(seed, xNorm, yNorm, params);
      const bg = backgroundRockColor(seed, xNorm, yNorm);
      let color: [number, number, number];
      if (cls === 4) {
        color = bg;
      } else {
        const [cr, cg, cb] = classColor(cls);
        const mix = cls === 3 ? 0.5 : 0.8;
        color = [
          bg[0] * (1 - mix) + cr * mix,
          bg[1] * (1 - mix) + cg * mix,
          bg[2] * (1 - mix) + cb * mix,
        ];
      }
      const idx = (j * sampleW + i) * 4;
      imgData.data[idx] = color[0];
      imgData.data[idx + 1] = color[1];
      imgData.data[idx + 2] = color[2];
      imgData.data[idx + 3] = 255;
    }
  }
  sampleCtx.putImageData(imgData, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(sampleCanvas, 0, 0, sampleW, sampleH, 0, 0, tilePxW, tilePxH);
}
