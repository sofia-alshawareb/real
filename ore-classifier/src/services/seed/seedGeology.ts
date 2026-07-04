import { generateMaskData, paramsForSeed } from '../grainModel';
import { calcMetrics } from '../metricsCalc';
import { classifyFrame } from '../rulesEngine';
import type { FrameMetrics, OreClass } from '../../types/models';

// Небольшое тестовое разрешение достаточно для оценки долей классов при подборе seed:
// это лишь черновая проверка результата, реальная маска кадра генерируется отдельно
// в рабочем разрешении (см. generateAndStoreResult/generateMaskDataChunked).
const TEST_W = 96;
const TEST_H = 60;

/** Подбор seed, дающего нужный класс руды на тестовой мини-маске (для воспроизводимых демо-данных). */
export function findSeedForClass(
  targetClass: OreClass,
  talcThreshold: number,
  startSeed: number,
  maxAttempts = 400,
): number {
  for (let i = 0; i < maxAttempts; i++) {
    const seed = startSeed + i;
    const params = paramsForSeed(seed);
    const data = generateMaskData(seed, TEST_W, TEST_H, params);
    const metrics = calcMetrics({ width: TEST_W, height: TEST_H, data });
    const { oreClass } = classifyFrame(metrics, talcThreshold);
    if (oreClass === targetClass) return seed;
  }
  return startSeed;
}

export interface SeedMatch {
  seed: number;
  data: Uint8Array;
  width: number;
  height: number;
  metrics: FrameMetrics;
  oreClass: OreClass;
  reason: string;
}

/**
 * Как findSeedForClass, но дополнительно возвращает уже посчитанные на тестовом разрешении
 * маску/метрики/класс, чтобы использовать их как быстрый предварительный результат кадра
 * сразу при сидировании — не дожидаясь фоновой донагенерации полноразмерной маски.
 */
export function findSeedAndMetricsForClass(
  targetClass: OreClass,
  talcThreshold: number,
  startSeed: number,
  maxAttempts = 400,
): SeedMatch {
  let fallback: SeedMatch | null = null;
  for (let i = 0; i < maxAttempts; i++) {
    const seed = startSeed + i;
    const params = paramsForSeed(seed);
    const data = generateMaskData(seed, TEST_W, TEST_H, params);
    const metrics = calcMetrics({ width: TEST_W, height: TEST_H, data });
    const { oreClass, reason } = classifyFrame(metrics, talcThreshold);
    const match: SeedMatch = { seed, data, width: TEST_W, height: TEST_H, metrics, oreClass, reason };
    if (!fallback) fallback = match;
    if (oreClass === targetClass) return match;
  }
  return fallback as SeedMatch;
}
