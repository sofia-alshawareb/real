import { generateMaskData, paramsForSeed } from '../grainModel';
import { calcMetrics } from '../metricsCalc';
import { classifyFrame } from '../rulesEngine';
import type { OreClass } from '../../types/models';

const TEST_W = 256;
const TEST_H = 160;

/** Подбор seed, дающего нужный класс руды на тестовой мини-маске (для воспроизводимых демо-данных). */
export function findSeedForClass(
  targetClass: OreClass,
  talcThreshold: number,
  startSeed: number,
  maxAttempts = 1200,
): number {
  for (let i = 0; i < maxAttempts; i++) {
    const seed = startSeed + i;
    const params = paramsForSeed(seed);
    const data = generateMaskData(seed, TEST_W, TEST_H, params);
    const metrics = calcMetrics({ width: TEST_W, height: TEST_H, data }, 0.5, 1);
    const { oreClass } = classifyFrame(metrics, talcThreshold);
    if (oreClass === targetClass) return seed;
  }
  return startSeed;
}
