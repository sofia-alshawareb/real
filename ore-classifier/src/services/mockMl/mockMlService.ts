// Локальная имитация ML-сегментации (демо-режим при включённом переключателе mlOffline).

import { generateMaskDataChunked, maskWorkingSize, paramsForSeed } from '../grainModel';
import { hashStringToSeed, mulberry32 } from '../rng';
import type { Frame } from '../../types/models';
import { useSettingsStore } from '../../stores/settingsStore';
import { MlUnavailableError } from '../ml/errors';
import type { SegmentationResult } from '../ml/types';

export { MlUnavailableError } from '../ml/errors';
export type { SegmentationResult } from '../ml/types';

function maskSeedForFrame(frame: Frame): number {
  return frame.source.kind === 'procedural' ? frame.source.seed : hashStringToSeed(frame.id);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function segmentFrameMock(
  frame: Frame,
  onProgress?: (share: number) => void,
): Promise<SegmentationResult> {
  await delay(1000 + Math.random() * 3000);

  if (Math.random() < useSettingsStore.getState().mlFailureRate) {
    throw new MlUnavailableError('Сбой демо-имитации анализа. Попробуйте повторить обработку.');
  }

  const seed = maskSeedForFrame(frame);
  const params = paramsForSeed(seed);
  const { mw, mh, scale } = maskWorkingSize(frame.width, frame.height);
  const data = await generateMaskDataChunked(seed, mw, mh, params, onProgress);
  const confidenceRnd = mulberry32(seed + 999)();
  const confidence = Math.round((0.7 + confidenceRnd * 0.28) * 100) / 100;

  return { data, mw, mh, maskToNativeScale: scale, confidence };
}

/** @deprecated use segmentFrameMock */
export const segmentFrame = segmentFrameMock;
