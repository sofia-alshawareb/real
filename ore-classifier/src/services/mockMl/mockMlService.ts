// Имитация ML-сервиса сегментации: задержка, сбои, генерация маски по общей grain-модели.

import { generateMaskDataChunked, maskWorkingSize, paramsForSeed } from '../grainModel';
import { hashStringToSeed, mulberry32 } from '../rng';
import type { Frame } from '../../types/models';
import { useSettingsStore } from '../../stores/settingsStore';

export class MlUnavailableError extends Error {
  constructor(message = 'Сервис анализа временно недоступен') {
    super(message);
    this.name = 'MlUnavailableError';
  }
}

export interface SegmentationResult {
  data: Uint8Array;
  mw: number;
  mh: number;
  maskToNativeScale: number;
  confidence: number;
}

function maskSeedForFrame(frame: Frame): number {
  return frame.source.kind === 'procedural' ? frame.source.seed : hashStringToSeed(frame.id);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function segmentFrame(
  frame: Frame,
  onProgress?: (share: number) => void,
): Promise<SegmentationResult> {
  const settings = useSettingsStore.getState();
  if (settings.mlOffline) {
    throw new MlUnavailableError();
  }

  await delay(1000 + Math.random() * 3000);

  // повторная проверка на случай переключения тумблера во время "обработки"
  if (useSettingsStore.getState().mlOffline) {
    throw new MlUnavailableError();
  }
  if (Math.random() < useSettingsStore.getState().mlFailureRate) {
    throw new MlUnavailableError('Сбой сервиса анализа. Попробуйте повторить обработку.');
  }

  const seed = maskSeedForFrame(frame);
  const params = paramsForSeed(seed);
  const { mw, mh, scale } = maskWorkingSize(frame.width, frame.height);
  const data = await generateMaskDataChunked(seed, mw, mh, params, onProgress);
  const confidenceRnd = mulberry32(seed + 999)();
  const confidence = Math.round((0.7 + confidenceRnd * 0.28) * 100) / 100;

  return { data, mw, mh, maskToNativeScale: scale, confidence };
}
