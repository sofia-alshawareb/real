import { db, type PositivePromptMaskRecord } from './db';
import { encodeBinaryHintMaskPngBlob, upscaleMaskNearest } from '../services/ml/maskUtils';
import { genId } from '../stores/experimentsStore';

export const POSITIVE_PROMPTS_SUBFOLDER = 'positive_prompts';

export interface SavePositivePromptStrokeParams {
  experimentId: string;
  frameId: string;
  frameName: string;
  hintWorking: Uint8Array;
  workingWidth: number;
  workingHeight: number;
  nativeWidth: number;
  nativeHeight: number;
}

/** Persist one user coarse (обычные срастания) stroke under {frameId}/positive_prompts/. */
export async function saveUserPositivePromptStroke(
  params: SavePositivePromptStrokeParams,
): Promise<PositivePromptMaskRecord> {
  const {
    experimentId,
    frameId,
    frameName,
    hintWorking,
    workingWidth,
    workingHeight,
    nativeWidth,
    nativeHeight,
  } = params;

  if (!hintWorking.some((v) => v > 0)) {
    throw new Error('empty positive prompt stroke');
  }

  const hintNative = upscaleMaskNearest(
    hintWorking,
    workingWidth,
    workingHeight,
    nativeWidth,
    nativeHeight,
  );

  const existing = await db.positivePromptMasks.where('frameId').equals(frameId).count();
  const strokeIndex = existing + 1;
  const blob = await encodeBinaryHintMaskPngBlob(hintNative, nativeWidth, nativeHeight);
  const id = genId('posPrompt');
  const folderKey = `${frameId}/${POSITIVE_PROMPTS_SUBFOLDER}`;
  const fileName = `${String(strokeIndex).padStart(4, '0')}_coarse.png`;

  const record: PositivePromptMaskRecord = {
    id,
    experimentId,
    frameId,
    frameName,
    folderKey,
    subfolder: POSITIVE_PROMPTS_SUBFOLDER,
    fileName,
    strokeIndex,
    nativeWidth,
    nativeHeight,
    blob,
    createdAt: Date.now(),
  };

  await db.positivePromptMasks.put(record);
  return record;
}

export async function listPositivePromptMasksForFrame(
  frameId: string,
): Promise<PositivePromptMaskRecord[]> {
  return db.positivePromptMasks.where('frameId').equals(frameId).sortBy('strokeIndex');
}
