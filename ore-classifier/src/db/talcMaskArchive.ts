import { db, type TalcDrawnMaskRecord } from './db';
import { encodeBinaryHintMaskPngBlob, upscaleMaskNearest } from '../services/ml/maskUtils';
import { genId } from '../stores/experimentsStore';

export interface SaveTalcStrokeParams {
  experimentId: string;
  frameId: string;
  frameName: string;
  hintWorking: Uint8Array;
  workingWidth: number;
  workingHeight: number;
  nativeWidth: number;
  nativeHeight: number;
}

/** Persist one user talc stroke as a grayscale PNG (white = drawn talc in this stroke). */
export async function saveUserTalcStroke(params: SaveTalcStrokeParams): Promise<TalcDrawnMaskRecord> {
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
    throw new Error('empty talc stroke');
  }

  const hintNative = upscaleMaskNearest(
    hintWorking,
    workingWidth,
    workingHeight,
    nativeWidth,
    nativeHeight,
  );

  const existing = await db.talcDrawnMasks.where('frameId').equals(frameId).count();
  const strokeIndex = existing + 1;
  const blob = await encodeBinaryHintMaskPngBlob(hintNative, nativeWidth, nativeHeight);
  const id = genId('talcMask');
  const folderKey = frameId;
  const fileName = `${String(strokeIndex).padStart(4, '0')}_talc.png`;

  const record: TalcDrawnMaskRecord = {
    id,
    experimentId,
    frameId,
    frameName,
    folderKey,
    fileName,
    strokeIndex,
    nativeWidth,
    nativeHeight,
    blob,
    createdAt: Date.now(),
  };

  await db.talcDrawnMasks.put(record);
  return record;
}

export async function listTalcDrawnMasksForFrame(frameId: string): Promise<TalcDrawnMaskRecord[]> {
  return db.talcDrawnMasks.where('frameId').equals(frameId).sortBy('strokeIndex');
}

export async function listTalcDrawnMaskFolders(): Promise<string[]> {
  const rows = await db.talcDrawnMasks.toArray();
  return [...new Set(rows.map((r) => r.folderKey))];
}
