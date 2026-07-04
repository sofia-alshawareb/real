import Dexie, { type Table } from 'dexie';
import type { ReportDraft } from '../types/models';

export interface SourceImageRecord {
  id: string;
  blob: Blob;
  width: number;
  height: number;
}

export interface TileRecord {
  key: string; // `${imageId}/${level}/${x}_${y}`
  blob: Blob;
}

export interface MaskRecord {
  id: string;
  frameId: string;
  width: number;
  height: number;
  data: Uint8Array; // 0=фон, 1=сульфид, 2=gangue, 3=тальк
}

/** One user-drawn talc stroke saved as a separate PNG (grouped by frameId folder). */
export interface TalcDrawnMaskRecord {
  id: string;
  experimentId: string;
  frameId: string;
  frameName: string;
  folderKey: string;
  fileName: string;
  strokeIndex: number;
  nativeWidth: number;
  nativeHeight: number;
  blob: Blob;
  createdAt: number;
}

/** User-drawn coarse (обычные срастания) stroke under frameId/positive_prompts/. */
export interface PositivePromptMaskRecord {
  id: string;
  experimentId: string;
  frameId: string;
  frameName: string;
  folderKey: string;
  subfolder: string;
  fileName: string;
  strokeIndex: number;
  nativeWidth: number;
  nativeHeight: number;
  blob: Blob;
  createdAt: number;
}

class OreDb extends Dexie {
  sourceImages!: Table<SourceImageRecord, string>;
  tiles!: Table<TileRecord, string>;
  masks!: Table<MaskRecord, string>;
  talcDrawnMasks!: Table<TalcDrawnMaskRecord, string>;
  positivePromptMasks!: Table<PositivePromptMaskRecord, string>;
  reportDrafts!: Table<ReportDraft, string>;

  constructor() {
    super('ore-classifier-db');
    this.version(1).stores({
      sourceImages: 'id',
      tiles: 'key',
      masks: 'id, frameId',
      reportDrafts: 'experimentId',
    });
    this.version(2).stores({
      sourceImages: 'id',
      tiles: 'key',
      masks: 'id, frameId',
      talcDrawnMasks: 'id, frameId, folderKey, createdAt',
      reportDrafts: 'experimentId',
    });
    this.version(3).stores({
      sourceImages: 'id',
      tiles: 'key',
      masks: 'id, frameId',
      talcDrawnMasks: 'id, frameId, folderKey, createdAt',
      positivePromptMasks: 'id, frameId, folderKey, createdAt',
      reportDrafts: 'experimentId',
    });
  }
}

export const db = new OreDb();

export async function clearAllData(): Promise<void> {
  await db.transaction('rw', db.tables, async () => {
    await Promise.all(db.tables.map((table) => table.clear()));
  });
}
