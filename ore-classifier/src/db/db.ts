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

class OreDb extends Dexie {
  sourceImages!: Table<SourceImageRecord, string>;
  tiles!: Table<TileRecord, string>;
  masks!: Table<MaskRecord, string>;
  reportDrafts!: Table<ReportDraft, string>;

  constructor() {
    super('ore-classifier-db');
    this.version(1).stores({
      sourceImages: 'id',
      tiles: 'key',
      masks: 'id, frameId',
      reportDrafts: 'experimentId',
    });
  }
}

export const db = new OreDb();

export async function clearAllData(): Promise<void> {
  await db.transaction('rw', db.sourceImages, db.tiles, db.masks, db.reportDrafts, async () => {
    await db.sourceImages.clear();
    await db.tiles.clear();
    await db.masks.clear();
    await db.reportDrafts.clear();
  });
}
