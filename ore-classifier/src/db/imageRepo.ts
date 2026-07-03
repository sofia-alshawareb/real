import { db } from './db';
import type { MaskRecord } from './db';

export async function saveSourceImage(id: string, blob: Blob, width: number, height: number) {
  await db.sourceImages.put({ id, blob, width, height });
}

export async function getSourceImage(id: string) {
  return db.sourceImages.get(id);
}

export function tileKey(imageId: string, level: number, x: number, y: number) {
  return `${imageId}/${level}/${x}_${y}`;
}

export async function putTile(imageId: string, level: number, x: number, y: number, blob: Blob) {
  await db.tiles.put({ key: tileKey(imageId, level, x, y), blob });
}

export async function getTile(imageId: string, level: number, x: number, y: number) {
  return db.tiles.get(tileKey(imageId, level, x, y));
}

export async function putMask(record: MaskRecord) {
  await db.masks.put(record);
}

export async function getMask(id: string) {
  return db.masks.get(id);
}

export async function getMaskByFrameId(frameId: string) {
  return db.masks.where('frameId').equals(frameId).first();
}

export async function deleteFrameData(frameId: string, imageId?: string) {
  await db.transaction('rw', db.masks, db.tiles, db.sourceImages, async () => {
    await db.masks.where('frameId').equals(frameId).delete();
    if (imageId) {
      const keys = await db.tiles.toCollection().primaryKeys();
      const toDelete = keys.filter((k) => typeof k === 'string' && k.startsWith(`${imageId}/`));
      if (toDelete.length) await db.tiles.bulkDelete(toDelete);
      await db.sourceImages.delete(imageId);
    }
  });
}
