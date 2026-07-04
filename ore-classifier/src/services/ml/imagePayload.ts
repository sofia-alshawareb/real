import { getSourceImage, getTile } from '../../db/imageRepo';
import { TILE_SIZE } from '../tiling/tileImporter';
import type { Frame } from '../../types/models';

async function stitchTilesToBlob(imageId: string, width: number, height: number): Promise<Blob> {
  const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const numX = Math.ceil(width / TILE_SIZE);
  const numY = Math.ceil(height / TILE_SIZE);

  for (let ty = 0; ty < numY; ty++) {
    for (let tx = 0; tx < numX; tx++) {
      const record = await getTile(imageId, maxLevel, tx, ty);
      if (!record) {
        throw new Error(`Тайл не найден: ${imageId} ${maxLevel}/${tx}_${ty}`);
      }
      const bitmap = await createImageBitmap(record.blob);
      const px = tx * TILE_SIZE;
      const py = ty * TILE_SIZE;
      ctx.drawImage(bitmap, px, py);
      bitmap.close();
    }
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Не удалось собрать изображение'))), 'image/png');
  });
}

/** Resolve uploadable image bytes for a dexie-backed frame. */
export async function getFrameImageBlob(frame: Frame): Promise<Blob> {
  if (frame.source.kind !== 'dexie') {
    throw new Error('Процедурный кадр не может быть отправлен на сервер ML');
  }
  const stored = await getSourceImage(frame.source.imageId);
  if (stored?.blob) {
    return stored.blob;
  }
  return stitchTilesToBlob(frame.source.imageId, frame.width, frame.height);
}
