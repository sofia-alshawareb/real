// Импорт загруженного файла в пирамиду тайлов Dexie.
// Ограничение прототипа: браузерный декодер не читает TIFF и очень большие PNG/JPEG —
// в проде эту роль берёт на себя серверный тайлинг.

import { genId } from '../../stores/experimentsStore';
import { putTile } from '../../db/imageRepo';

export const TILE_SIZE = 512;
const MAX_LONG_SIDE = 16384;
const SUPPORTED_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

export class UnsupportedImageError extends Error {}

export interface ImportedImage {
  imageId: string;
  width: number;
  height: number;
  maxLevel: number;
  downscaled: boolean;
}

export async function importImageFile(file: File, onProgress?: (share: number) => void): Promise<ImportedImage> {
  if (!SUPPORTED_TYPES.includes(file.type)) {
    throw new UnsupportedImageError(
      `Формат «${file.type || 'неизвестен'}» не поддерживается браузером. Используйте PNG, JPEG или WebP — TIFF браузер не декодирует, в проде тайлит сервер.`,
    );
  }

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new UnsupportedImageError(
      'Не удалось декодировать изображение — вероятно, оно повреждено или слишком велико для браузера.',
    );
  }

  let width = bitmap.width;
  let height = bitmap.height;
  let downscaled = false;
  let sourceBitmap: ImageBitmap | HTMLCanvasElement = bitmap;
  const longSide = Math.max(width, height);
  if (longSide > MAX_LONG_SIDE) {
    const scale = MAX_LONG_SIDE / longSide;
    const newW = Math.max(1, Math.round(width * scale));
    const newH = Math.max(1, Math.round(height * scale));
    const canvas = document.createElement('canvas');
    canvas.width = newW;
    canvas.height = newH;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(bitmap, 0, 0, newW, newH);
    sourceBitmap = canvas;
    width = newW;
    height = newH;
    downscaled = true;
  }

  const imageId = genId('img');
  const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));

  const levelDims: Array<{ level: number; w: number; h: number }> = [];
  let totalTiles = 0;
  for (let level = 0; level <= maxLevel; level++) {
    const scale = 1 / Math.pow(2, maxLevel - level);
    const w = Math.max(1, Math.round(width * scale));
    const h = Math.max(1, Math.round(height * scale));
    levelDims.push({ level, w, h });
    totalTiles += Math.ceil(w / TILE_SIZE) * Math.ceil(h / TILE_SIZE);
  }

  let processed = 0;
  for (const { level, w, h } of levelDims) {
    const levelCanvas = document.createElement('canvas');
    levelCanvas.width = w;
    levelCanvas.height = h;
    const levelCtx = levelCanvas.getContext('2d')!;
    levelCtx.imageSmoothingEnabled = true;
    levelCtx.imageSmoothingQuality = 'high';
    levelCtx.drawImage(sourceBitmap, 0, 0, w, h);

    const numX = Math.ceil(w / TILE_SIZE);
    const numY = Math.ceil(h / TILE_SIZE);
    for (let ty = 0; ty < numY; ty++) {
      for (let tx = 0; tx < numX; tx++) {
        const px = tx * TILE_SIZE;
        const py = ty * TILE_SIZE;
        const tw = Math.min(TILE_SIZE, w - px);
        const th = Math.min(TILE_SIZE, h - py);
        const tileCanvas = document.createElement('canvas');
        tileCanvas.width = tw;
        tileCanvas.height = th;
        const tileCtx = tileCanvas.getContext('2d')!;
        tileCtx.drawImage(levelCanvas, px, py, tw, th, 0, 0, tw, th);
        const blob = await new Promise<Blob>((resolve, reject) =>
          tileCanvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Не удалось сохранить тайл'))), 'image/webp', 0.85),
        );
        await putTile(imageId, level, tx, ty, blob);
        processed++;
        if (processed % 8 === 0) {
          onProgress?.(processed / totalTiles);
          await new Promise((r) => setTimeout(r, 0));
        }
      }
    }
  }
  onProgress?.(1);
  bitmap.close();
  return { imageId, width, height, maxLevel, downscaled };
}
