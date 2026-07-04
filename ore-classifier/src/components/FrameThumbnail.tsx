import { useEffect, useRef } from 'react';
import type { Frame } from '../types/models';
import { paramsForSeed, classifyPoint, backgroundRockColor, classColor } from '../services/grainModel';
import { getTile, getMask } from '../db/imageRepo';
import { TILE_SIZE } from '../services/tiling/tileImporter';

/**
 * Собирает миниатюру загруженного (dexie) кадра из тайлов подходящего уровня пирамиды.
 * Уровень 0 — самый мелкий (почти всё изображение сжато в один тайл), поэтому рисовать его
 * растянутым на всю миниатюру даёт почти однородную тёмную заливку ("чёрное" изображение).
 * Здесь выбирается минимальный уровень, чья длинная сторона не меньше стороны миниатюры.
 */
async function drawDexieOriginal(
  ctx: CanvasRenderingContext2D,
  imageId: string,
  frameWidth: number,
  frameHeight: number,
  width: number,
  height: number,
): Promise<void> {
  const maxLevel = Math.ceil(Math.log2(Math.max(frameWidth, frameHeight, 1)));
  const targetLongSide = Math.max(width, height);

  let renderLevel = maxLevel;
  for (let level = 0; level <= maxLevel; level++) {
    const scale = 1 / Math.pow(2, maxLevel - level);
    const longSide = Math.max(frameWidth, frameHeight) * scale;
    if (longSide >= targetLongSide) {
      renderLevel = level;
      break;
    }
  }

  const scale = 1 / Math.pow(2, maxLevel - renderLevel);
  const lw = Math.max(1, Math.round(frameWidth * scale));
  const lh = Math.max(1, Math.round(frameHeight * scale));
  const numX = Math.ceil(lw / TILE_SIZE);
  const numY = Math.ceil(lh / TILE_SIZE);
  const sx = width / lw;
  const sy = height / lh;

  const tileCoords: Array<{ tx: number; ty: number }> = [];
  for (let ty = 0; ty < numY; ty++) {
    for (let tx = 0; tx < numX; tx++) tileCoords.push({ tx, ty });
  }

  const tiles = await Promise.all(
    tileCoords.map(async ({ tx, ty }) => {
      const record = await getTile(imageId, renderLevel, tx, ty);
      return { tx, ty, record };
    }),
  );

  let drewAny = false;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  for (const { tx, ty, record } of tiles) {
    if (!record) continue;
    const tw = Math.min(TILE_SIZE, lw - tx * TILE_SIZE);
    const th = Math.min(TILE_SIZE, lh - ty * TILE_SIZE);
    const bitmap = await createImageBitmap(record.blob);
    ctx.drawImage(bitmap, tx * TILE_SIZE * sx, ty * TILE_SIZE * sy, tw * sx, th * sy);
    bitmap.close();
    drewAny = true;
  }

  if (!drewAny) {
    // Фолбэк на самый мелкий уровень, если на выбранном уровне тайлов не нашлось.
    const fallback = await getTile(imageId, 0, 0, 0);
    if (fallback) {
      const bitmap = await createImageBitmap(fallback.blob);
      ctx.drawImage(bitmap, 0, 0, width, height);
      bitmap.close();
    }
  }
}

export async function drawFrameThumbnail(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  width: number,
  height: number,
  showMask = true,
): Promise<void> {
  ctx.fillStyle = '#20242A';
  ctx.fillRect(0, 0, width, height);

  if (frame.source.kind === 'procedural') {
    const seed = frame.source.seed;
    const params = paramsForSeed(seed);
    const longSide = Math.max(frame.width, frame.height);
    const imgData = ctx.createImageData(width, height);
    for (let j = 0; j < height; j++) {
      const yNorm = ((j / height) * frame.height) / longSide;
      for (let i = 0; i < width; i++) {
        const xNorm = ((i / width) * frame.width) / longSide;
        const { cls } = classifyPoint(seed, xNorm, yNorm, params);
        const bg = backgroundRockColor(seed, xNorm, yNorm);
        let color = bg;
        if (cls !== 2) {
          const [cr, cg, cb] = classColor(cls);
          const mix = cls === 3 ? 0.5 : 0.8;
          color = [bg[0] * (1 - mix) + cr * mix, bg[1] * (1 - mix) + cg * mix, bg[2] * (1 - mix) + cb * mix];
        }
        const idx = (j * width + i) * 4;
        imgData.data[idx] = color[0];
        imgData.data[idx + 1] = color[1];
        imgData.data[idx + 2] = color[2];
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  } else {
    await drawDexieOriginal(ctx, frame.source.imageId, frame.width, frame.height, width, height);
  }

  if (showMask && frame.maskId) {
    const maskRec = await getMask(frame.maskId);
    if (maskRec) {
      const { width: mw, height: mh, data } = maskRec;
      const maskImgData = ctx.createImageData(width, height);
      for (let j = 0; j < height; j++) {
        const my = Math.min(mh - 1, Math.floor((j / height) * mh));
        const rowOffset = my * mw;
        for (let i = 0; i < width; i++) {
          const mx = Math.min(mw - 1, Math.floor((i / width) * mw));
          const v = data[rowOffset + mx];
          const idx = (j * width + i) * 4;
          if (v === 1) {
            // обычные срастания — зелёный
            maskImgData.data[idx] = 46;
            maskImgData.data[idx + 1] = 125;
            maskImgData.data[idx + 2] = 50;
            maskImgData.data[idx + 3] = 140;
          } else if (v === 2) {
            // тонкие срастания — красный
            maskImgData.data[idx] = 198;
            maskImgData.data[idx + 1] = 40;
            maskImgData.data[idx + 2] = 40;
            maskImgData.data[idx + 3] = 140;
          } else if (v === 3) {
            // тальк — синий
            maskImgData.data[idx] = 21;
            maskImgData.data[idx + 1] = 101;
            maskImgData.data[idx + 2] = 192;
            maskImgData.data[idx + 3] = 140;
          } else {
            maskImgData.data[idx + 3] = 0;
          }
        }
      }
      const tmp = document.createElement('canvas');
      tmp.width = width;
      tmp.height = height;
      tmp.getContext('2d')!.putImageData(maskImgData, 0, 0);
      ctx.drawImage(tmp, 0, 0);
    }
  }
}

export async function getFrameThumbnailDataUrl(frame: Frame, width = 320, height = 200, showMask = true): Promise<string> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  await drawFrameThumbnail(ctx, frame, width, height, showMask);
  return canvas.toDataURL('image/png');
}

interface FrameThumbnailProps {
  frame: Frame;
  width?: number;
  height?: number;
  showMask?: boolean;
}

export function FrameThumbnail({ frame, width = 160, height = 100, showMask = true }: FrameThumbnailProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    void drawFrameThumbnail(ctx, frame, width, height, showMask).then(() => {
      if (cancelled) return;
    });
    return () => {
      cancelled = true;
    };
  }, [frame.id, frame.maskId, frame.source, width, height, showMask, frame.width, frame.height]);

  return <canvas ref={canvasRef} style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 4 }} />;
}
