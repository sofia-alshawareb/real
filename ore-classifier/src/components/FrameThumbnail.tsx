import { useEffect, useRef } from 'react';
import type { Frame } from '../types/models';
import { paramsForSeed, classifyPoint, backgroundRockColor, classColor } from '../services/grainModel';
import { getTile, getMask } from '../db/imageRepo';

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
    const tile = await getTile(frame.source.imageId, 0, 0, 0);
    if (tile) {
      const bitmap = await createImageBitmap(tile.blob);
      ctx.drawImage(bitmap, 0, 0, width, height);
      bitmap.close();
    }
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
            maskImgData.data[idx] = 255;
            maskImgData.data[idx + 1] = 179;
            maskImgData.data[idx + 2] = 0;
            maskImgData.data[idx + 3] = 140;
          } else if (v === 3) {
            maskImgData.data[idx] = 0;
            maskImgData.data[idx + 1] = 105;
            maskImgData.data[idx + 2] = 92;
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
