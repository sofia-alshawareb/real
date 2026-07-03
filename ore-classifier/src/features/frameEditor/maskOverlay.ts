// Canvas-оверлей маски поверх OpenSeadragon: синхронизация с viewport, послойная колоризация.

import OpenSeadragon from 'openseadragon';
import { MASK_CLASSES, type MaskClassMeta } from '../../theme/palette';
import type { MaskClassKey } from '../../stores/editorStore';

export interface MaskBuffer {
  width: number;
  height: number;
  data: Uint8Array;
}

const CLASS_VALUES: Record<MaskClassKey, number> = {
  sulfide: MASK_CLASSES.sulfide.value,
  gangue: MASK_CLASSES.gangue.value,
  talc: MASK_CLASSES.talc.value,
};

function hexToRgb(hex: string): [number, number, number] {
  const v = hex.replace('#', '');
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

export class MaskOverlay {
  private readonly viewer: OpenSeadragon.Viewer;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly layerCanvases: Record<MaskClassKey, HTMLCanvasElement>;
  private mask: MaskBuffer;
  private readonly maskToNativeScale: number;
  visibleLayers: Record<MaskClassKey, boolean> = { sulfide: true, gangue: true, talc: true };
  opacity = 0.55;
  private disposed = false;
  private readonly unbind: Array<() => void> = [];

  constructor(viewer: OpenSeadragon.Viewer, mask: MaskBuffer, maskToNativeScale: number) {
    this.viewer = viewer;
    this.mask = mask;
    this.maskToNativeScale = maskToNativeScale;

    this.canvas = document.createElement('canvas');
    Object.assign(this.canvas.style, {
      position: 'absolute',
      left: '0',
      top: '0',
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
    });
    this.ctx = this.canvas.getContext('2d')!;
    viewer.canvas.appendChild(this.canvas);

    this.layerCanvases = {
      sulfide: document.createElement('canvas'),
      gangue: document.createElement('canvas'),
      talc: document.createElement('canvas'),
    };

    this.rebuildLayers();
    this.resize();

    const redraw = () => this.redraw();
    const onResize = () => {
      this.resize();
      this.redraw();
    };
    viewer.addHandler('update-viewport', redraw);
    viewer.addHandler('animation', redraw);
    viewer.addHandler('open', redraw);
    viewer.addHandler('resize', onResize);
    this.unbind.push(
      () => viewer.removeHandler('update-viewport', redraw),
      () => viewer.removeHandler('animation', redraw),
      () => viewer.removeHandler('open', redraw),
      () => viewer.removeHandler('resize', onResize),
    );
  }

  getOverlayCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  setMask(mask: MaskBuffer): void {
    this.mask = mask;
    this.rebuildLayers();
    this.redraw();
  }

  /** Быстрая перекраска только изменённого региона (dirty-rect) в координатах маски. */
  updateRegion(x: number, y: number, w: number, h: number): void {
    const clampedX = Math.max(0, x);
    const clampedY = Math.max(0, y);
    const clampedW = Math.min(this.mask.width, x + w) - clampedX;
    const clampedH = Math.min(this.mask.height, y + h) - clampedY;
    if (clampedW <= 0 || clampedH <= 0) return;
    this.paintLayersRegion(clampedX, clampedY, clampedW, clampedH);
    this.redraw();
  }

  setVisibleLayers(layers: Record<MaskClassKey, boolean>): void {
    this.visibleLayers = layers;
    this.redraw();
  }

  setOpacity(opacity: number): void {
    this.opacity = opacity;
    this.redraw();
  }

  resize(): void {
    const rect = this.viewer.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
  }

  private rebuildLayers(): void {
    (Object.keys(this.layerCanvases) as MaskClassKey[]).forEach((key) => {
      const canvas = this.layerCanvases[key];
      canvas.width = this.mask.width;
      canvas.height = this.mask.height;
    });
    this.paintLayersRegion(0, 0, this.mask.width, this.mask.height);
  }

  private paintLayersRegion(x: number, y: number, w: number, h: number): void {
    const { width, data } = this.mask;
    (Object.keys(this.layerCanvases) as MaskClassKey[]).forEach((key) => {
      const meta: MaskClassMeta = MASK_CLASSES[key];
      const [r, g, b] = hexToRgb(meta.color);
      const targetValue = CLASS_VALUES[key];
      const layerCtx = this.layerCanvases[key].getContext('2d')!;
      const imgData = layerCtx.createImageData(w, h);
      for (let j = 0; j < h; j++) {
        const srcRow = (y + j) * width;
        for (let i = 0; i < w; i++) {
          const v = data[srcRow + x + i];
          const idx = (j * w + i) * 4;
          if (v === targetValue) {
            imgData.data[idx] = r;
            imgData.data[idx + 1] = g;
            imgData.data[idx + 2] = b;
            imgData.data[idx + 3] = 255;
          } else {
            imgData.data[idx + 3] = 0;
          }
        }
      }
      layerCtx.clearRect(x, y, w, h);
      layerCtx.putImageData(imgData, x, y);
    });
  }

  private redraw(): void {
    if (this.disposed) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const rect = this.viewer.canvas.getBoundingClientRect();
    let topLeftImg: OpenSeadragon.Point;
    let bottomRightImg: OpenSeadragon.Point;
    try {
      topLeftImg = this.viewer.viewport.viewerElementToImageCoordinates(new OpenSeadragon.Point(0, 0));
      bottomRightImg = this.viewer.viewport.viewerElementToImageCoordinates(
        new OpenSeadragon.Point(rect.width, rect.height),
      );
    } catch {
      return;
    }

    const nx0 = topLeftImg.x / this.maskToNativeScale;
    const ny0 = topLeftImg.y / this.maskToNativeScale;
    const nx1 = bottomRightImg.x / this.maskToNativeScale;
    const ny1 = bottomRightImg.y / this.maskToNativeScale;

    const mx0 = Math.max(0, Math.min(this.mask.width, nx0));
    const my0 = Math.max(0, Math.min(this.mask.height, ny0));
    const mx1 = Math.max(0, Math.min(this.mask.width, nx1));
    const my1 = Math.max(0, Math.min(this.mask.height, ny1));
    const mw = mx1 - mx0;
    const mh = my1 - my0;
    if (mw <= 0 || mh <= 0) return;

    // соответствующий прямоугольник на экранном канвасе (в его собственных, увеличенных на dpr, пикселях)
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    const screenX0 = ((mx0 * this.maskToNativeScale - topLeftImg.x) / (bottomRightImg.x - topLeftImg.x)) * rect.width * scaleX;
    const screenY0 =
      ((my0 * this.maskToNativeScale - topLeftImg.y) / (bottomRightImg.y - topLeftImg.y)) * rect.height * scaleY;
    const screenX1 = ((mx1 * this.maskToNativeScale - topLeftImg.x) / (bottomRightImg.x - topLeftImg.x)) * rect.width * scaleX;
    const screenY1 =
      ((my1 * this.maskToNativeScale - topLeftImg.y) / (bottomRightImg.y - topLeftImg.y)) * rect.height * scaleY;

    ctx.globalAlpha = this.opacity;
    ctx.imageSmoothingEnabled = false;
    (Object.keys(this.layerCanvases) as MaskClassKey[]).forEach((key) => {
      if (!this.visibleLayers[key]) return;
      ctx.drawImage(
        this.layerCanvases[key],
        mx0,
        my0,
        mw,
        mh,
        screenX0,
        screenY0,
        screenX1 - screenX0,
        screenY1 - screenY0,
      );
    });
    ctx.globalAlpha = 1;
  }

  destroy(): void {
    this.disposed = true;
    this.unbind.forEach((fn) => fn());
    this.canvas.remove();
  }
}
