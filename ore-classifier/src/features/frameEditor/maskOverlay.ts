// Canvas-оверлей маски поверх OpenSeadragon: синхронизация с viewport, послойная колоризация.

import OpenSeadragon from 'openseadragon';
import { MASK_CLASSES, type MaskClassMeta, type MaskClassKey } from '../../theme/palette';

export interface MaskBuffer {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface PreviewPoint {
  x: number;
  y: number;
}

const MASK_CLASS_KEYS = Object.keys(MASK_CLASSES) as MaskClassKey[];

const CLASS_VALUES: Record<MaskClassKey, number> = Object.fromEntries(
  MASK_CLASS_KEYS.map((key) => [key, MASK_CLASSES[key].value]),
) as Record<MaskClassKey, number>;

function hexToRgb(hex: string): [number, number, number] {
  const v = hex.replace('#', '');
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

export class MaskOverlay {
  private readonly viewer: OpenSeadragon.Viewer;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly previewCanvas: HTMLCanvasElement;
  private readonly previewCtx: CanvasRenderingContext2D;
  private readonly layerCanvases: Record<MaskClassKey, HTMLCanvasElement>;
  private mask: MaskBuffer;
  private readonly maskToNativeScale: number;
  private previewPoints: PreviewPoint[] | null = null;
  private previewClosed = false;
  visibleLayers: Record<MaskClassKey, boolean> = Object.fromEntries(
    MASK_CLASS_KEYS.map((key) => [key, true]),
  ) as Record<MaskClassKey, boolean>;
  opacity = 0.55;
  viewMode: 'overlay' | 'original' | 'mask' = 'overlay';
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

    // Отдельный канвас поверх основного — для превью контура полигона/лассо, не зависит от opacity/viewMode маски.
    this.previewCanvas = document.createElement('canvas');
    Object.assign(this.previewCanvas.style, {
      position: 'absolute',
      left: '0',
      top: '0',
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
    });
    this.previewCanvas.dataset.testid = 'mask-preview-canvas';
    this.previewCtx = this.previewCanvas.getContext('2d')!;
    viewer.canvas.appendChild(this.previewCanvas);

    this.layerCanvases = Object.fromEntries(
      MASK_CLASS_KEYS.map((key) => [key, document.createElement('canvas')]),
    ) as Record<MaskClassKey, HTMLCanvasElement>;

    this.rebuildLayers();
    this.resize();

    const redrawAll = () => {
      this.redraw();
      this.redrawPreview();
    };
    const onResize = () => {
      this.resize();
      this.redraw();
      this.redrawPreview();
    };
    viewer.addHandler('update-viewport', redrawAll);
    viewer.addHandler('animation', redrawAll);
    viewer.addHandler('open', redrawAll);
    viewer.addHandler('resize', onResize);
    this.unbind.push(
      () => viewer.removeHandler('update-viewport', redrawAll),
      () => viewer.removeHandler('animation', redrawAll),
      () => viewer.removeHandler('open', redrawAll),
      () => viewer.removeHandler('resize', onResize),
    );
  }

  getOverlayCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  /**
   * Показывает «резиновый» контур в процессе рисования полигоном/лассо (в координатах маски).
   * closed=true — дополнительно соединяет последнюю точку с первой (для превью замкнутой фигуры).
   * points=null — скрывает превью.
   */
  setPreview(points: PreviewPoint[] | null, closed = false): void {
    this.previewPoints = points;
    this.previewClosed = closed;
    this.redrawPreview();
  }

  private maskToScreen(mx: number, my: number): { x: number; y: number } {
    const nativeX = mx * this.maskToNativeScale;
    const nativeY = my * this.maskToNativeScale;
    const viewerPoint = this.viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(nativeX, nativeY));
    const dpr = window.devicePixelRatio || 1;
    return { x: viewerPoint.x * dpr, y: viewerPoint.y * dpr };
  }

  private redrawPreview(): void {
    if (this.disposed) return;
    const ctx = this.previewCtx;
    ctx.clearRect(0, 0, this.previewCanvas.width, this.previewCanvas.height);
    const points = this.previewPoints;
    if (!points || points.length === 0) return;

    let screenPoints: { x: number; y: number }[];
    try {
      screenPoints = points.map((p) => this.maskToScreen(p.x, p.y));
    } catch {
      return;
    }

    const drawPath = () => {
      ctx.beginPath();
      ctx.moveTo(screenPoints[0].x, screenPoints[0].y);
      for (let i = 1; i < screenPoints.length; i++) ctx.lineTo(screenPoints[i].x, screenPoints[i].y);
      if (this.previewClosed && screenPoints.length > 2) ctx.closePath();
    };

    // Тёмная подложка + светлая пунктирная линия поверх — заметно на любом фоне/цвете маски.
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(0,0,0,0.65)';
    ctx.setLineDash([]);
    drawPath();
    ctx.stroke();

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#ffffff';
    ctx.setLineDash([6, 4]);
    drawPath();
    ctx.stroke();
    ctx.setLineDash([]);

    // Точки-вершины
    ctx.fillStyle = '#ffffff';
    for (const p of screenPoints) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
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

  /** original — скрыть маску полностью, mask — показать маску на 100% (без просвечивания снимка), overlay — обычный режим с прозрачностью. */
  setViewMode(mode: 'overlay' | 'original' | 'mask'): void {
    this.viewMode = mode;
    this.redraw();
  }

  resize(): void {
    const rect = this.viewer.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    this.previewCanvas.width = this.canvas.width;
    this.previewCanvas.height = this.canvas.height;
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
    if (this.viewMode === 'original') return;

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

    ctx.globalAlpha = this.viewMode === 'mask' ? 1 : this.opacity;
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
    this.previewCanvas.remove();
  }
}
