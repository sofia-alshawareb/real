// Растеризация штрихов кисти/ластика и полигонов в индексированную маску.

export interface DirtyRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface MaskLike {
  width: number;
  height: number;
  data: Uint8Array;
}

export function unionRect(a: DirtyRect, b: DirtyRect): DirtyRect {
  const x0 = Math.min(a.x, b.x);
  const y0 = Math.min(a.y, b.y);
  const x1 = Math.max(a.x + a.w, b.x + b.w);
  const y1 = Math.max(a.y + a.h, b.y + b.h);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

export function stampCircle(mask: MaskLike, cx: number, cy: number, radius: number, value: number): DirtyRect {
  const x0 = Math.max(0, Math.floor(cx - radius));
  const y0 = Math.max(0, Math.floor(cy - radius));
  const x1 = Math.min(mask.width - 1, Math.ceil(cx + radius));
  const y1 = Math.min(mask.height - 1, Math.ceil(cy + radius));
  const r2 = radius * radius;
  for (let y = y0; y <= y1; y++) {
    const dy = y + 0.5 - cy;
    const rowOffset = y * mask.width;
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - cx;
      if (dx * dx + dy * dy <= r2) {
        mask.data[rowOffset + x] = value;
      }
    }
  }
  return { x: x0, y: y0, w: Math.max(0, x1 - x0 + 1), h: Math.max(0, y1 - y0 + 1) };
}

export function stampSegment(
  mask: MaskLike,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  radius: number,
  value: number,
): DirtyRect {
  const dist = Math.hypot(x1 - x0, y1 - y0);
  const step = Math.max(1, radius / 2.5);
  const steps = Math.max(1, Math.ceil(dist / step));
  let rect: DirtyRect | null = null;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const cx = x0 + (x1 - x0) * t;
    const cy = y0 + (y1 - y0) * t;
    const r = stampCircle(mask, cx, cy, radius, value);
    rect = rect ? unionRect(rect, r) : r;
  }
  return rect ?? { x: Math.floor(x0), y: Math.floor(y0), w: 1, h: 1 };
}

export interface Point2D {
  x: number;
  y: number;
}

export function rasterizePolygon(mask: MaskLike, points: Point2D[], value: number): DirtyRect {
  if (points.length < 3) return { x: 0, y: 0, w: 0, h: 0 };
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  const y0 = Math.max(0, Math.floor(minY));
  const y1 = Math.min(mask.height - 1, Math.ceil(maxY));
  const x0b = Math.max(0, Math.floor(minX));
  const x1b = Math.min(mask.width - 1, Math.ceil(maxX));

  for (let y = y0; y <= y1; y++) {
    const yc = y + 0.5;
    const xs: number[] = [];
    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      const b = points[(i + 1) % points.length];
      if ((a.y <= yc && b.y > yc) || (b.y <= yc && a.y > yc)) {
        const t = (yc - a.y) / (b.y - a.y);
        xs.push(a.x + t * (b.x - a.x));
      }
    }
    xs.sort((m, n) => m - n);
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const xStart = Math.max(x0b, Math.round(xs[i]));
      const xEnd = Math.min(x1b, Math.round(xs[i + 1]));
      const rowOffset = y * mask.width;
      for (let x = xStart; x <= xEnd; x++) {
        mask.data[rowOffset + x] = value;
      }
    }
  }
  return { x: x0b, y: y0, w: Math.max(0, x1b - x0b + 1), h: Math.max(0, y1 - y0 + 1) };
}

export function imagePointToMask(imgX: number, imgY: number, maskToNativeScale: number): Point2D {
  return { x: imgX / maskToNativeScale, y: imgY / maskToNativeScale };
}
