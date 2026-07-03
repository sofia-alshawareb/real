// Источник тайлов OpenSeadragon, рисующий панораму шлифа на лету по grain-модели.
// Ноль хранения на диске, детерминированный "гигапиксельный" deep-zoom для демо-данных.

import OpenSeadragon from 'openseadragon';
import { renderTextureTile, paramsForSeed } from '../grainModel';

// @types/openseadragon не описывает fail() и точную форму ImageJob.tile,
// поэтому override-методы намеренно типизированы как `any` (см. downloadTileStart/Abort).

export class ProceduralTileSource extends OpenSeadragon.TileSource {
  private readonly seed: number;

  constructor(seed: number, width: number, height: number) {
    super({ width, height, tileSize: 512, tileOverlap: 0 });
    this.seed = seed;
  }

  getTileUrl(level: number, x: number, y: number): string {
    return `procedural://${this.seed}/${level}/${x}_${y}`;
  }

  downloadTileStart(context: any): void {
    try {
      const { bounds, sourceBounds } = context.tile;
      const w = Math.max(1, Math.round(sourceBounds.width));
      const h = Math.max(1, Math.round(sourceBounds.height));
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        context.fail('Canvas 2D недоступен');
        return;
      }
      const params = paramsForSeed(this.seed);
      renderTextureTile(ctx, this.seed, bounds.x, bounds.y, bounds.width, bounds.height, w, h, params);
      context.finish(ctx, null, 'context2d');
    } catch (err) {
      context.fail(err instanceof Error ? err.message : 'Ошибка генерации тайла');
    }
  }

  downloadTileAbort(): void {
    // процедурная генерация синхронна — отменять нечего
  }
}
