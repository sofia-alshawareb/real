// Источник тайлов OpenSeadragon для загруженных пользователем изображений, хранящихся в Dexie.

import OpenSeadragon from 'openseadragon';
import { getTile } from '../../db/imageRepo';

// @types/openseadragon не описывает fail() и точную форму ImageJob.userData,
// поэтому override-методы намеренно типизированы как `any` (см. downloadTileStart/Abort).

export class DexieTileSource extends OpenSeadragon.TileSource {
  private readonly imageId: string;

  constructor(imageId: string, width: number, height: number) {
    super({ width, height, tileSize: 512, tileOverlap: 0 });
    this.imageId = imageId;
  }

  getTileUrl(level: number, x: number, y: number): string {
    return `dexie://${this.imageId}/${level}/${x}_${y}`;
  }

  downloadTileStart(context: any): void {
    const { level, x, y } = context.tile;
    getTile(this.imageId, level, x, y)
      .then((record) => {
        if (!record) {
          context.fail('Тайл не найден в локальном хранилище');
          return;
        }
        const url = URL.createObjectURL(record.blob);
        context.userData.objectUrl = url;
        const image = new Image();
        image.onload = () => {
          context.finish(image, null, 'image');
          URL.revokeObjectURL(url);
        };
        image.onerror = () => {
          URL.revokeObjectURL(url);
          context.fail('Не удалось декодировать тайл');
        };
        image.src = url;
      })
      .catch((err) => context.fail(err instanceof Error ? err.message : 'Ошибка чтения тайла из хранилища'));
  }

  downloadTileAbort(context: any): void {
    const url = context.userData.objectUrl as string | undefined;
    if (url) URL.revokeObjectURL(url);
  }
}
