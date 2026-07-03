import { useEffect, useRef } from 'react';
import OpenSeadragon from 'openseadragon';
import type { Frame } from '../../types/models';
import { ProceduralTileSource } from '../../services/tiling/ProceduralTileSource';
import { DexieTileSource } from '../../services/tiling/DexieTileSource';

interface ViewerCanvasProps {
  frame: Frame;
  onViewerReady: (viewer: OpenSeadragon.Viewer) => void;
  onViewerDestroy?: () => void;
}

export function ViewerCanvas({ frame, onViewerReady, onViewerDestroy }: ViewerCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const tileSource =
      frame.source.kind === 'procedural'
        ? new ProceduralTileSource(frame.source.seed, frame.width, frame.height)
        : new DexieTileSource(frame.source.imageId, frame.width, frame.height);

    const viewer = OpenSeadragon({
      element: containerRef.current,
      tileSources: tileSource,
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      navigatorHeight: 110,
      navigatorWidth: 150,
      showNavigationControl: false,
      showZoomControl: false,
      showHomeControl: false,
      showFullPageControl: false,
      animationTime: 0.35,
      springStiffness: 9,
      visibilityRatio: 1,
      constrainDuringPan: true,
      minZoomImageRatio: 0.7,
      maxZoomPixelRatio: 6,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true, flickEnabled: false },
      zoomPerScroll: 1.3,
      immediateRender: true,
    });

    onViewerReady(viewer);

    return () => {
      onViewerDestroy?.();
      viewer.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame.id]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', position: 'relative', background: '#20242A' }}
    />
  );
}
