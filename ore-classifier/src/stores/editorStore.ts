import { create } from 'zustand';
import type { MaskClassMeta, MaskClassKey } from '../theme/palette';
import { MASK_CLASSES } from '../theme/palette';

export type EditorTool = 'pan' | 'brush' | 'eraser' | 'polygon' | 'fill' | 'lasso';
export type { MaskClassKey };

interface EditorState {
  tool: EditorTool;
  activeClass: MaskClassKey;
  brushRadius: number;
  overlayOpacity: number;
  visibleLayers: Record<MaskClassKey, boolean>;
  compareMode: boolean;
  viewMode: 'overlay' | 'original' | 'mask';
  setTool: (tool: EditorTool) => void;
  setActiveClass: (cls: MaskClassKey) => void;
  setBrushRadius: (r: number) => void;
  setOverlayOpacity: (o: number) => void;
  toggleLayer: (cls: MaskClassKey) => void;
  setCompareMode: (v: boolean) => void;
  setViewMode: (v: 'overlay' | 'original' | 'mask') => void;
  reset: () => void;
}

export function activeClassMeta(cls: MaskClassKey): MaskClassMeta {
  return MASK_CLASSES[cls];
}

const BRUSH_RADIUS_MAX = 128;
export const DEFAULT_BRUSH_RADIUS = Math.round(BRUSH_RADIUS_MAX * 0.3);

export const useEditorStore = create<EditorState>((set) => ({
  tool: 'pan',
  activeClass: 'coarse',
  brushRadius: DEFAULT_BRUSH_RADIUS,
  overlayOpacity: 0.55,
  visibleLayers: { coarse: true, fine: true, talc: true, matrix: true },
  compareMode: false,
  viewMode: 'overlay',
  setTool: (tool) => set({ tool }),
  setActiveClass: (activeClass) => set({ activeClass }),
  setBrushRadius: (brushRadius) => set({ brushRadius }),
  setOverlayOpacity: (overlayOpacity) => set({ overlayOpacity }),
  toggleLayer: (cls) => set((s) => ({ visibleLayers: { ...s.visibleLayers, [cls]: !s.visibleLayers[cls] } })),
  setCompareMode: (compareMode) => set({ compareMode }),
  setViewMode: (viewMode) => set({ viewMode }),
  reset: () => set({ tool: 'pan', brushRadius: DEFAULT_BRUSH_RADIUS, overlayOpacity: 0.55, compareMode: false, viewMode: 'overlay' }),
}));

export const BRUSH_RADIUS_RANGE = { min: 2, max: BRUSH_RADIUS_MAX };
