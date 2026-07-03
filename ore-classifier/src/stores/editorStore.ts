import { create } from 'zustand';
import type { MaskClassMeta } from '../theme/palette';
import { MASK_CLASSES } from '../theme/palette';

export type EditorTool = 'pan' | 'brush' | 'eraser' | 'polygon';
export type MaskClassKey = keyof typeof MASK_CLASSES;

interface EditorState {
  tool: EditorTool;
  activeClass: MaskClassKey;
  brushRadius: number;
  overlayOpacity: number;
  visibleLayers: Record<MaskClassKey, boolean>;
  compareMode: boolean;
  setTool: (tool: EditorTool) => void;
  setActiveClass: (cls: MaskClassKey) => void;
  setBrushRadius: (r: number) => void;
  setOverlayOpacity: (o: number) => void;
  toggleLayer: (cls: MaskClassKey) => void;
  setCompareMode: (v: boolean) => void;
  reset: () => void;
}

export function activeClassMeta(cls: MaskClassKey): MaskClassMeta {
  return MASK_CLASSES[cls];
}

export const useEditorStore = create<EditorState>((set) => ({
  tool: 'pan',
  activeClass: 'sulfide',
  brushRadius: 24,
  overlayOpacity: 0.55,
  visibleLayers: { sulfide: true, gangue: true, talc: true },
  compareMode: false,
  setTool: (tool) => set({ tool }),
  setActiveClass: (activeClass) => set({ activeClass }),
  setBrushRadius: (brushRadius) => set({ brushRadius }),
  setOverlayOpacity: (overlayOpacity) => set({ overlayOpacity }),
  toggleLayer: (cls) => set((s) => ({ visibleLayers: { ...s.visibleLayers, [cls]: !s.visibleLayers[cls] } })),
  setCompareMode: (compareMode) => set({ compareMode }),
  reset: () => set({ tool: 'pan', brushRadius: 24, overlayOpacity: 0.55, compareMode: false }),
}));
