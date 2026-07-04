export interface SegmentationResult {
  data: Uint8Array;
  mw: number;
  mh: number;
  maskToNativeScale: number;
  confidence: number;
}
