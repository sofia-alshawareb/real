// Доменные типы прототипа классификации руды по OM-шлифам

export type OreClass = 'routine' | 'hard' | 'talc';

export type MineralRole = 'sulfide' | 'talc' | 'gangue' | 'other';

export type FrameStatus =
  | 'queued'
  | 'processing'
  | 'ml_unavailable'
  | 'manual_only'
  | 'ready'
  | 'reviewed'
  | 'segmentation_edited';

export type ExperimentStatus =
  | 'draft'
  | 'in_progress'
  | 'has_discrepancies'
  | 'completed'
  | 'reported';

export type AnalysisMode = 'ml' | 'manual' | 'mixed';

export interface Mineral {
  id: string;
  name: string;
  role: MineralRole;
  colorHex: string;
  note?: string;
}

export interface Deposit {
  id: string;
  name: string;
  code: string;
  talcThreshold: number; // доля, напр. 0.10
  minerals: Mineral[];
  notes?: string;
  archived: boolean;
  updatedBy: string;
  updatedAt: number;
}

export interface FrameMetrics {
  talcFraction: number;
  sulfideFraction: number;
  gangueFraction: number;
  coarseFraction: number;
  fineFraction: number;
  classifiedShare: number;
}

export type FrameSource =
  | { kind: 'dexie'; imageId: string }
  | { kind: 'procedural'; seed: number };

export interface Frame {
  id: string;
  index: number;
  name: string;
  source: FrameSource;
  width: number;
  height: number;
  pixelSizeUm: number;
  status: FrameStatus;
  isReference: boolean;
  autoMaskId?: string;
  maskId?: string;
  metrics?: FrameMetrics;
  frameClass?: OreClass;
  classReason?: string;
  confidence?: number;
  manuallyEditedMask: boolean;
  manualClassOverride?: OreClass;
  updatedAt: number;
}

export interface ExperimentHistoryEntry {
  at: number;
  author: string;
  action: string;
}

export interface Experiment {
  id: string;
  title: string;
  depositId: string;
  author: string;
  status: ExperimentStatus;
  analysisMode: AnalysisMode;
  referenceFrameId?: string;
  frames: Frame[];
  experimentClass?: OreClass;
  classDerivation?: 'majority' | 'reference' | 'manual';
  history: ExperimentHistoryEntry[];
  openedBy?: string;
  openedAt?: number;
  createdAt: number;
  updatedAt: number;
}

export interface ReportDraft {
  experimentId: string;
  intro: string;
  conclusion: string;
  recommendations: string;
  includedFrameIds: string[];
  snapshotAt: number;
}

export interface MlQueueItem {
  frameId: string;
  experimentId: string;
  attempts: number;
  enqueuedAt: number;
}
