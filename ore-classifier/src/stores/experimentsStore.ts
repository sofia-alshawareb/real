import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  Experiment,
  ExperimentStatus,
  Frame,
  FrameMetrics,
  FrameStatus,
  OreClass,
} from '../types/models';
import { aggregateExperiment } from '../services/rulesEngine';
import { deleteFrameData } from '../db/imageRepo';

export function genId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

const SOFT_LOCK_TTL_MS = 5 * 60 * 1000;

interface ExperimentsState {
  experiments: Experiment[];
  createExperiment: (input: { title: string; depositId: string; author: string }) => Experiment;
  addFrames: (experimentId: string, frames: Frame[]) => void;
  setReferenceFrame: (experimentId: string, frameId: string) => void;
  updateFrame: (experimentId: string, frameId: string, patch: Partial<Frame>, author?: string) => void;
  setFrameResult: (
    experimentId: string,
    frameId: string,
    result: {
      status: FrameStatus;
      maskId?: string;
      autoMaskId?: string;
      metrics?: FrameMetrics;
      frameClass?: OreClass;
      classReason?: string;
      confidence?: number;
    },
  ) => void;
  setManualClassOverride: (experimentId: string, frameId: string, oreClass: OreClass | undefined, author: string) => void;
  recomputeAggregate: (experimentId: string) => void;
  completeExperiment: (experimentId: string, author: string) => void;
  markReported: (experimentId: string, author: string) => void;
  reopenExperiment: (experimentId: string, author: string) => void;
  addHistory: (experimentId: string, author: string, action: string) => void;
  getExperiment: (id: string) => Experiment | undefined;
  deleteExperiment: (id: string) => void;
  openExperiment: (id: string, author: string) => 'locked' | 'ok';
  releaseExperiment: (id: string, author: string) => void;
  forceEditExperiment: (id: string, author: string) => void;
  replaceAll: (experiments: Experiment[]) => void;
}

function computeStatus(currentStatus: ExperimentStatus, frames: Frame[], hasDiscrepancies: boolean): ExperimentStatus {
  if (frames.length === 0) return 'draft';
  if (currentStatus === 'completed' || currentStatus === 'reported') return currentStatus;
  return hasDiscrepancies ? 'has_discrepancies' : 'in_progress';
}

export const useExperimentsStore = create<ExperimentsState>()(
  persist(
    (set, get) => ({
      experiments: [],

      createExperiment: ({ title, depositId, author }) => {
        const now = Date.now();
        const experiment: Experiment = {
          id: genId('exp'),
          title,
          depositId,
          author,
          status: 'draft',
          analysisMode: 'ml',
          frames: [],
          history: [{ at: now, author, action: 'Эксперимент создан' }],
          createdAt: now,
          updatedAt: now,
        };
        set((s) => ({ experiments: [...s.experiments, experiment] }));
        return experiment;
      },

      addFrames: (experimentId, frames) => {
        set((s) => ({
          experiments: s.experiments.map((e) => {
            if (e.id !== experimentId) return e;
            const referenceFrameId = e.referenceFrameId ?? frames[0]?.id;
            const updatedFrames = e.frames.concat(
              frames.map((f, idx) => ({ ...f, isReference: f.id === referenceFrameId, index: e.frames.length + idx })),
            );
            return {
              ...e,
              frames: updatedFrames,
              referenceFrameId,
              status: computeStatus(e.status, updatedFrames, false),
              updatedAt: Date.now(),
              history: [...e.history, { at: Date.now(), author: e.author, action: `Загружено кадров: ${frames.length}` }],
            };
          }),
        }));
      },

      setReferenceFrame: (experimentId, frameId) => {
        set((s) => ({
          experiments: s.experiments.map((e) =>
            e.id === experimentId
              ? {
                  ...e,
                  referenceFrameId: frameId,
                  frames: e.frames.map((f) => ({ ...f, isReference: f.id === frameId })),
                  updatedAt: Date.now(),
                }
              : e,
          ),
        }));
        get().recomputeAggregate(experimentId);
      },

      updateFrame: (experimentId, frameId, patch) => {
        set((s) => ({
          experiments: s.experiments.map((e) =>
            e.id === experimentId
              ? {
                  ...e,
                  frames: e.frames.map((f) => (f.id === frameId ? { ...f, ...patch, updatedAt: Date.now() } : f)),
                  updatedAt: Date.now(),
                }
              : e,
          ),
        }));
      },

      setFrameResult: (experimentId, frameId, result) => {
        get().updateFrame(experimentId, frameId, { ...result, updatedAt: Date.now() });
        get().recomputeAggregate(experimentId);
      },

      setManualClassOverride: (experimentId, frameId, oreClass, author) => {
        get().updateFrame(experimentId, frameId, { manualClassOverride: oreClass, status: 'reviewed' });
        get().addHistory(
          experimentId,
          author,
          oreClass ? `Класс кадра исправлен вручную на "${oreClass}"` : 'Ручная правка класса кадра снята',
        );
        get().recomputeAggregate(experimentId);
      },

      recomputeAggregate: (experimentId) => {
        set((s) => ({
          experiments: s.experiments.map((e) => {
            if (e.id !== experimentId) return e;
            const agg = aggregateExperiment(e.frames, e.referenceFrameId);
            const modes = new Set(
              e.frames.map((f) =>
                f.status === 'manual_only' ? 'manual' : f.manuallyEditedMask ? 'manual' : 'ml',
              ),
            );
            const analysisMode = modes.size > 1 ? 'mixed' : modes.has('manual') ? 'manual' : 'ml';
            return {
              ...e,
              experimentClass: agg.experimentClass,
              classDerivation: agg.derivation,
              analysisMode,
              status: computeStatus(e.status, e.frames, agg.hasDiscrepancies),
              updatedAt: Date.now(),
            };
          }),
        }));
      },

      completeExperiment: (experimentId, author) => {
        set((s) => ({
          experiments: s.experiments.map((e) =>
            e.id === experimentId ? { ...e, status: 'completed', updatedAt: Date.now() } : e,
          ),
        }));
        get().addHistory(experimentId, author, 'Эксперимент завершён');
      },

      markReported: (experimentId, author) => {
        set((s) => ({
          experiments: s.experiments.map((e) =>
            e.id === experimentId ? { ...e, status: 'reported', updatedAt: Date.now() } : e,
          ),
        }));
        get().addHistory(experimentId, author, 'Отчёт выгружен');
      },

      reopenExperiment: (experimentId, author) => {
        set((s) => ({
          experiments: s.experiments.map((e) =>
            e.id === experimentId ? { ...e, status: 'in_progress', updatedAt: Date.now() } : e,
          ),
        }));
        get().addHistory(experimentId, author, 'Эксперимент возвращён в работу');
        get().recomputeAggregate(experimentId);
      },

      addHistory: (experimentId, author, action) => {
        set((s) => ({
          experiments: s.experiments.map((e) =>
            e.id === experimentId
              ? { ...e, history: [...e.history, { at: Date.now(), author, action }] }
              : e,
          ),
        }));
      },

      getExperiment: (id) => get().experiments.find((e) => e.id === id),

      deleteExperiment: (id) => {
        const experiment = get().getExperiment(id);
        if (experiment) {
          for (const frame of experiment.frames) {
            const imageId = frame.source.kind === 'dexie' ? frame.source.imageId : undefined;
            void deleteFrameData(frame.id, imageId);
          }
        }
        set((s) => ({ experiments: s.experiments.filter((e) => e.id !== id) }));
      },

      openExperiment: (id, author) => {
        const e = get().getExperiment(id);
        if (!e) return 'ok';
        const now = Date.now();
        if (e.openedBy && e.openedBy !== author && e.openedAt && now - e.openedAt < SOFT_LOCK_TTL_MS) {
          return 'locked';
        }
        set((s) => ({
          experiments: s.experiments.map((exp) =>
            exp.id === id ? { ...exp, openedBy: author, openedAt: now } : exp,
          ),
        }));
        return 'ok';
      },

      releaseExperiment: (id, author) => {
        set((s) => ({
          experiments: s.experiments.map((exp) =>
            exp.id === id && exp.openedBy === author ? { ...exp, openedBy: undefined, openedAt: undefined } : exp,
          ),
        }));
      },

      forceEditExperiment: (id, author) => {
        set((s) => ({
          experiments: s.experiments.map((exp) =>
            exp.id === id ? { ...exp, openedBy: author, openedAt: Date.now() } : exp,
          ),
        }));
      },

      replaceAll: (experiments) => set({ experiments }),
    }),
    { name: 'ore.experiments' },
  ),
);
