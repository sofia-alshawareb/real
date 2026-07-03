import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type QueueItemStatus = 'pending' | 'processing' | 'failed';

export interface QueueItem {
  frameId: string;
  experimentId: string;
  attempts: number;
  status: QueueItemStatus;
  enqueuedAt: number;
}

interface MlQueueState {
  queue: QueueItem[];
  enqueue: (frameId: string, experimentId: string) => void;
  markProcessing: (frameId: string) => void;
  markFailed: (frameId: string) => void;
  markDone: (frameId: string) => void;
  retryFrame: (frameId: string, experimentId: string) => void;
  retryAllFailed: () => number;
  removeByExperiment: (experimentId: string) => void;
  pendingCount: () => number;
  failedCount: () => number;
}

export const useMlQueueStore = create<MlQueueState>()(
  persist(
    (set, get) => ({
      queue: [],

      enqueue: (frameId, experimentId) => {
        set((s) => {
          if (s.queue.some((q) => q.frameId === frameId)) return s;
          return {
            queue: [...s.queue, { frameId, experimentId, attempts: 0, status: 'pending', enqueuedAt: Date.now() }],
          };
        });
      },

      markProcessing: (frameId) => {
        set((s) => ({
          queue: s.queue.map((q) => (q.frameId === frameId ? { ...q, status: 'processing', attempts: q.attempts + 1 } : q)),
        }));
      },

      markFailed: (frameId) => {
        set((s) => ({ queue: s.queue.map((q) => (q.frameId === frameId ? { ...q, status: 'failed' } : q)) }));
      },

      markDone: (frameId) => {
        set((s) => ({ queue: s.queue.filter((q) => q.frameId !== frameId) }));
      },

      retryFrame: (frameId, experimentId) => {
        set((s) => {
          if (s.queue.some((q) => q.frameId === frameId)) {
            return { queue: s.queue.map((q) => (q.frameId === frameId ? { ...q, status: 'pending' } : q)) };
          }
          return {
            queue: [...s.queue, { frameId, experimentId, attempts: 0, status: 'pending', enqueuedAt: Date.now() }],
          };
        });
      },

      retryAllFailed: () => {
        let count = 0;
        set((s) => ({
          queue: s.queue.map((q) => {
            if (q.status === 'failed') {
              count++;
              return { ...q, status: 'pending' as QueueItemStatus };
            }
            return q;
          }),
        }));
        return count;
      },

      removeByExperiment: (experimentId) => {
        set((s) => ({ queue: s.queue.filter((q) => q.experimentId !== experimentId) }));
      },

      pendingCount: () => get().queue.filter((q) => q.status === 'pending').length,
      failedCount: () => get().queue.filter((q) => q.status === 'failed').length,
    }),
    { name: 'ore.mlqueue' },
  ),
);
