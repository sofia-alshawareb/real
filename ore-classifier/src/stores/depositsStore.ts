import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Deposit, Mineral } from '../types/models';

function genId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

interface DepositsState {
  deposits: Deposit[];
  addDeposit: (input: Omit<Deposit, 'id' | 'updatedAt' | 'archived'>) => Deposit;
  updateDeposit: (id: string, patch: Partial<Omit<Deposit, 'id'>>) => void;
  archiveDeposit: (id: string, archived: boolean) => void;
  deleteDeposit: (id: string) => boolean;
  addMineral: (depositId: string, mineral: Omit<Mineral, 'id'>) => void;
  updateMineral: (depositId: string, mineralId: string, patch: Partial<Omit<Mineral, 'id'>>) => void;
  removeMineral: (depositId: string, mineralId: string) => void;
  getDeposit: (id: string) => Deposit | undefined;
  hasLinkedExperiments: (id: string) => boolean;
  registerLinkChecker: (fn: (depositId: string) => boolean) => void;
}

let linkChecker: (depositId: string) => boolean = () => false;

export const useDepositsStore = create<DepositsState>()(
  persist(
    (set, get) => ({
      deposits: [],
      addDeposit: (input) => {
        const deposit: Deposit = {
          ...input,
          id: genId('dep'),
          archived: false,
          updatedAt: Date.now(),
        };
        set((s) => ({ deposits: [...s.deposits, deposit] }));
        return deposit;
      },
      updateDeposit: (id, patch) => {
        set((s) => ({
          deposits: s.deposits.map((d) => (d.id === id ? { ...d, ...patch, updatedAt: Date.now() } : d)),
        }));
      },
      archiveDeposit: (id, archived) => {
        set((s) => ({
          deposits: s.deposits.map((d) => (d.id === id ? { ...d, archived, updatedAt: Date.now() } : d)),
        }));
      },
      deleteDeposit: (id) => {
        if (linkChecker(id)) return false;
        set((s) => ({ deposits: s.deposits.filter((d) => d.id !== id) }));
        return true;
      },
      addMineral: (depositId, mineral) => {
        set((s) => ({
          deposits: s.deposits.map((d) =>
            d.id === depositId
              ? { ...d, minerals: [...d.minerals, { ...mineral, id: genId('min') }], updatedAt: Date.now() }
              : d,
          ),
        }));
      },
      updateMineral: (depositId, mineralId, patch) => {
        set((s) => ({
          deposits: s.deposits.map((d) =>
            d.id === depositId
              ? {
                  ...d,
                  minerals: d.minerals.map((m) => (m.id === mineralId ? { ...m, ...patch } : m)),
                  updatedAt: Date.now(),
                }
              : d,
          ),
        }));
      },
      removeMineral: (depositId, mineralId) => {
        set((s) => ({
          deposits: s.deposits.map((d) =>
            d.id === depositId
              ? { ...d, minerals: d.minerals.filter((m) => m.id !== mineralId), updatedAt: Date.now() }
              : d,
          ),
        }));
      },
      getDeposit: (id) => get().deposits.find((d) => d.id === id),
      hasLinkedExperiments: (id) => linkChecker(id),
      registerLinkChecker: (fn) => {
        linkChecker = fn;
      },
    }),
    { name: 'ore.deposits' },
  ),
);
