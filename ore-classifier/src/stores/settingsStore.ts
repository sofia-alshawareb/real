import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsState {
  authorName: string;
  mlOffline: boolean;
  mlFailureRate: number;
  setAuthorName: (name: string) => void;
  setMlOffline: (offline: boolean) => void;
  setMlFailureRate: (rate: number) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      authorName: '',
      mlOffline: false,
      mlFailureRate: 0.1,
      setAuthorName: (name) => set({ authorName: name }),
      setMlOffline: (offline) => set({ mlOffline: offline }),
      setMlFailureRate: (rate) => set({ mlFailureRate: rate }),
    }),
    { name: 'ore.settings' },
  ),
);
