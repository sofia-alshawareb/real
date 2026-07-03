import { useSettingsStore } from '../stores/settingsStore';
import { useDepositsStore } from '../stores/depositsStore';
import { useExperimentsStore } from '../stores/experimentsStore';
import { retryAllFailed, ensureQueueRunning } from './mockMl/queueRunner';
import { seedIfEmpty } from './seed/seedData';

let bootstrapped = false;

export async function bootstrapApp(): Promise<void> {
  if (bootstrapped) return;
  bootstrapped = true;

  useDepositsStore
    .getState()
    .registerLinkChecker((depositId) => useExperimentsStore.getState().experiments.some((e) => e.depositId === depositId));

  await seedIfEmpty();

  let prevOffline = useSettingsStore.getState().mlOffline;
  useSettingsStore.subscribe((state) => {
    if (prevOffline && !state.mlOffline) {
      retryAllFailed();
    }
    prevOffline = state.mlOffline;
  });

  void ensureQueueRunning();
}
