import { clearAllData } from '../db/db';
import { useDepositsStore } from '../stores/depositsStore';
import { useExperimentsStore } from '../stores/experimentsStore';
import { useMlQueueStore } from '../stores/mlQueueStore';
import { seedDepositsIfEmpty } from './depositCatalog';

const LEGACY_SEEDED_FLAG = 'ore.seeded.v2';
const LEGACY_FILL_QUEUE_KEY = 'ore.seedFillQueue.v2';
const DEMO_REMOVED_FLAG = 'ore.demoRemoved.v1';

/** Wipe experiments, deposits, masks, and ML queue from browser storage. */
export async function clearAppData(): Promise<void> {
  await clearAllData();
  useExperimentsStore.setState({ experiments: [] });
  useDepositsStore.setState({ deposits: [] });
  useMlQueueStore.setState({ queue: [] });
  localStorage.removeItem(LEGACY_SEEDED_FLAG);
  localStorage.removeItem(LEGACY_FILL_QUEUE_KEY);
  seedDepositsIfEmpty();
}

/**
 * One-time removal of bundled demo experiments/deposits.
 * Runs once per browser after upgrade; does not recreate demo content.
 */
export async function removeLegacyDemoDataOnce(): Promise<void> {
  if (localStorage.getItem(DEMO_REMOVED_FLAG) === 'true') return;

  const hadDemoBundle =
    localStorage.getItem(LEGACY_SEEDED_FLAG) === 'true' ||
    useExperimentsStore.getState().experiments.length > 0 ||
    useDepositsStore.getState().deposits.length > 0;

  if (hadDemoBundle) {
    await clearAppData();
  }

  localStorage.setItem(DEMO_REMOVED_FLAG, 'true');
}
