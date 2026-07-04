// Воркер очереди ML: mock (демо) или реальный backend в зависимости от mlOffline.

import { useMlQueueStore, type QueueItem } from '../../stores/mlQueueStore';
import { useExperimentsStore, genId } from '../../stores/experimentsStore';
import { useDepositsStore } from '../../stores/depositsStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { putMask } from '../../db/imageRepo';
import { calcMetrics } from '../metricsCalc';
import { classifyFrame } from '../rulesEngine';
import { segmentFrameMock, MlUnavailableError } from './mockMlService';
import { segmentFrameReal } from '../ml/realMlClient';
import { notify } from '../../utils/toast';

let running = false;

export { MlUnavailableError } from '../ml/errors';

export function enqueueFrame(experimentId: string, frameId: string): void {
  useMlQueueStore.getState().enqueue(frameId, experimentId);
  void ensureQueueRunning();
}

export function retryFrame(experimentId: string, frameId: string): void {
  useExperimentsStore.getState().updateFrame(experimentId, frameId, { status: 'queued' });
  useMlQueueStore.getState().retryFrame(frameId, experimentId);
  void ensureQueueRunning();
}

export function retryAllFailed(): void {
  const count = useMlQueueStore.getState().retryAllFailed();
  if (count > 0) {
    notify(`Очередь ML: повторная обработка ${count} кадр(ов)`, 'success');
    void ensureQueueRunning();
  }
}

async function processItem(item: QueueItem): Promise<void> {
  const queueStore = useMlQueueStore.getState();
  const expStore = useExperimentsStore.getState();
  const experiment = expStore.getExperiment(item.experimentId);
  const frame = experiment?.frames.find((f) => f.id === item.frameId);
  if (!experiment || !frame) {
    queueStore.markDone(item.frameId);
    return;
  }

  queueStore.markProcessing(item.frameId);
  expStore.updateFrame(item.experimentId, item.frameId, { status: 'processing' });

  const useMock = useSettingsStore.getState().mlOffline || frame.source.kind === 'procedural';

  try {
    let seg;
    let backendImageId: string | undefined;

    if (useMock) {
      seg = await segmentFrameMock(frame);
    } else {
      const real = await segmentFrameReal(frame);
      seg = real;
      backendImageId = real.backendImageId;
    }

    if (backendImageId) {
      expStore.updateFrame(item.experimentId, item.frameId, { backendImageId });
    }

    const maskId = genId('mask');
    await putMask({ id: maskId, frameId: frame.id, width: seg.mw, height: seg.mh, data: seg.data });

    const deposit = useDepositsStore.getState().getDeposit(experiment.depositId);
    const threshold = deposit?.talcThreshold ?? 0.1;
    const metrics = calcMetrics({ width: seg.mw, height: seg.mh, data: seg.data });
    const { oreClass, reason } = classifyFrame(metrics, threshold);

    expStore.setFrameResult(item.experimentId, item.frameId, {
      status: 'ready',
      maskId,
      autoMaskId: maskId,
      metrics,
      frameClass: oreClass,
      classReason: reason,
      confidence: seg.confidence,
    });
    expStore.addHistory(item.experimentId, experiment.author, `Кадр «${frame.name}» обработан: ${reason}`);
    queueStore.markDone(item.frameId);
  } catch (err) {
    expStore.updateFrame(item.experimentId, item.frameId, { status: 'ml_unavailable' });
    queueStore.markFailed(item.frameId);
    const message =
      err instanceof MlUnavailableError
        ? `${err.message}. Изображение сохранено, обработка возобновится автоматически. Можно разметить кадр вручную.`
        : 'Не удалось обработать кадр. Попробуйте позже.';
    notify(message, 'warning');
  }
}

export async function ensureQueueRunning(): Promise<void> {
  if (running) return;
  running = true;
  try {
    for (;;) {
      const item = useMlQueueStore.getState().queue.find((q) => q.status === 'pending');
      if (!item) break;
      await processItem(item);
    }
  } finally {
    running = false;
  }
}
