import { useDepositsStore } from '../../stores/depositsStore';
import { useExperimentsStore, genId } from '../../stores/experimentsStore';
import { useMlQueueStore } from '../../stores/mlQueueStore';
import { clearAllData } from '../../db/db';
import { putMask } from '../../db/imageRepo';
import { generateMaskData, paramsForSeed, maskWorkingSize } from '../grainModel';
import { calcMetrics } from '../metricsCalc';
import { classifyFrame } from '../rulesEngine';
import { findSeedForClass } from './seedGeology';
import type { Frame, Mineral } from '../../types/models';

const SEEDED_FLAG = 'ore.seeded.v1';
const DEMO_AUTHOR = 'Демо-эксперт';
const NATIVE_W = 12000;
const NATIVE_H = 7500;

function minerals(list: Array<[string, Mineral['role'], string]>): Mineral[] {
  return list.map(([name, role, colorHex]) => ({ id: genId('min'), name, role, colorHex }));
}

function makeFrameStub(index: number, seed: number, name: string): Frame {
  return {
    id: genId('frame'),
    index,
    name,
    source: { kind: 'procedural', seed },
    width: NATIVE_W,
    height: NATIVE_H,
    pixelSizeUm: 0.5,
    status: 'queued',
    isReference: false,
    manuallyEditedMask: false,
    updatedAt: Date.now(),
  };
}

async function generateAndStoreResult(experimentId: string, frame: Frame, talcThreshold: number, reviewed: boolean) {
  const seed = frame.source.kind === 'procedural' ? frame.source.seed : 0;
  const params = paramsForSeed(seed);
  const { mw, mh, scale } = maskWorkingSize(frame.width, frame.height);
  const data = generateMaskData(seed, mw, mh, params);
  const maskId = genId('mask');
  await putMask({ id: maskId, frameId: frame.id, width: mw, height: mh, data });
  const metrics = calcMetrics({ width: mw, height: mh, data }, frame.pixelSizeUm, scale);
  const { oreClass, reason } = classifyFrame(metrics, talcThreshold);
  const confidence = Math.round((0.78 + Math.random() * 0.2) * 100) / 100;
  useExperimentsStore.getState().setFrameResult(experimentId, frame.id, {
    status: reviewed ? 'reviewed' : 'ready',
    maskId,
    autoMaskId: maskId,
    metrics,
    frameClass: oreClass,
    classReason: reason,
    confidence,
  });
}

export async function seedIfEmpty(): Promise<void> {
  if (localStorage.getItem(SEEDED_FLAG) === 'true') return;
  await runSeed();
  localStorage.setItem(SEEDED_FLAG, 'true');
}

export async function resetDemoData(): Promise<void> {
  await clearAllData();
  useExperimentsStore.setState({ experiments: [] });
  useDepositsStore.setState({ deposits: [] });
  useMlQueueStore.setState({ queue: [] });
  localStorage.removeItem(SEEDED_FLAG);
  await runSeed();
  localStorage.setItem(SEEDED_FLAG, 'true');
}

async function runSeed(): Promise<void> {
  const depositsStore = useDepositsStore.getState();

  const talnakh = depositsStore.addDeposit({
    name: 'Норильск-Талнах',
    code: 'NT',
    talcThreshold: 0.1,
    notes: 'Сульфидные медно-никелевые руды, участок Талнахский.',
    minerals: minerals([
      ['Пирротин', 'sulfide', '#D9A441'],
      ['Пентландит', 'sulfide', '#E8B94A'],
      ['Халькопирит', 'sulfide', '#C98A2B'],
      ['Тальк', 'talc', '#2F6F63'],
      ['Серпентин', 'gangue', '#9C9C9E'],
      ['Оливин', 'gangue', '#8B8B96'],
    ]),
    updatedBy: DEMO_AUTHOR,
  });

  const oktyabrskoe = depositsStore.addDeposit({
    name: 'Октябрьское',
    code: 'OKT',
    talcThreshold: 0.08,
    notes: 'Богатые сплошные руды, повышенный риск оталькования.',
    minerals: minerals([
      ['Пирротин', 'sulfide', '#D9A441'],
      ['Пентландит', 'sulfide', '#E8B94A'],
      ['Тальк', 'talc', '#2F6F63'],
      ['Серпентин', 'gangue', '#9C9C9E'],
    ]),
    updatedBy: DEMO_AUTHOR,
  });

  const zhdanov = depositsStore.addDeposit({
    name: 'Жданов',
    code: 'ZHD',
    talcThreshold: 0.12,
    notes: 'Вкраплённые руды, тонкая вкрапленность сульфидов.',
    minerals: minerals([
      ['Халькопирит', 'sulfide', '#C98A2B'],
      ['Пентландит', 'sulfide', '#E8B94A'],
      ['Тальк', 'talc', '#2F6F63'],
      ['Оливин', 'gangue', '#8B8B96'],
    ]),
    updatedBy: DEMO_AUTHOR,
  });

  const expStore = useExperimentsStore.getState();

  // Эксперимент 1: завершён, 5 кадров, рядовая руда
  const exp1 = expStore.createExperiment({
    title: 'Партия 1042 — смена 2, участок Талнахский',
    depositId: talnakh.id,
    author: DEMO_AUTHOR,
  });
  const exp1Frames = Array.from({ length: 5 }, (_, i) =>
    makeFrameStub(i, findSeedForClass('routine', talnakh.talcThreshold, 1000 + i * 137), `Шлиф ${i + 1}`),
  );
  expStore.addFrames(exp1.id, exp1Frames);
  for (const f of exp1Frames) {
    await generateAndStoreResult(exp1.id, f, talnakh.talcThreshold, true);
  }
  expStore.completeExperiment(exp1.id, DEMO_AUTHOR);

  // Эксперимент 2: расхождения между кадрами, 4 кадра разных классов
  const exp2 = expStore.createExperiment({
    title: 'Партия 2077 — входной контроль',
    depositId: oktyabrskoe.id,
    author: DEMO_AUTHOR,
  });
  const targets2: Array<'routine' | 'hard' | 'talc'> = ['routine', 'routine', 'hard', 'talc'];
  const exp2Frames = targets2.map((t, i) =>
    makeFrameStub(i, findSeedForClass(t, oktyabrskoe.talcThreshold, 5000 + i * 251), `Шлиф ${i + 1}`),
  );
  expStore.addFrames(exp2.id, exp2Frames);
  for (const f of exp2Frames) {
    await generateAndStoreResult(exp2.id, f, oktyabrskoe.talcThreshold, false);
  }

  // Эксперимент 3: в работе, кадры ещё в очереди — оживает после запуска приложения
  const exp3 = expStore.createExperiment({
    title: 'Партия 3311 — экспресс-проба',
    depositId: zhdanov.id,
    author: DEMO_AUTHOR,
  });
  const exp3Frames = Array.from({ length: 3 }, (_, i) => makeFrameStub(i, 9000 + i * 91, `Шлиф ${i + 1}`));
  expStore.addFrames(exp3.id, exp3Frames);
  for (const f of exp3Frames) {
    useMlQueueStore.getState().enqueue(f.id, exp3.id);
  }

  // Эксперимент 4: черновик без кадров
  expStore.createExperiment({
    title: 'Новая партия — черновик',
    depositId: talnakh.id,
    author: DEMO_AUTHOR,
  });
}
