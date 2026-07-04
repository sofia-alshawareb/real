import { useDepositsStore } from '../../stores/depositsStore';
import { useExperimentsStore, genId } from '../../stores/experimentsStore';
import { useMlQueueStore } from '../../stores/mlQueueStore';
import { clearAllData } from '../../db/db';
import { putMask } from '../../db/imageRepo';
import { generateMaskDataChunked, paramsForSeed, maskWorkingSize } from '../grainModel';
import { calcMetrics } from '../metricsCalc';
import { classifyFrame } from '../rulesEngine';
import { findSeedAndMetricsForClass } from './seedGeology';
import type { Deposit, DepositReserves, Frame, Mineral, MetalGrades, OreClass } from '../../types/models';

const SEEDED_FLAG = 'ore.seeded.v2';
const FILL_QUEUE_KEY = 'ore.seedFillQueue.v2';
const DEMO_AUTHOR = 'Демо-эксперт';
const NATIVE_W = 12000;
const NATIVE_H = 7500;
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

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

/**
 * Задание на «доотсеивание» кадра тяжёлыми вычислениями (полноразмерная маска + уточнённые метрики).
 * Структурные сущности (месторождения/эксперименты/кадры) уже созданы и сохранены к моменту постановки
 * задания в очередь, а сами кадры уже получили быстрый предварительный результат (см. buildQuickFrame) —
 * очередь лишь заменяет его более точным полноразмерным результатом. Очередь хранится в localStorage,
 * чтобы фон обработки переживал перезагрузку страницы.
 */
interface SeedFillJob {
  experimentId: string;
  frameId: string;
  threshold: number;
  reviewed: boolean;
  /** Если задан, экспермент «архивный» (backdated) — после доработки кадра метку времени нужно
   * восстановить, иначе setFrameResult молча вернёт updatedAt эксперимента к «сейчас» и он
   * всплывёт наверх списка экспериментов, отсортированного по updatedAt. */
  backdateAt?: number;
}

function readFillQueue(): SeedFillJob[] {
  try {
    const raw = localStorage.getItem(FILL_QUEUE_KEY);
    return raw ? (JSON.parse(raw) as SeedFillJob[]) : [];
  } catch {
    return [];
  }
}

function writeFillQueue(jobs: SeedFillJob[]): void {
  localStorage.setItem(FILL_QUEUE_KEY, JSON.stringify(jobs));
}

function randomConfidence(): number {
  return Math.round((0.78 + Math.random() * 0.2) * 100) / 100;
}

async function generateAndStoreResult(experimentId: string, frame: Frame, talcThreshold: number, reviewed: boolean) {
  const seed = frame.source.kind === 'procedural' ? frame.source.seed : 0;
  const params = paramsForSeed(seed);
  const { mw, mh } = maskWorkingSize(frame.width, frame.height);
  // Чанкованная асинхронная генерация — не блокирует основной поток надолго во время сидирования.
  const data = await generateMaskDataChunked(seed, mw, mh, params);
  const maskId = genId('mask');
  await putMask({ id: maskId, frameId: frame.id, width: mw, height: mh, data });
  const metrics = calcMetrics({ width: mw, height: mh, data });
  const { oreClass, reason } = classifyFrame(metrics, talcThreshold);
  useExperimentsStore.getState().setFrameResult(experimentId, frame.id, {
    status: reviewed ? 'reviewed' : 'ready',
    maskId,
    autoMaskId: maskId,
    metrics,
    frameClass: oreClass,
    classReason: reason,
    confidence: randomConfidence(),
  });
}

/** Прямая правка меток времени завершённого демо-эксперимента, минуя now()-логику стора (для «архивных» сидов). */
function backdateExperiment(experimentId: string, at: number): void {
  useExperimentsStore.setState((s) => ({
    experiments: s.experiments.map((e) =>
      e.id === experimentId
        ? { ...e, createdAt: at, updatedAt: at, history: e.history.map((h) => ({ ...h, at })) }
        : e,
    ),
  }));
}

/**
 * Быстро строит кадр вместе с предварительным результатом (маска и метрики тестового разрешения),
 * не дожидаясь тяжёлой полноразмерной генерации. Это даёт немедленно ненулевые, реально посчитанные
 * метрики/класс — важно для графиков дашборда, которые иначе показывали бы пустые данные, пока
 * фоновая очередь не доберётся до этих кадров (что могло занимать десятки секунд).
 */
async function buildQuickFrame(
  index: number,
  name: string,
  targetClass: OreClass,
  talcThreshold: number,
  startSeed: number,
): Promise<{
  frame: Frame;
  maskId: string;
  metrics: ReturnType<typeof calcMetrics>;
  frameClass: OreClass;
  classReason: string;
  confidence: number;
}> {
  const match = findSeedAndMetricsForClass(targetClass, talcThreshold, startSeed);
  const frame = makeFrameStub(index, match.seed, name);
  const maskId = genId('mask');
  await putMask({ id: maskId, frameId: frame.id, width: match.width, height: match.height, data: match.data });
  return {
    frame,
    maskId,
    metrics: match.metrics,
    frameClass: match.oreClass,
    classReason: match.reason,
    confidence: randomConfidence(),
  };
}

/** In-flight guard: переживает двойной вызов эффекта в React StrictMode и повторные вызовы bootstrapApp. */
let seedingPromise: Promise<void> | null = null;
/** In-flight guard для фоновой доработки кадров — не должен запускаться параллельно сам с собой. */
let fillPromise: Promise<void> | null = null;
/**
 * «Поколение» фоновой доработки. Сброс демо-данных увеличивает счётчик, чтобы уже запущенный
 * (устаревший) цикл доработки заметил это на следующей итерации и тихо остановился, не перетирая
 * localStorage данными от уже удалённого набора экспериментов (иначе получаем гонку состояний
 * с новым циклом, запущенным поверх свежепересозданных данных).
 */
let fillEpoch = 0;

export async function seedIfEmpty(): Promise<void> {
  if (!seedingPromise) {
    seedingPromise = (async () => {
      const alreadySeeded = localStorage.getItem(SEEDED_FLAG) === 'true';
      const depositsCount = useDepositsStore.getState().deposits.length;
      const experimentsCount = useExperimentsStore.getState().experiments.length;

      if (alreadySeeded && depositsCount > 0) {
        // Структура уже создана (возможно, в прошлой вкладке/загрузке). Если фоновая доработка
        // кадров (маски/метрики) была прервана перезагрузкой — просто продолжаем с того места,
        // где остановились, вместо того чтобы пересоздавать всё заново с новыми id.
        void processFillQueue();
        return;
      }

      if (depositsCount > 0 || experimentsCount > 0) {
        // Флаг отсутствует (в т.ч. после апгрейда версии сида), но данные уже есть — пересобираем
        // с нуля, чтобы избежать частичных/устаревших/задвоенных данных.
        fillEpoch += 1; // отменяем любой уже запущенный (устаревший) цикл доработки кадров
        fillPromise = null;
        await clearAllData();
        useExperimentsStore.setState({ experiments: [] });
        useDepositsStore.setState({ deposits: [] });
        useMlQueueStore.setState({ queue: [] });
        localStorage.removeItem(FILL_QUEUE_KEY);
      }

      // Структурная часть (месторождения, эксперименты, кадры) быстрая: метрики/класс кадров
      // считаются сразу на тестовом разрешении. Как только она завершена и id стабилизировались,
      // сразу помечаем сид как выполненный: полноразмерная перегенерация масок для части кадров
      // продолжится в фоне и переживёт перезагрузку.
      await runSeedStructure();
      localStorage.setItem(SEEDED_FLAG, 'true');
      void processFillQueue();
    })();
  }
  return seedingPromise;
}

export async function resetDemoData(): Promise<void> {
  seedingPromise = null;
  fillPromise = null;
  fillEpoch += 1;
  await clearAllData();
  useExperimentsStore.setState({ experiments: [] });
  useDepositsStore.setState({ deposits: [] });
  useMlQueueStore.setState({ queue: [] });
  localStorage.removeItem(SEEDED_FLAG);
  localStorage.removeItem(FILL_QUEUE_KEY);
  await seedIfEmpty();
}

/** Обрабатывает накопленную очередь «доотсеивания» кадров — переживает перезагрузку страницы. */
async function processFillQueue(): Promise<void> {
  if (fillPromise) return fillPromise;
  const myEpoch = fillEpoch;
  fillPromise = (async () => {
    let jobs = readFillQueue();
    while (jobs.length > 0) {
      if (myEpoch !== fillEpoch) return; // данные сброшены/пересозданы — этот цикл устарел, тихо выходим

      const job = jobs[0];
      const experiment = useExperimentsStore.getState().getExperiment(job.experimentId);
      const frame = experiment?.frames.find((f) => f.id === job.frameId);
      if (experiment && frame) {
        try {
          await generateAndStoreResult(job.experimentId, frame, job.threshold, job.reviewed);
          if (job.backdateAt) backdateExperiment(job.experimentId, job.backdateAt);
        } catch {
          // Пропускаем повреждённое задание, чтобы не блокировать остальную очередь.
        }
      }

      if (myEpoch !== fillEpoch) return; // могли устареть, пока ждали генерацию маски выше

      jobs = jobs.slice(1);
      writeFillQueue(jobs);
    }
  })().finally(() => {
    if (myEpoch === fillEpoch) fillPromise = null;
  });
  return fillPromise;
}

/** 2-3 «архивных» эксперимента с разнесёнными по прошлым неделям датами — питают динамику на дашборде. */
async function seedHistoricalExperiments(deposit: Deposit, seedBase: number, jobs: SeedFillJob[]): Promise<void> {
  const expStore = useExperimentsStore.getState();
  const configs: Array<{ weeksAgo: number; suffix: string; targets: OreClass[] }> = [
    { weeksAgo: 3, suffix: 'архив, 3 нед. назад', targets: ['routine', 'routine', 'talc'] },
    { weeksAgo: 2, suffix: 'архив, 2 нед. назад', targets: ['routine', 'hard'] },
    { weeksAgo: 1, suffix: 'архив, 1 нед. назад', targets: ['talc', 'talc', 'routine'] },
  ];

  for (const cfg of configs) {
    const exp = expStore.createExperiment({
      title: `Партия ${seedBase + cfg.weeksAgo} — ${cfg.suffix}`,
      depositId: deposit.id,
      author: DEMO_AUTHOR,
    });
    const quickFrames = await Promise.all(
      cfg.targets.map((target, i) =>
        buildQuickFrame(
          i,
          `Шлиф ${i + 1}`,
          target,
          deposit.talcThreshold,
          seedBase * 100 + cfg.weeksAgo * 777 + i * 43,
        ),
      ),
    );
    expStore.addFrames(exp.id, quickFrames.map((q) => q.frame));
    const backdateAt = Date.now() - cfg.weeksAgo * WEEK_MS - Math.floor(Math.random() * WEEK_MS * 0.25);
    quickFrames.forEach((q) => {
      expStore.setFrameResult(exp.id, q.frame.id, {
        status: 'reviewed',
        maskId: q.maskId,
        autoMaskId: q.maskId,
        metrics: q.metrics,
        frameClass: q.frameClass,
        classReason: q.classReason,
        confidence: q.confidence,
      });
      jobs.push({ experimentId: exp.id, frameId: q.frame.id, threshold: deposit.talcThreshold, reviewed: true, backdateAt });
    });
    expStore.completeExperiment(exp.id, DEMO_AUTHOR);
    backdateExperiment(exp.id, backdateAt);
  }
}

interface DepositSeedSpec {
  name: string;
  code: string;
  oreCluster: string;
  region: string;
  oreTypes: string[];
  talcThreshold: number;
  notes: string;
  minerals: Mineral[];
  reserves?: DepositReserves;
  metalGrades?: MetalGrades;
}

/** Профиль минералов, типичный для сульфидных медно-никелевых руд (Норильск/Кольский п-ов). */
function cuNiMinerals(): Mineral[] {
  return minerals([
    ['Пирротин', 'sulfide', '#D9A441'],
    ['Пентландит', 'sulfide', '#E8B94A'],
    ['Халькопирит', 'sulfide', '#C98A2B'],
    ['Тальк', 'talc', '#2F6F63'],
    ['Серпентин', 'gangue', '#9C9C9E'],
    ['Оливин', 'gangue', '#8B8B96'],
  ]);
}

const TALNAKH_RESERVES: DepositReserves = {
  provenProbable: '622,8 млн т',
  measuredIndicated: '1 546,3 млн т',
  balance: '1 979,6 млн т',
};
const TALNAKH_GRADES: MetalGrades = { nickel: '2,22%', copper: '3,54%', mpg: '10,27 г/т' };

const NORILSK1_RESERVES: DepositReserves = {
  provenProbable: '40,3 млн т',
  measuredIndicated: '156,6 млн т',
  balance: '156,6 млн т',
};
const NORILSK1_GRADES: MetalGrades = { nickel: '0,18%', copper: '0,18%', mpg: '3,91 г/т' };

const KOLA_EAST_RESERVES: DepositReserves = {
  provenProbable: '79,7 млн т',
  measuredIndicated: '315,6 млн т',
  balance: '457,8 млн т',
};
const KOLA_GRADES: MetalGrades = {
  nickel: '3,1 млн т (баланс. запасы металла по узлу)',
  copper: '1,5 млн т (баланс. запасы металла по узлу)',
};

function depositSeedSpecs(): DepositSeedSpec[] {
  return [
    {
      name: 'Октябрьское',
      code: 'OKT',
      oreCluster: 'Талнахский рудный узел',
      region: 'Красноярский край, Норильский промышленный район',
      oreTypes: ['богатые', 'медистые', 'вкрапленные'],
      talcThreshold: 0.08,
      notes: 'Богатые сплошные руды, повышенный риск оталькования.',
      minerals: cuNiMinerals(),
      reserves: TALNAKH_RESERVES,
      metalGrades: TALNAKH_GRADES,
    },
    {
      name: 'Талнахское',
      code: 'TAL',
      oreCluster: 'Талнахский рудный узел',
      region: 'Красноярский край, Норильский промышленный район',
      oreTypes: ['богатые', 'медистые', 'вкрапленные'],
      talcThreshold: 0.1,
      notes: 'Сульфидные медно-никелевые руды, участок Талнахский.',
      minerals: cuNiMinerals(),
      reserves: TALNAKH_RESERVES,
      metalGrades: TALNAKH_GRADES,
    },
    {
      name: 'Норильск-1',
      code: 'NR1',
      oreCluster: 'Норильский рудный узел',
      region: 'Красноярский край, Норильский промышленный район (северная часть)',
      oreTypes: ['вкрапленные'],
      talcThreshold: 0.12,
      notes: 'Вкраплённые руды, тонкая вкрапленность сульфидов, северная часть месторождения.',
      minerals: minerals([
        ['Халькопирит', 'sulfide', '#C98A2B'],
        ['Пентландит', 'sulfide', '#E8B94A'],
        ['Тальк', 'talc', '#2F6F63'],
        ['Оливин', 'gangue', '#8B8B96'],
      ]),
      reserves: NORILSK1_RESERVES,
      metalGrades: NORILSK1_GRADES,
    },
    {
      name: 'Ждановское',
      code: 'ZHD',
      oreCluster: 'Кольская ГМК — Восточный рудный узел',
      region: 'Мурманская область, между п. Никель и г. Заполярный',
      oreTypes: ['сульфидные медно-никелевые'],
      talcThreshold: 0.11,
      notes: 'Разработка Восточного рудного узла ведётся с 1960 года.',
      minerals: cuNiMinerals(),
      reserves: KOLA_EAST_RESERVES,
      metalGrades: KOLA_GRADES,
    },
    {
      name: 'Заполярное',
      code: 'ZAP',
      oreCluster: 'Кольская ГМК — Восточный рудный узел',
      region: 'Мурманская область, между п. Никель и г. Заполярный',
      oreTypes: ['сульфидные медно-никелевые'],
      talcThreshold: 0.1,
      notes: 'Разработка Восточного рудного узла ведётся с 1960 года.',
      minerals: cuNiMinerals(),
      reserves: KOLA_EAST_RESERVES,
      metalGrades: KOLA_GRADES,
    },
    {
      name: 'Тундровое',
      code: 'TND',
      oreCluster: 'Кольская ГМК — Восточный рудный узел',
      region: 'Мурманская область, между п. Никель и г. Заполярный',
      oreTypes: ['сульфидные медно-никелевые'],
      talcThreshold: 0.09,
      notes: 'Разработка Восточного рудного узла ведётся с 1960 года.',
      minerals: cuNiMinerals(),
      reserves: KOLA_EAST_RESERVES,
      metalGrades: KOLA_GRADES,
    },
    {
      name: 'Спутник',
      code: 'SPT',
      oreCluster: 'Кольская ГМК — Восточный рудный узел',
      region: 'Мурманская область, между п. Никель и г. Заполярный',
      oreTypes: ['сульфидные медно-никелевые'],
      talcThreshold: 0.1,
      notes: 'Разработка Восточного рудного узла ведётся с 1960 года.',
      minerals: cuNiMinerals(),
      reserves: KOLA_EAST_RESERVES,
      metalGrades: KOLA_GRADES,
    },
    {
      name: 'Верхнее',
      code: 'VRH',
      oreCluster: 'Кольская ГМК — Восточный рудный узел',
      region: 'Мурманская область, между п. Никель и г. Заполярный',
      oreTypes: ['сульфидные медно-никелевые'],
      talcThreshold: 0.12,
      notes: 'Разработка Восточного рудного узла ведётся с 1960 года.',
      minerals: cuNiMinerals(),
      reserves: KOLA_EAST_RESERVES,
      metalGrades: KOLA_GRADES,
    },
    {
      name: 'Котсельваара-Каммикиви',
      code: 'KTK',
      oreCluster: 'Кольская ГМК — Западный рудный узел',
      region: 'Мурманская область, между п. Никель и г. Заполярный',
      oreTypes: ['сульфидные медно-никелевые'],
      talcThreshold: 0.1,
      notes: 'Разработка Западного рудного узла ведётся с 1930-х годов.',
      minerals: cuNiMinerals(),
      reserves: KOLA_EAST_RESERVES,
      metalGrades: KOLA_GRADES,
    },
    {
      name: 'Семилетка',
      code: 'SML',
      oreCluster: 'Кольская ГМК — Западный рудный узел',
      region: 'Мурманская область, между п. Никель и г. Заполярный',
      oreTypes: ['сульфидные медно-никелевые'],
      talcThreshold: 0.1,
      notes: 'Разработка Западного рудного узла ведётся с 1930-х годов.',
      minerals: cuNiMinerals(),
      reserves: KOLA_EAST_RESERVES,
      metalGrades: KOLA_GRADES,
    },
    {
      name: 'Быстринское',
      code: 'BYS',
      oreCluster: 'Быстринский ГОК',
      region: 'Забайкальский край, 16 км восточнее с. Газимурский Завод',
      oreTypes: ['золото-железо-медные'],
      talcThreshold: 0.08,
      notes: 'Скарновое золото-железо-медное месторождение, тальк не является типичным минералом (порог занижен намеренно для демонстрации).',
      minerals: minerals([
        ['Халькопирит', 'sulfide', '#C98A2B'],
        ['Пирит', 'sulfide', '#B8860B'],
        ['Магнетит', 'gangue', '#5C5C63'],
        ['Золото', 'other', '#D4AF37'],
        ['Тальк', 'talc', '#2F6F63'],
      ]),
      reserves: { balance: '300,9 млн т' },
      metalGrades: {
        copper: '2,1 млн т (баланс.)',
        gold: '8,1 млн тр. унций (баланс.)',
        silver: '67,5 млн тр. унций (баланс.)',
        iron: '36,9 млн т (баланс.)',
      },
    },
    {
      name: 'Nkomati',
      code: 'NKM',
      oreCluster: 'Nkomati',
      region: 'ЮАР, провинция Мпумаланга',
      oreTypes: ['вкрапленные сульфидные медно-никелевые', 'хромитовые'],
      talcThreshold: 0.1,
      notes: 'Вкраплённые сульфидные Cu-Ni руды с хромитовой минерализацией.',
      minerals: minerals([
        ['Пирротин', 'sulfide', '#D9A441'],
        ['Пентландит', 'sulfide', '#E8B94A'],
        ['Халькопирит', 'sulfide', '#C98A2B'],
        ['Хромит', 'other', '#4B3F72'],
        ['Тальк', 'talc', '#2F6F63'],
        ['Серпентин', 'gangue', '#9C9C9E'],
      ]),
      reserves: { provenProbable: '0,9 млн т', measuredIndicated: '168,5 млн т' },
      metalGrades: {
        nickel: '590 тыс. т (оцен.+выявл.)',
        copper: '227 тыс. т (оцен.+выявл.)',
        cobalt: '29 тыс. т (оцен.+выявл.)',
        mpg: '4,9 млн тр. унций (оцен.+выявл.)',
      },
    },
  ];
}

async function runSeedStructure(): Promise<void> {
  const depositsStore = useDepositsStore.getState();
  const jobs: SeedFillJob[] = [];

  const seeded = depositSeedSpecs().map((spec) =>
    depositsStore.addDeposit({
      name: spec.name,
      code: spec.code,
      talcThreshold: spec.talcThreshold,
      notes: spec.notes,
      minerals: spec.minerals,
      oreCluster: spec.oreCluster,
      region: spec.region,
      oreTypes: spec.oreTypes,
      reserves: spec.reserves,
      metalGrades: spec.metalGrades,
      updatedBy: DEMO_AUTHOR,
    }),
  );

  const byName = new Map(seeded.map((d) => [d.name, d]));
  const talnakh = byName.get('Талнахское')!;
  const oktyabrskoe = byName.get('Октябрьское')!;
  const zhdanov = byName.get('Ждановское')!;

  const expStore = useExperimentsStore.getState();

  // Эксперимент 0: черновик без кадров — создаём первым, чтобы после него ещё что-то обновлялось
  // и он не всплывал наверх списка экспериментов (который сортируется по updatedAt по убыванию).
  expStore.createExperiment({
    title: 'Новая партия — черновик',
    depositId: talnakh.id,
    author: DEMO_AUTHOR,
  });

  // Эксперимент 1: завершён, 5 кадров, рядовая руда
  const exp1 = expStore.createExperiment({
    title: 'Партия 1042 — смена 2, участок Талнахский',
    depositId: talnakh.id,
    author: DEMO_AUTHOR,
  });
  const exp1Quick = await Promise.all(
    Array.from({ length: 5 }, (_, i) =>
      buildQuickFrame(i, `Шлиф ${i + 1}`, 'routine', talnakh.talcThreshold, 1000 + i * 137),
    ),
  );
  expStore.addFrames(exp1.id, exp1Quick.map((q) => q.frame));
  exp1Quick.forEach((q) => {
    expStore.setFrameResult(exp1.id, q.frame.id, {
      status: 'reviewed',
      maskId: q.maskId,
      autoMaskId: q.maskId,
      metrics: q.metrics,
      frameClass: q.frameClass,
      classReason: q.classReason,
      confidence: q.confidence,
    });
    jobs.push({ experimentId: exp1.id, frameId: q.frame.id, threshold: talnakh.talcThreshold, reviewed: true });
  });
  expStore.completeExperiment(exp1.id, DEMO_AUTHOR);

  // Эксперимент 2: расхождения между кадрами, 4 кадра разных классов
  const exp2 = expStore.createExperiment({
    title: 'Партия 2077 — входной контроль',
    depositId: oktyabrskoe.id,
    author: DEMO_AUTHOR,
  });
  const targets2: OreClass[] = ['routine', 'routine', 'hard', 'talc'];
  const exp2Quick = await Promise.all(
    targets2.map((t, i) => buildQuickFrame(i, `Шлиф ${i + 1}`, t, oktyabrskoe.talcThreshold, 5000 + i * 251)),
  );
  expStore.addFrames(exp2.id, exp2Quick.map((q) => q.frame));
  exp2Quick.forEach((q) => {
    expStore.setFrameResult(exp2.id, q.frame.id, {
      status: 'ready',
      maskId: q.maskId,
      autoMaskId: q.maskId,
      metrics: q.metrics,
      frameClass: q.frameClass,
      classReason: q.classReason,
      confidence: q.confidence,
    });
    jobs.push({ experimentId: exp2.id, frameId: q.frame.id, threshold: oktyabrskoe.talcThreshold, reviewed: false });
  });

  // Эксперимент 3: в работе, кадры ещё в очереди — оживает после запуска приложения (демо очереди ML)
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

  // Архивные (backdated) эксперименты по нескольким месторождениям — нужны, чтобы графики динамики
  // на дашборде сразу показывали несколько точек по неделям с реальной средней долей талька.
  await seedHistoricalExperiments(talnakh, 200, jobs);
  await seedHistoricalExperiments(oktyabrskoe, 300, jobs);
  await seedHistoricalExperiments(zhdanov, 400, jobs);

  writeFillQueue(jobs);
}
