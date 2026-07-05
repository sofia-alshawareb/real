import { useDepositsStore } from '../stores/depositsStore';
import { genId } from '../stores/experimentsStore';
import type { DepositReserves, MetalGrades, Mineral } from '../types/models';

const CATALOG_AUTHOR = 'Справочник';

function minerals(list: Array<[string, Mineral['role'], string]>): Mineral[] {
  return list.map(([name, role, colorHex]) => ({ id: genId('min'), name, role, colorHex }));
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

interface DepositCatalogSpec {
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

export function depositCatalogSpecs(): DepositCatalogSpec[] {
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
      notes: 'Скарновое золото-железо-медное месторождение.',
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

/** Заполняет справочник месторождений, если он пуст (без демо-экспериментов). */
export function seedDepositsIfEmpty(): void {
  const store = useDepositsStore.getState();
  if (store.deposits.length > 0) return;

  for (const spec of depositCatalogSpecs()) {
    store.addDeposit({
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
      updatedBy: CATALOG_AUTHOR,
    });
  }
}
