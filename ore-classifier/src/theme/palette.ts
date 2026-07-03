import type { OreClass, MineralRole, FrameStatus, ExperimentStatus } from '../types/models';

export interface OreClassMeta {
  label: string;
  color: string;
  bg: string;
  icon: 'check_circle' | 'warning' | 'layers';
}

export const ORE_CLASS_META: Record<OreClass, OreClassMeta> = {
  routine: { label: 'Рядовая', color: '#2E7D32', bg: '#E8F5E9', icon: 'check_circle' },
  hard: { label: 'Труднообогатимая', color: '#E65100', bg: '#FFF3E0', icon: 'warning' },
  talc: { label: 'Оталькованная', color: '#6A1B9A', bg: '#F3E5F5', icon: 'layers' },
};

export interface MaskClassMeta {
  label: string;
  color: string;
  value: number;
}

export const MASK_CLASSES: Record<Exclude<MineralRole, 'other'>, MaskClassMeta> = {
  sulfide: { label: 'Сульфид', color: '#FFB300', value: 1 },
  gangue: { label: 'Нерудная матрица', color: '#9E9E9E', value: 2 },
  talc: { label: 'Тальк', color: '#00695C', value: 3 },
};

export const MASK_VALUE_TO_ROLE: Record<number, Exclude<MineralRole, 'other'>> = {
  1: 'sulfide',
  2: 'gangue',
  3: 'talc',
};

export interface StatusMeta {
  label: string;
  color: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
}

export const FRAME_STATUS_META: Record<FrameStatus, StatusMeta> = {
  queued: { label: 'В очереди', color: 'default' },
  processing: { label: 'Обрабатывается', color: 'info' },
  ml_unavailable: { label: 'ML недоступен', color: 'warning' },
  manual_only: { label: 'Ручная разметка', color: 'secondary' },
  ready: { label: 'Обработано', color: 'primary' },
  reviewed: { label: 'Подтверждено', color: 'success' },
  segmentation_edited: { label: 'Сегментация правлена', color: 'secondary' },
};

export const EXPERIMENT_STATUS_META: Record<ExperimentStatus, StatusMeta> = {
  draft: { label: 'Черновик', color: 'default' },
  in_progress: { label: 'В работе', color: 'info' },
  has_discrepancies: { label: 'Есть расхождения', color: 'warning' },
  completed: { label: 'Завершён', color: 'success' },
  reported: { label: 'Отчёт выгружен', color: 'primary' },
};

export const COARSE_GRAIN_UM = 70;
