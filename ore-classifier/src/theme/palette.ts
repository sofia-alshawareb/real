import type { OreClass, FrameStatus, ExperimentStatus } from '../types/models';

export interface OreClassMeta {
  label: string;
  color: string;
  bg: string;
  icon: 'check_circle' | 'settings' | 'layers';
}

export const ORE_CLASS_META: Record<OreClass, OreClassMeta> = {
  routine: { label: 'Рядовая', color: '#2E7D32', bg: '#E8F5E9', icon: 'check_circle' },
  hard: { label: 'Труднообогатимая', color: '#4527A0', bg: '#EDE7F6', icon: 'settings' },
  talc: { label: 'Оталькованная', color: '#6A1B9A', bg: '#F3E5F5', icon: 'layers' },
};

/** Классы разметки маски шлифа: обычные/тонкие срастания, тальк, нерудная матрица. */
export type MaskClassKey = 'coarse' | 'fine' | 'talc' | 'matrix';

export interface MaskClassMeta {
  label: string;
  color: string;
  value: number;
}

export const MASK_CLASSES: Record<MaskClassKey, MaskClassMeta> = {
  coarse: { label: 'Обычные срастания', color: '#2E7D32', value: 1 }, // зелёный
  fine: { label: 'Тонкие срастания', color: '#C62828', value: 2 }, // красный
  talc: { label: 'Тальк', color: '#1565C0', value: 3 }, // синий
  matrix: { label: 'Нерудная матрица', color: '#9E9E9E', value: 4 }, // серый
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
