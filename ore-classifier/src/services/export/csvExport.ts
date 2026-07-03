import Papa from 'papaparse';
import type { Deposit, Experiment } from '../../types/models';
import { ORE_CLASS_META } from '../../theme/palette';
import { formatDateTime, formatPercent } from '../../utils/format';

export function buildCsv(experiment: Experiment, deposit?: Deposit): string {
  const rows = experiment.frames.map((f) => {
    const effectiveClass = f.manualClassOverride ?? f.frameClass;
    return {
      Эксперимент: experiment.title,
      Месторождение: deposit?.name ?? '',
      Кадр: f.name,
      Опорный: f.isReference ? 'да' : 'нет',
      Статус: f.status,
      Класс: effectiveClass ? ORE_CLASS_META[effectiveClass].label : '',
      'Правка вручную': f.manualClassOverride ? 'да' : 'нет',
      'Доля талька': f.metrics ? formatPercent(f.metrics.talcFraction) : '',
      'Доля сульфидов': f.metrics ? formatPercent(f.metrics.sulfideFraction) : '',
      'Крупные срастания': f.metrics ? formatPercent(f.metrics.coarseFraction) : '',
      'Тонкие срастания': f.metrics ? formatPercent(f.metrics.fineFraction) : '',
      Уверенность: f.confidence !== undefined ? `${Math.round(f.confidence * 100)}%` : '',
      Обновлён: formatDateTime(f.updatedAt),
    };
  });
  const csv = Papa.unparse(rows, { delimiter: ';' });
  return `\uFEFF${csv}`;
}

export function downloadCsv(experiment: Experiment, deposit?: Deposit): void {
  const csv = buildCsv(experiment, deposit);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${experiment.title.replace(/[^\wа-яА-Я0-9-]+/gi, '_')}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
