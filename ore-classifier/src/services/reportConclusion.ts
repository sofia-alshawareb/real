// Автогенерация текстового вывода отчёта по усреднённым метрикам эксперимента.

import { ORE_CLASS_META } from '../theme/palette';
import { pct } from './rulesEngine';
import type { Experiment } from '../types/models';

export function buildConclusionText(experiment: Experiment): string {
  const framesWithMetrics = experiment.frames.filter((f) => f.metrics);
  if (framesWithMetrics.length === 0 || !experiment.experimentClass) {
    return '';
  }
  const sum = framesWithMetrics.reduce(
    (acc, f) => ({
      talc: acc.talc + f.metrics!.talcFraction,
      coarse: acc.coarse + f.metrics!.coarseFraction,
      fine: acc.fine + f.metrics!.fineFraction,
    }),
    { talc: 0, coarse: 0, fine: 0 },
  );
  const n = framesWithMetrics.length;
  const avgTalc = sum.talc / n;
  const avgCoarse = sum.coarse / n;
  const avgFine = sum.fine / n;
  const classLabel = ORE_CLASS_META[experiment.experimentClass].label.toLowerCase();
  const dominant =
    avgCoarse >= avgFine
      ? `преобладание обычных срастаний — ${pct(avgCoarse)}`
      : `преобладание тонких срастаний — ${pct(avgFine)}`;
  return `Руда классифицирована как ${classLabel}: содержание талька — ${pct(avgTalc)}, ${dominant}.`;
}
