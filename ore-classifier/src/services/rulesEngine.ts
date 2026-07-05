// Экспертная логика классификации руды. Чистые функции — переиспользуются в what-if слайдере.

import type { Frame, FrameMetrics, OreClass } from '../types/models';

export function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

export interface ClassificationResult {
  oreClass: OreClass;
  reason: string;
}

function fmtPixels(n: number): string {
  return n.toLocaleString('ru-RU');
}

export function classifyFrame(m: FrameMetrics, talcThreshold: number): ClassificationResult {
  if (m.talcFraction > talcThreshold) {
    return {
      oreClass: 'talc',
      reason: `Доля талька ${pct(m.talcFraction)} превышает порог ${pct(talcThreshold)} — оталькованная руда`,
    };
  }
  const coarsePx = m.coarsePixels;
  const finePx = m.finePixels;
  const fineDominates =
    coarsePx != null && finePx != null ? finePx > coarsePx : m.fineFraction > m.coarseFraction;
  if (fineDominates) {
    const coarseLabel = coarsePx != null ? fmtPixels(coarsePx) : pct(m.coarseFraction);
    const fineLabel = finePx != null ? fmtPixels(finePx) : pct(m.fineFraction);
    return {
      oreClass: 'hard',
      reason:
        coarsePx != null && finePx != null
          ? `Доля талька ${pct(m.talcFraction)} ≤ порога, площадь тонких срастаний (${fineLabel} px) больше обычных (${coarseLabel} px) — труднообогатимая руда`
          : `Доля талька ${pct(m.talcFraction)} ≤ порога, преобладают тонкие срастания (${fineLabel} против ${coarseLabel}) — труднообогатимая руда`,
    };
  }
  const coarseLabel = coarsePx != null ? fmtPixels(coarsePx) : pct(m.coarseFraction);
  const fineLabel = finePx != null ? fmtPixels(finePx) : pct(m.fineFraction);
  return {
    oreClass: 'routine',
    reason:
      coarsePx != null && finePx != null
        ? `Доля талька ${pct(m.talcFraction)} ≤ порога, площадь обычных срастаний (${coarseLabel} px) не меньше тонких (${fineLabel} px) — рядовая руда`
        : `Доля талька ${pct(m.talcFraction)} ≤ порога, преобладают обычные срастания (${coarseLabel} против ${fineLabel}) — рядовая руда`,
  };
}

export interface AggregateResult {
  experimentClass?: OreClass;
  derivation?: 'majority' | 'reference' | 'manual';
  hasDiscrepancies: boolean;
}

function effectiveClass(frame: Frame): OreClass | undefined {
  return frame.manualClassOverride ?? frame.frameClass;
}

export function aggregateExperiment(frames: Frame[], referenceFrameId?: string): AggregateResult {
  const classified = frames.filter((f) => effectiveClass(f) !== undefined);
  if (classified.length === 0) {
    return { hasDiscrepancies: false };
  }
  const counts = new Map<OreClass, number>();
  for (const f of classified) {
    const c = effectiveClass(f)!;
    counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  const distinctClasses = Array.from(counts.keys());
  const hasDiscrepancies = distinctClasses.length > 1;

  if (!hasDiscrepancies) {
    return { experimentClass: distinctClasses[0], derivation: 'majority', hasDiscrepancies: false };
  }

  let maxCount = -1;
  let winners: OreClass[] = [];
  for (const [cls, count] of counts) {
    if (count > maxCount) {
      maxCount = count;
      winners = [cls];
    } else if (count === maxCount) {
      winners.push(cls);
    }
  }

  if (winners.length === 1) {
    return { experimentClass: winners[0], derivation: 'majority', hasDiscrepancies: true };
  }

  const referenceFrame = frames.find((f) => f.id === referenceFrameId);
  const referenceClass = referenceFrame ? effectiveClass(referenceFrame) : undefined;
  if (referenceClass) {
    return { experimentClass: referenceClass, derivation: 'reference', hasDiscrepancies: true };
  }
  return { experimentClass: winners[0], derivation: 'majority', hasDiscrepancies: true };
}
