import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import Typography from '@mui/material/Typography';
import type { Experiment } from '../../types/models';
import { ORE_CLASS_META } from '../../theme/palette';

export function DiscrepancyBanner({ experiment }: { experiment: Experiment }) {
  if (experiment.status !== 'has_discrepancies') return null;

  const classified = experiment.frames.filter((f) => f.manualClassOverride ?? f.frameClass);
  const groups = new Map<string, number>();
  for (const f of classified) {
    const cls = (f.manualClassOverride ?? f.frameClass)!;
    groups.set(cls, (groups.get(cls) ?? 0) + 1);
  }
  const summary = Array.from(groups.entries())
    .map(([cls, count]) => `${ORE_CLASS_META[cls as keyof typeof ORE_CLASS_META].label} — ${count}`)
    .join(', ');

  return (
    <Alert severity="warning" sx={{ mb: 2 }}>
      <AlertTitle>Классы кадров расходятся</AlertTitle>
      <Typography variant="body2">
        {summary}. Итоговый класс определён {experiment.classDerivation === 'reference' ? 'по опорному кадру' : 'по большинству кадров'} —
        проверьте расхождения перед завершением эксперимента.
      </Typography>
    </Alert>
  );
}
