import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import { BarChart } from '@mui/x-charts/BarChart';
import { LineChart } from '@mui/x-charts/LineChart';
import { PieChart } from '@mui/x-charts/PieChart';
import AddIcon from '@mui/icons-material/Add';
import DashboardIcon from '@mui/icons-material/Dashboard';
import { useExperimentsStore } from '../../stores/experimentsStore';
import { useDepositsStore } from '../../stores/depositsStore';
import { EmptyState } from '../../components/EmptyState';
import { ORE_CLASS_META } from '../../theme/palette';
import type { Experiment, OreClass } from '../../types/models';

function computeWeeklyData(experiments: Experiment[]) {
  const byWeek = new Map<string, { count: number; talcSum: number; talcN: number }>();
  experiments.forEach((e) => {
    const week = new Date(e.createdAt);
    const key = `${week.getFullYear()}-${String(Math.ceil(week.getDate() / 7)).padStart(2, '0')}-${week.getMonth() + 1}`;
    const entry = byWeek.get(key) ?? { count: 0, talcSum: 0, talcN: 0 };
    entry.count++;
    e.frames.forEach((f) => {
      if (f.metrics) {
        entry.talcSum += f.metrics.talcFraction;
        entry.talcN++;
      }
    });
    byWeek.set(key, entry);
  });
  return Array.from(byWeek.entries())
    .sort((a, b) => (a[0] > b[0] ? 1 : -1))
    .map(([week, v]) => ({ week, count: v.count, avgTalc: v.talcN ? (v.talcSum / v.talcN) * 100 : 0 }));
}

function KpiCard({ label, value }: { label: string; value: string | number }) {
  return (
    <Paper variant="outlined" sx={{ p: 2.5, flex: 1, minWidth: 160 }}>
      <Typography variant="overline" sx={{
        color: "text.secondary"
      }}>
        {label}
      </Typography>
      <Typography variant="h4">{value}</Typography>
    </Paper>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();
  const experiments = useExperimentsStore((s) => s.experiments);
  const deposits = useDepositsStore((s) => s.deposits);

  const totalFrames = experiments.reduce((acc, e) => acc + e.frames.length, 0);
  const discrepanciesCount = experiments.filter((e) => e.status === 'has_discrepancies').length;

  const classByDeposit = useMemo(() => {
    return deposits.map((d) => {
      const depExperiments = experiments.filter((e) => e.depositId === d.id);
      const counts: Record<OreClass, number> = { routine: 0, hard: 0, talc: 0 };
      depExperiments.forEach((e) => {
        e.frames.forEach((f) => {
          const cls = f.manualClassOverride ?? f.frameClass;
          if (cls) counts[cls]++;
        });
      });
      return { deposit: d.name, ...counts };
    });
  }, [deposits, experiments]);

  const weeklyByDeposit = useMemo(() => {
    return deposits
      .map((d) => ({ deposit: d, data: computeWeeklyData(experiments.filter((e) => e.depositId === d.id)) }))
      .filter((entry) => entry.data.length > 0);
  }, [deposits, experiments]);

  const modeCounts = useMemo(() => {
    const counts = { ml: 0, manual: 0, mixed: 0 };
    experiments.forEach((e) => counts[e.analysisMode]++);
    return counts;
  }, [experiments]);

  if (experiments.length === 0) {
    return (
      <Paper variant="outlined">
        <EmptyState
          icon={<DashboardIcon sx={{ fontSize: 56, opacity: 0.4 }} />}
          title="Пока нет данных для дашборда"
          description="Создайте первый эксперимент, чтобы увидеть статистику по классам руды и месторождениям."
          action={
            <Button variant="contained" startIcon={<AddIcon />} onClick={() => navigate('/experiments/new')}>
              Создать первый эксперимент
            </Button>
          }
        />
      </Paper>
    );
  }

  return (
    <Box>
      <Typography variant="h5" sx={{
        mb: 2
      }}>
        Дашборд
      </Typography>
      <Stack
        direction="row"
        spacing={2}
        sx={{
          mb: 3,
          flexWrap: "wrap"
        }}>
        <KpiCard label="Всего экспериментов" value={experiments.length} />
        <KpiCard label="Всего кадров" value={totalFrames} />
        <KpiCard label="С расхождениями" value={discrepanciesCount} />
        <KpiCard label="Месторождений" value={deposits.length} />
      </Stack>
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, md: 7 }}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Классы руды по месторождениям
            </Typography>
            <BarChart
              layout="horizontal"
              height={Math.max(300, classByDeposit.length * 34)}
              margin={{ left: 170 }}
              dataset={classByDeposit}
              yAxis={[{ scaleType: 'band', dataKey: 'deposit', width: 160 }]}
              xAxis={[{}]}
              series={[
                { dataKey: 'routine', label: ORE_CLASS_META.routine.label, color: ORE_CLASS_META.routine.color, stack: 'total' },
                { dataKey: 'hard', label: ORE_CLASS_META.hard.label, color: ORE_CLASS_META.hard.color, stack: 'total' },
                { dataKey: 'talc', label: ORE_CLASS_META.talc.label, color: ORE_CLASS_META.talc.color, stack: 'total' },
              ]}
            />
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, md: 5 }}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Режимы анализа
            </Typography>
            <PieChart
              height={300}
              series={[
                {
                  data: [
                    { id: 'ml', value: modeCounts.ml, label: 'ML', color: '#3B5B7C' },
                    { id: 'manual', value: modeCounts.manual, label: 'Ручной', color: '#00695C' },
                    { id: 'mixed', value: modeCounts.mixed, label: 'Смешанный', color: '#E65100' },
                  ],
                  innerRadius: 50,
                },
              ]}
            />
          </Paper>
        </Grid>
      </Grid>

      <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
        По месторождениям
      </Typography>
      {weeklyByDeposit.length === 0 ? (
        <Paper variant="outlined" sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
          Пока недостаточно данных по кадрам, чтобы построить динамику по месторождениям.
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {weeklyByDeposit.map(({ deposit, data }) => (
            <Grid key={deposit.id} size={{ xs: 12, md: 6 }}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  {deposit.name} — динамика экспериментов и средняя доля талька
                </Typography>
                <LineChart
                  height={260}
                  dataset={data}
                  xAxis={[{ scaleType: 'point', dataKey: 'week' }]}
                  series={[
                    { dataKey: 'count', label: 'Экспериментов', color: '#3B5B7C' },
                    { dataKey: 'avgTalc', label: 'Средняя доля талька, %', color: '#6A1B9A' },
                  ]}
                />
              </Paper>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
}
