import { useMemo, useState } from 'react';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Slider from '@mui/material/Slider';
import Collapse from '@mui/material/Collapse';
import TuneIcon from '@mui/icons-material/Tune';
import DescriptionIcon from '@mui/icons-material/Description';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlineOutlined';
import { OreClassBadge } from '../../components/OreClassBadge';
import type { Experiment } from '../../types/models';
import { classifyFrame, aggregateExperiment } from '../../services/rulesEngine';
import { formatPercent } from '../../utils/format';

const DERIVATION_LABEL: Record<string, string> = {
  majority: 'по большинству кадров',
  reference: 'по опорному кадру',
  manual: 'задан вручную',
};

interface ExperimentSummaryPanelProps {
  experiment: Experiment;
  talcThreshold: number;
  onApplyThresholdToDeposit: (newThreshold: number) => void;
  onComplete: () => void;
  onGoToReport: () => void;
}

export function ExperimentSummaryPanel({
  experiment,
  talcThreshold,
  onApplyThresholdToDeposit,
  onComplete,
  onGoToReport,
}: ExperimentSummaryPanelProps) {
  const [whatIfOpen, setWhatIfOpen] = useState(false);
  const [whatIfThreshold, setWhatIfThreshold] = useState(talcThreshold);

  const framesWithMetrics = experiment.frames.filter((f) => f.metrics);
  const avgMetrics = useMemo(() => {
    if (framesWithMetrics.length === 0) return null;
    const sum = framesWithMetrics.reduce(
      (acc, f) => ({
        talcFraction: acc.talcFraction + f.metrics!.talcFraction,
        coarseFraction: acc.coarseFraction + f.metrics!.coarseFraction,
        fineFraction: acc.fineFraction + f.metrics!.fineFraction,
      }),
      { talcFraction: 0, coarseFraction: 0, fineFraction: 0 },
    );
    const n = framesWithMetrics.length;
    return { talcFraction: sum.talcFraction / n, coarseFraction: sum.coarseFraction / n, fineFraction: sum.fineFraction / n };
  }, [framesWithMetrics]);

  const whatIfResult = useMemo(() => {
    if (!whatIfOpen) return null;
    const recomputed = experiment.frames.map((f) =>
      f.metrics ? { ...f, frameClass: classifyFrame(f.metrics, whatIfThreshold).oreClass, manualClassOverride: undefined } : f,
    );
    return aggregateExperiment(recomputed, experiment.referenceFrameId);
  }, [whatIfOpen, whatIfThreshold, experiment.frames, experiment.referenceFrameId]);

  const canComplete = experiment.frames.length > 0 && experiment.status !== 'completed' && experiment.status !== 'reported';

  return (
    <Paper variant="outlined" sx={{ p: 2.5 }}>
      <Stack
        direction="row"
        sx={{
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: 2
        }}>
        <Box>
          <Typography variant="overline" sx={{
            color: "text.secondary"
          }}>
            Итоговый класс эксперимента
          </Typography>
          <Stack
            direction="row"
            spacing={1.5}
            sx={{
              alignItems: "center",
              mt: 0.5
            }}>
            <OreClassBadge oreClass={experiment.experimentClass} size="medium" />
            {experiment.classDerivation && (
              <Typography variant="body2" sx={{
                color: "text.secondary"
              }}>
                {DERIVATION_LABEL[experiment.classDerivation]}
              </Typography>
            )}
          </Stack>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" startIcon={<TuneIcon />} onClick={() => setWhatIfOpen((v) => !v)}>
            Что если?
          </Button>
          <Button variant="outlined" startIcon={<DescriptionIcon />} onClick={onGoToReport}>
            Сформировать отчёт
          </Button>
          <Button variant="contained" startIcon={<CheckCircleOutlineIcon />} disabled={!canComplete} onClick={onComplete}>
            Завершить эксперимент
          </Button>
        </Stack>
      </Stack>
      <Collapse in={whatIfOpen}>
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle2" gutterBottom>
          What-if: порог доли талька
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: "text.secondary",
            display: "block",
            mb: 1
          }}>
          Слайдер локально пересчитывает классификацию по сохранённым метрикам — данные месторождения не меняются, пока вы не примените порог.
        </Typography>
        <Box
          sx={{
            maxWidth: 420,
            px: 1
          }}>
          <Slider
            value={whatIfThreshold}
            min={0}
            max={0.3}
            step={0.005}
            valueLabelDisplay="on"
            valueLabelFormat={(v) => formatPercent(v)}
            onChange={(_, v) => setWhatIfThreshold(v as number)}
          />
        </Box>
        <Stack
          direction="row"
          spacing={2}
          sx={{
            alignItems: "center",
            mt: 1
          }}>
          <Typography variant="body2">Новый итоговый класс:</Typography>
          <OreClassBadge oreClass={whatIfResult?.experimentClass} />
          {whatIfResult?.hasDiscrepancies && <Chip label="Останутся расхождения" size="small" color="warning" />}
          <Button size="small" onClick={() => onApplyThresholdToDeposit(whatIfThreshold)}>
            Применить порог к месторождению
          </Button>
        </Stack>
      </Collapse>
      {avgMetrics && (
        <>
          <Divider sx={{ my: 2 }} />
          <Stack direction="row" spacing={4} sx={{
            flexWrap: "wrap"
          }}>
            <Box>
              <Typography variant="caption" sx={{
                color: "text.secondary"
              }}>
                Средняя доля талька
              </Typography>
              <Typography variant="h6">{formatPercent(avgMetrics.talcFraction)}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" sx={{
                color: "text.secondary"
              }}>
                Крупные срастания
              </Typography>
              <Typography variant="h6">{formatPercent(avgMetrics.coarseFraction)}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" sx={{
                color: "text.secondary"
              }}>
                Тонкие срастания
              </Typography>
              <Typography variant="h6">{formatPercent(avgMetrics.fineFraction)}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" sx={{
                color: "text.secondary"
              }}>
                Режим анализа
              </Typography>
              <Typography variant="h6">
                {{ ml: 'ML', manual: 'Ручной', mixed: 'Смешанный' }[experiment.analysisMode]}
              </Typography>
            </Box>
          </Stack>
        </>
      )}
    </Paper>
  );
}
