import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import IconButton from '@mui/material/IconButton';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import HistoryIcon from '@mui/icons-material/History';
import { useExperimentsStore } from '../../stores/experimentsStore';
import { useDepositsStore } from '../../stores/depositsStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { FrameFilmstrip } from './FrameFilmstrip';
import { ExperimentSummaryPanel } from './ExperimentSummaryPanel';
import { DiscrepancyBanner } from './DiscrepancyBanner';
import { ExperimentStatusChip } from '../../components/StatusChip';
import { formatDateTime } from '../../utils/format';
import { notify } from '../../utils/toast';

export function ExperimentCardPage() {
  const { experimentId } = useParams<{ experimentId: string }>();
  const navigate = useNavigate();
  const author = useSettingsStore((s) => s.authorName) || 'Без имени';
  const experiment = useExperimentsStore((s) => s.experiments.find((e) => e.id === experimentId));
  const setReferenceFrame = useExperimentsStore((s) => s.setReferenceFrame);
  const completeExperiment = useExperimentsStore((s) => s.completeExperiment);
  const openExperiment = useExperimentsStore((s) => s.openExperiment);
  const releaseExperiment = useExperimentsStore((s) => s.releaseExperiment);
  const forceEditExperiment = useExperimentsStore((s) => s.forceEditExperiment);
  const deposit = useDepositsStore((s) => s.deposits.find((d) => d.id === experiment?.depositId));
  const updateDeposit = useDepositsStore((s) => s.updateDeposit);

  const [locked, setLocked] = useState(false);

  useEffect(() => {
    if (!experimentId) return;
    const result = openExperiment(experimentId, author);
    setLocked(result === 'locked');
    return () => releaseExperiment(experimentId, author);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experimentId, author]);

  if (!experiment) {
    return (
      <Box>
        <Typography>Эксперимент не найден.</Typography>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/')}>
          К списку экспериментов
        </Button>
      </Box>
    );
  }

  const lockOwner = experiment.openedBy;

  return (
    <Box>
      <Stack
        direction="row"
        spacing={1}
        sx={{
          alignItems: "center",
          mb: 2
        }}>
        <IconButton onClick={() => navigate('/')}>
          <ArrowBackIcon />
        </IconButton>
        <Box sx={{
          flexGrow: 1
        }}>
          <Typography variant="h5">{experiment.title}</Typography>
          <Stack
            direction="row"
            spacing={1}
            sx={{
              alignItems: "center",
              mt: 0.5
            }}>
            <Typography variant="body2" sx={{
              color: "text.secondary"
            }}>
              {deposit?.name ?? '—'} · автор: {experiment.author}
            </Typography>
            <ExperimentStatusChip status={experiment.status} />
          </Stack>
        </Box>
      </Stack>
      {locked && lockOwner && lockOwner !== author && (
        <Alert
          severity="warning"
          sx={{ mb: 2 }}
          action={
            <Button
              color="inherit"
              size="small"
              onClick={() => {
                forceEditExperiment(experimentId!, author);
                setLocked(false);
              }}
            >
              Всё равно редактировать
            </Button>
          }
        >
          Эксперимент открыт пользователем «{lockOwner}» менее 5 минут назад. Изменения могут конфликтовать.
        </Alert>
      )}
      <DiscrepancyBanner experiment={experiment} />
      <Box sx={{
        mb: 3
      }}>
        <ExperimentSummaryPanel
          experiment={experiment}
          talcThreshold={deposit?.talcThreshold ?? 0.1}
          onApplyThresholdToDeposit={(newThreshold) => {
            if (deposit) {
              updateDeposit(deposit.id, { talcThreshold: newThreshold });
              notify(`Порог талька для «${deposit.name}» обновлён на ${(newThreshold * 100).toFixed(1)}%`, 'success');
            }
          }}
          onComplete={() => completeExperiment(experiment.id, author)}
          onGoToReport={() => navigate(`/experiments/${experiment.id}/report`)}
        />
      </Box>
      <Typography variant="subtitle1" sx={{
        mb: 1
      }}>
        Кадры ({experiment.frames.length})
      </Typography>
      {experiment.frames.length > 0 ? (
        <FrameFilmstrip experiment={experiment} onSetReference={(frameId) => setReferenceFrame(experiment.id, frameId)} />
      ) : (
        <Paper variant="outlined" sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
          В этом эксперименте пока нет кадров.
        </Paper>
      )}
      <Alert severity="info" sx={{ mt: 3 }} icon={<Chip label="Экспериментально" size="small" sx={{ mr: -0.5 }} />}>
        Перенос сегментации между кадрами (experimental propagation) — функция в разработке. Пока каждый кадр
        обрабатывается и редактируется независимо.
      </Alert>
      <Box sx={{
        mt: 4
      }}>
        <Stack
          direction="row"
          spacing={1}
          sx={{
            alignItems: "center",
            mb: 1
          }}>
          <HistoryIcon fontSize="small" color="action" />
          <Typography variant="subtitle1">История эксперимента</Typography>
        </Stack>
        <Paper variant="outlined">
          <List dense>
            {[...experiment.history]
              .sort((a, b) => b.at - a.at)
              .map((entry, idx) => (
                <ListItem key={idx} divider>
                  <ListItemText
                    primary={entry.action}
                    secondary={`${formatDateTime(entry.at)} · ${entry.author}`}
                  />
                </ListItem>
              ))}
          </List>
        </Paper>
      </Box>
    </Box>
  );
}
