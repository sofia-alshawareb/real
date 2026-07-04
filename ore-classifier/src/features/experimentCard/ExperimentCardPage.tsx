import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
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
import LinearProgress from '@mui/material/LinearProgress';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import HistoryIcon from '@mui/icons-material/History';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { useExperimentsStore, genId } from '../../stores/experimentsStore';
import { useDepositsStore } from '../../stores/depositsStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { FrameFilmstrip } from './FrameFilmstrip';
import { ExperimentSummaryPanel } from './ExperimentSummaryPanel';
import { DiscrepancyBanner } from './DiscrepancyBanner';
import { ExperimentStatusChip } from '../../components/StatusChip';
import { formatDateTime } from '../../utils/format';
import { notify } from '../../utils/toast';
import { importImageFile, UnsupportedImageError } from '../../services/tiling/tileImporter';
import { enqueueFrame } from '../../services/mockMl/queueRunner';
import type { Frame } from '../../types/models';

const DEFAULT_PIXEL_SIZE_UM = 0.5;

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
  const removeFrame = useExperimentsStore((s) => s.removeFrame);
  const addFrames = useExperimentsStore((s) => s.addFrames);
  const deposit = useDepositsStore((s) => s.deposits.find((d) => d.id === experiment?.depositId));
  const updateDeposit = useDepositsStore((s) => s.updateDeposit);

  const [locked, setLocked] = useState(false);
  const [uploading, setUploading] = useState(0);
  const [uploadTotal, setUploadTotal] = useState(0);

  const onDrop = (accepted: File[]) => {
    if (!experiment || accepted.length === 0) return;
    setUploadTotal((t) => t + accepted.length);
    accepted.forEach((file) => {
      importImageFile(file)
        .then((result) => {
          const newFrame: Frame = {
            id: genId('frame'),
            index: experiment.frames.length,
            name: file.name,
            source: { kind: 'dexie', imageId: result.imageId },
            width: result.width,
            height: result.height,
            pixelSizeUm: DEFAULT_PIXEL_SIZE_UM,
            status: 'queued',
            isReference: experiment.frames.length === 0,
            manuallyEditedMask: false,
            updatedAt: Date.now(),
          };
          addFrames(experiment.id, [newFrame]);
          enqueueFrame(experiment.id, newFrame.id);
          if (result.downscaled) {
            notify(`«${file.name}» уменьшено до допустимого разрешения браузера`, 'info');
          }
        })
        .catch((err) => {
          const message = err instanceof UnsupportedImageError ? err.message : `Не удалось обработать «${file.name}»`;
          notify(message, 'warning');
        })
        .finally(() => setUploading((n) => n + 1));
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'], 'image/webp': ['.webp'] },
    maxSize: 500 * 1024 * 1024,
  });

  const isUploadingFrames = uploadTotal > 0 && uploading < uploadTotal;

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
      <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="subtitle1">Кадры ({experiment.frames.length})</Typography>
      </Stack>
      {experiment.frames.length > 0 ? (
        <FrameFilmstrip
          experiment={experiment}
          onSetReference={(frameId) => setReferenceFrame(experiment.id, frameId)}
          onDeleteFrame={(frameId) => {
            removeFrame(experiment.id, frameId);
            notify('Кадр удалён', 'success');
          }}
        />
      ) : (
        <Paper variant="outlined" sx={{ p: 3, textAlign: 'center', color: 'text.secondary', mb: 2 }}>
          В этом эксперименте пока нет кадров.
        </Paper>
      )}
      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'divider',
          borderRadius: 2,
          p: 2,
          mt: 2,
          textAlign: 'center',
          cursor: 'pointer',
          bgcolor: isDragActive ? 'action.hover' : 'transparent',
        }}
      >
        <input {...getInputProps()} />
        <Stack direction="row" spacing={1} sx={{ alignItems: 'center', justifyContent: 'center' }}>
          <UploadFileIcon fontSize="small" sx={{ color: 'text.secondary' }} />
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Перетащите изображения сюда или нажмите, чтобы добавить кадры в эксперимент
          </Typography>
        </Stack>
        {isUploadingFrames && <LinearProgress sx={{ mt: 1 }} />}
      </Box>
      <Alert severity="info" sx={{ mt: 3 }} icon={<Chip label="Экспериментально" size="small" sx={{ mr: -0.5 }} />}>
        Перенос сегментации между кадрами (experimental propagation) — функция в разработке. Пока каждый кадр
        обрабатывается и редактируется независимо.
        <br />
        Опорный кадр задаётся звёздочкой на миниатюре кадра — наведите курсор на кадр и нажмите значок звезды
        в правом нижнем углу, чтобы сделать его опорным.
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
