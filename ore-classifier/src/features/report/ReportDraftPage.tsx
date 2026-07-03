import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';
import IconButton from '@mui/material/IconButton';
import CircularProgress from '@mui/material/CircularProgress';
import Grid from '@mui/material/Grid';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import TableChartIcon from '@mui/icons-material/TableChart';
import { pdf } from '@react-pdf/renderer';
import { db } from '../../db/db';
import { useExperimentsStore } from '../../stores/experimentsStore';
import { useDepositsStore } from '../../stores/depositsStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { FramePicker } from './FramePicker';
import { ReportPreview } from './ReportPreview';
import { downloadCsv } from '../../services/export/csvExport';
import { ReportPdf } from '../../services/export/ReportPdf';
import { getFrameThumbnailDataUrl } from '../../components/FrameThumbnail';
import type { ReportDraft } from '../../types/models';
import { notify } from '../../utils/toast';

function buildDefaultDraft(experimentId: string, title: string, frameIds: string[]): ReportDraft {
  return {
    experimentId,
    intro: `Проведён анализ панорамных OM-изображений полированных шлифов в рамках эксперимента «${title}». Ниже приведены результаты автоматической и ручной классификации по кадрам.`,
    conclusion: '',
    recommendations: '',
    includedFrameIds: frameIds,
    snapshotAt: Date.now(),
  };
}

export function ReportDraftPage() {
  const { experimentId } = useParams<{ experimentId: string }>();
  const navigate = useNavigate();
  const experiment = useExperimentsStore((s) => s.experiments.find((e) => e.id === experimentId));
  const deposit = useDepositsStore((s) => s.deposits.find((d) => d.id === experiment?.depositId));
  const markReported = useExperimentsStore((s) => s.markReported);
  const author = useSettingsStore((s) => s.authorName) || 'Без имени';

  const [draft, setDraft] = useState<ReportDraft | null>(null);
  const [exportingPdf, setExportingPdf] = useState(false);
  const saveTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!experiment) return;
    void db.reportDrafts.get(experiment.id).then((existing) => {
      setDraft(existing ?? buildDefaultDraft(experiment.id, experiment.title, experiment.frames.map((f) => f.id)));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experiment?.id]);

  const persistDraft = useCallback((next: ReportDraft) => {
    if (saveTimeout.current) clearTimeout(saveTimeout.current);
    saveTimeout.current = setTimeout(() => void db.reportDrafts.put(next), 600);
  }, []);

  const updateDraft = (patch: Partial<ReportDraft>) => {
    setDraft((prev) => {
      if (!prev) return prev;
      const next = { ...prev, ...patch };
      persistDraft(next);
      return next;
    });
  };

  const handleToggleFrame = (frameId: string) => {
    if (!draft) return;
    const included = draft.includedFrameIds.includes(frameId)
      ? draft.includedFrameIds.filter((id) => id !== frameId)
      : [...draft.includedFrameIds, frameId];
    updateDraft({ includedFrameIds: included });
  };

  const handleRefreshSnapshot = () => updateDraft({ snapshotAt: Date.now() });

  const handleExportCsv = () => {
    if (!experiment) return;
    downloadCsv(experiment, deposit);
    markReported(experiment.id, author);
    notify('CSV выгружен', 'success');
  };

  const handleExportPdf = async () => {
    if (!experiment || !draft) return;
    setExportingPdf(true);
    try {
      const includedFrames = experiment.frames.filter((f) => draft.includedFrameIds.includes(f.id));
      const thumbnails: Record<string, string> = {};
      for (const f of includedFrames) {
        thumbnails[f.id] = await getFrameThumbnailDataUrl(f, 320, 200, true);
      }
      const blob = await pdf(<ReportPdf experiment={experiment} deposit={deposit} draft={draft} frameThumbnails={thumbnails} />).toBlob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${experiment.title.replace(/[^\wа-яА-Я0-9-]+/gi, '_')}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
      markReported(experiment.id, author);
      notify('PDF выгружен', 'success');
    } catch (err) {
      notify('Не удалось сформировать PDF', 'error');
      console.error(err);
    } finally {
      setExportingPdf(false);
    }
  };

  if (!experiment || !draft) {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          py: 6
        }}>
        <CircularProgress />
      </Box>
    );
  }

  const isStale = experiment.updatedAt > draft.snapshotAt;

  return (
    <Box>
      <Stack
        direction="row"
        spacing={1}
        sx={{
          alignItems: "center",
          mb: 2
        }}>
        <IconButton onClick={() => navigate(`/experiments/${experiment.id}`)}>
          <ArrowBackIcon />
        </IconButton>
        <Typography variant="h5" sx={{
          flexGrow: 1
        }}>
          Отчёт: {experiment.title}
        </Typography>
        <Button variant="outlined" startIcon={<TableChartIcon />} onClick={handleExportCsv}>
          Экспорт CSV
        </Button>
        <Button variant="contained" startIcon={<PictureAsPdfIcon />} onClick={handleExportPdf} disabled={exportingPdf}>
          {exportingPdf ? 'Формирование...' : 'Экспорт PDF'}
        </Button>
      </Stack>
      {isStale && (
        <Alert severity="warning" sx={{ mb: 2 }} action={<Button onClick={handleRefreshSnapshot}>Обновить черновик</Button>}>
          Данные эксперимента изменились после последнего сохранения черновика.
        </Alert>
      )}
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper variant="outlined" sx={{ p: 3, mb: 3 }}>
            <Stack spacing={2}>
              <TextField
                label="Введение"
                multiline
                minRows={3}
                value={draft.intro}
                onChange={(e) => updateDraft({ intro: e.target.value })}
              />
              <TextField
                label="Выводы"
                multiline
                minRows={3}
                value={draft.conclusion}
                onChange={(e) => updateDraft({ conclusion: e.target.value })}
              />
              <TextField
                label="Рекомендации"
                multiline
                minRows={3}
                value={draft.recommendations}
                onChange={(e) => updateDraft({ recommendations: e.target.value })}
              />
            </Stack>
          </Paper>

          <Typography variant="subtitle1" sx={{
            mb: 1
          }}>
            Кадры в отчёте
          </Typography>
          <FramePicker frames={experiment.frames} includedIds={draft.includedFrameIds} onToggle={handleToggleFrame} />
        </Grid>

        <Grid size={{ xs: 12, md: 6 }}>
          <ReportPreview experiment={experiment} deposit={deposit} draft={draft} />
        </Grid>
      </Grid>
    </Box>
  );
}
