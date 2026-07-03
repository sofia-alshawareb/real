import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import Chip from '@mui/material/Chip';
import Box from '@mui/material/Box';
import { OreClassBadge } from '../../components/OreClassBadge';
import { FrameThumbnail } from '../../components/FrameThumbnail';
import type { Deposit, Experiment, ReportDraft } from '../../types/models';

const DERIVATION_LABEL: Record<string, string> = {
  majority: 'по большинству кадров',
  reference: 'по опорному кадру',
  manual: 'вручную',
};

interface ReportPreviewProps {
  experiment: Experiment;
  deposit?: Deposit;
  draft: ReportDraft;
}

export function ReportPreview({ experiment, deposit, draft }: ReportPreviewProps) {
  const includedFrames = experiment.frames.filter((f) => draft.includedFrameIds.includes(f.id));

  return (
    <Paper variant="outlined" sx={{ p: 3 }}>
      <Chip label="Внутренняя справка · не для передачи третьим лицам" size="small" sx={{ mb: 2 }} />
      <Typography variant="h6">{experiment.title}</Typography>
      <Typography variant="body2" gutterBottom sx={{
        color: "text.secondary"
      }}>
        {deposit?.name ?? '—'} · Автор: {experiment.author}
      </Typography>
      <Stack
        direction="row"
        spacing={2}
        sx={{
          alignItems: "center",
          my: 2
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
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle2">Введение</Typography>
      <Typography
        variant="body2"
        sx={{
          whiteSpace: "pre-wrap",
          mb: 2
        }}>
        {draft.intro || '—'}
      </Typography>
      <Typography variant="subtitle2">Иллюстрации ({includedFrames.length})</Typography>
      <Stack
        direction="row"
        sx={{
          flexWrap: "wrap",
          gap: 1,
          my: 1
        }}>
        {includedFrames.map((f) => (
          <Box key={f.id} sx={{
            width: 110
          }}>
            <FrameThumbnail frame={f} width={110} height={70} />
            <Typography variant="caption" noWrap sx={{
              display: "block"
            }}>
              {f.name}
            </Typography>
          </Box>
        ))}
      </Stack>
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle2">Выводы</Typography>
      <Typography
        variant="body2"
        sx={{
          whiteSpace: "pre-wrap",
          mb: 2
        }}>
        {draft.conclusion || '—'}
      </Typography>
      <Typography variant="subtitle2">Рекомендации</Typography>
      <Typography variant="body2" sx={{
        whiteSpace: "pre-wrap"
      }}>
        {draft.recommendations || '—'}
      </Typography>
    </Paper>
  );
}
