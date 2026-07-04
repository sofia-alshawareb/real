import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import Box from '@mui/material/Box';
import Table from '@mui/material/Table';
import TableHead from '@mui/material/TableHead';
import TableBody from '@mui/material/TableBody';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';
import TextField from '@mui/material/TextField';
import { OreClassBadge } from '../../components/OreClassBadge';
import { FrameThumbnail } from '../../components/FrameThumbnail';
import { ROLE_LABELS } from '../deposits/MineralProfileEditor';
import { formatPercent } from '../../utils/format';
import { ORE_CLASS_META } from '../../theme/palette';
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
  onMineralNoteChange?: (mineralId: string, text: string) => void;
}

export function ReportPreview({ experiment, deposit, draft, onMineralNoteChange }: ReportPreviewProps) {
  const includedFrames = experiment.frames.filter((f) => draft.includedFrameIds.includes(f.id));
  const depositMetaLine = deposit
    ? [deposit.oreCluster, deposit.region, deposit.oreTypes?.length ? deposit.oreTypes.join(', ') : undefined]
        .filter(Boolean)
        .join(' · ')
    : '';

  return (
    <Paper variant="outlined" sx={{ p: 3 }}>
      <Typography variant="h6">{experiment.title}</Typography>
      <Typography variant="body2" sx={{
        color: "text.secondary"
      }}>
        {deposit?.name ?? '—'} · Автор: {experiment.author}
      </Typography>
      {depositMetaLine && (
        <Typography variant="caption" gutterBottom sx={{ color: 'text.secondary', display: 'block' }}>
          {depositMetaLine}
        </Typography>
      )}
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

      <Typography variant="subtitle2" gutterBottom>
        Количественные метрики по кадрам
      </Typography>
      <Table size="small" sx={{ mb: 2 }}>
        <TableHead>
          <TableRow>
            <TableCell>Кадр</TableCell>
            <TableCell align="right">Сульфиды всего</TableCell>
            <TableCell align="right">Обычные</TableCell>
            <TableCell align="right">Тонкие</TableCell>
            <TableCell align="right">Тальк</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {includedFrames.map((f) => (
            <TableRow key={f.id}>
              <TableCell>{f.name}</TableCell>
              <TableCell align="right">{f.metrics ? formatPercent(f.metrics.sulfideFraction) : '—'}</TableCell>
              <TableCell align="right">{f.metrics ? formatPercent(f.metrics.coarseFraction) : '—'}</TableCell>
              <TableCell align="right">{f.metrics ? formatPercent(f.metrics.fineFraction) : '—'}</TableCell>
              <TableCell align="right">{f.metrics ? formatPercent(f.metrics.talcFraction) : '—'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {deposit && deposit.minerals.length > 0 && (
        <>
          <Typography variant="subtitle2" gutterBottom>
            Профиль минералов месторождения
          </Typography>
          <Table size="small" sx={{ mb: 2 }}>
            <TableHead>
              <TableRow>
                <TableCell>Минерал</TableCell>
                <TableCell>Роль</TableCell>
                <TableCell>Заметка</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {deposit.minerals.map((m) => (
                <TableRow key={m.id}>
                  <TableCell>
                    <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
                      <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: m.colorHex, flexShrink: 0 }} />
                      <span>{m.name}</span>
                    </Stack>
                  </TableCell>
                  <TableCell>{ROLE_LABELS[m.role]}</TableCell>
                  <TableCell sx={{ minWidth: 180 }}>
                    <TextField
                      size="small"
                      variant="standard"
                      fullWidth
                      placeholder="Заметка"
                      value={draft.mineralNotes?.[m.id] ?? m.note ?? ''}
                      onChange={(e) => onMineralNoteChange?.(m.id, e.target.value)}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </>
      )}

      <Typography variant="subtitle2">Иллюстрации ({includedFrames.length})</Typography>
      <Stack spacing={2} sx={{ my: 1 }}>
        {includedFrames.map((f) => (
          <Box key={f.id}>
            <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
              {f.name}
              {f.frameClass && (
                <> · {ORE_CLASS_META[f.manualClassOverride ?? f.frameClass].label}</>
              )}
            </Typography>
            <Stack direction="row" spacing={1}>
              <Box sx={{ width: 150 }}>
                <FrameThumbnail frame={f} width={150} height={95} showMask={false} />
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', textAlign: 'center' }}>
                  Исходное изображение
                </Typography>
              </Box>
              <Box sx={{ width: 150 }}>
                <FrameThumbnail frame={f} width={150} height={95} showMask />
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', textAlign: 'center' }}>
                  С маской сегментации
                </Typography>
              </Box>
            </Stack>
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
