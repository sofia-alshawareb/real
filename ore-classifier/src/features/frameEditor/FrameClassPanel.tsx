import { useState } from 'react';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Divider from '@mui/material/Divider';
import LinearProgress from '@mui/material/LinearProgress';
import Box from '@mui/material/Box';
import Alert from '@mui/material/Alert';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import AddIcon from '@mui/icons-material/Add';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import VerifiedIcon from '@mui/icons-material/Verified';
import EditIcon from '@mui/icons-material/Edit';
import { OreClassBadge } from '../../components/OreClassBadge';
import { ORE_CLASS_META } from '../../theme/palette';
import { classifyFrame } from '../../services/rulesEngine';
import type { Frame, FrameMetrics, OreClass, MineralRole } from '../../types/models';
import { formatPercent } from '../../utils/format';
import { useDepositsStore } from '../../stores/depositsStore';

interface FrameClassPanelProps {
  frame: Frame;
  depositId: string;
  talcThreshold: number;
  liveMetrics?: FrameMetrics;
  onConfirm: () => void;
  onManualOverride: (oreClass: OreClass | undefined) => void;
}

export function FrameClassPanel({ frame, depositId, talcThreshold, liveMetrics, onConfirm, onManualOverride }: FrameClassPanelProps) {
  const [editing, setEditing] = useState(false);
  const [addMineralOpen, setAddMineralOpen] = useState(false);
  const addMineral = useDepositsStore((s) => s.addMineral);

  const isReviewed = frame.status === 'reviewed';
  const displayMetrics = liveMetrics ?? frame.metrics;

  const liveClassification = displayMetrics ? classifyFrame(displayMetrics, talcThreshold) : undefined;
  const expectedClass = liveClassification?.oreClass;
  const conflictsWithMetrics = Boolean(
    frame.manualClassOverride && expectedClass && frame.manualClassOverride !== expectedClass,
  );
  // Без ручной правки — показываем класс, пересчитанный по текущей (в т.ч. несохранённой) разметке.
  const effectiveClass = frame.manualClassOverride ?? (liveMetrics ? expectedClass : frame.frameClass);
  const displayReason = frame.manualClassOverride ? frame.classReason : (liveClassification?.reason ?? frame.classReason);

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Классификация кадра
      </Typography>
      {displayMetrics ? (
        <Stack spacing={1} sx={{
          mb: 2
        }}>
          {liveMetrics && (
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Предварительно, по текущей (несохранённой) разметке
            </Typography>
          )}
          <MetricRow label="Доля талька" value={displayMetrics.talcFraction} />
          <MetricRow label="Доля сульфидов" value={displayMetrics.sulfideFraction} />
          <MetricRow label="Обычные срастания" value={displayMetrics.coarseFraction} />
          <MetricRow label="Тонкие срастания" value={displayMetrics.fineFraction} />
        </Stack>
      ) : (
        <Typography
          variant="body2"
          sx={{
            color: "text.secondary",
            mb: 2
          }}>
          Метрики появятся после обработки или разметки кадра.
        </Typography>
      )}
      <Divider sx={{ mb: 2 }} />
      {!editing ? (
        <Stack spacing={1}>
          <OreClassBadge oreClass={effectiveClass} size="medium" />
          {frame.manualClassOverride && (
            <Typography
              variant="caption"
              sx={{
                color: "secondary.main",
                display: "flex",
                alignItems: "center",
                gap: 0.5
              }}>
              <EditIcon fontSize="inherit" /> Класс скорректирован вручную
            </Typography>
          )}
          {displayReason && (
            <Typography variant="body2" sx={{
              color: "text.secondary"
            }}>
              {displayReason}
            </Typography>
          )}
          {frame.confidence !== undefined && (
            <Typography variant="caption" sx={{
              color: "text.secondary"
            }}>
              Уверенность модели: {(frame.confidence * 100).toFixed(0)}%
            </Typography>
          )}
        </Stack>
      ) : (
        <Select
          size="small"
          fullWidth
          value={effectiveClass ?? ''}
          onChange={(e) => {
            onManualOverride(e.target.value as OreClass);
            setEditing(false);
          }}
        >
          {(Object.keys(ORE_CLASS_META) as OreClass[]).map((c) => (
            <MenuItem key={c} value={c}>
              {ORE_CLASS_META[c].label}
            </MenuItem>
          ))}
        </Select>
      )}
      {conflictsWithMetrics && expectedClass && (
        <Alert severity="warning" sx={{ mt: 1.5 }}>
          Выбранный класс «{ORE_CLASS_META[frame.manualClassOverride!].label}» не соответствует расчётным метрикам
          (ожидается «{ORE_CLASS_META[expectedClass].label}»). Правка всё равно сохранена — проверьте обоснование.
        </Alert>
      )}
      <Stack
        direction="row"
        spacing={1}
        sx={{
          mt: 2
        }}>
        <Button
          fullWidth
          variant={isReviewed ? 'outlined' : 'contained'}
          color={isReviewed ? 'success' : 'primary'}
          startIcon={isReviewed ? <VerifiedIcon /> : <CheckCircleIcon />}
          onClick={onConfirm}
          disabled={isReviewed || !effectiveClass}
        >
          {isReviewed ? 'Класс утверждён' : 'Утвердить класс (проверено)'}
        </Button>
        <Button fullWidth variant="outlined" startIcon={<EditIcon />} onClick={() => setEditing((v) => !v)}>
          Исправить класс
        </Button>
      </Stack>
      {frame.manualClassOverride && (
        <Button size="small" color="inherit" sx={{ mt: 1 }} onClick={() => onManualOverride(undefined)}>
          Сбросить правку
        </Button>
      )}
      <Divider sx={{ my: 2 }} />
      <Button size="small" startIcon={<AddIcon />} onClick={() => setAddMineralOpen(true)}>
        Минерал в профиль месторождения
      </Button>
      <AddMineralDialog
        open={addMineralOpen}
        onClose={() => setAddMineralOpen(false)}
        onSave={(name, role, colorHex) => {
          addMineral(depositId, { name, role, colorHex });
          setAddMineralOpen(false);
        }}
      />
    </Paper>
  );
}

function MetricRow({ label, value }: { label: string; value: number }) {
  return (
    <Box>
      <Stack direction="row" sx={{
        justifyContent: "space-between"
      }}>
        <Typography variant="body2" sx={{
          color: "text.secondary"
        }}>
          {label}
        </Typography>
        <Typography variant="body2">{formatPercent(value)}</Typography>
      </Stack>
      <LinearProgress variant="determinate" value={Math.min(100, value * 100)} sx={{ height: 4, borderRadius: 2 }} />
    </Box>
  );
}

function AddMineralDialog({
  open,
  onClose,
  onSave,
}: {
  open: boolean;
  onClose: () => void;
  onSave: (name: string, role: MineralRole, colorHex: string) => void;
}) {
  const [name, setName] = useState('');
  const [role, setRole] = useState<MineralRole>('sulfide');
  const [color, setColor] = useState('#D9A441');

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>Добавить минерал в справочник</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{
          mt: 1
        }}>
          <TextField label="Название минерала" value={name} onChange={(e) => setName(e.target.value)} autoFocus />
          <Select value={role} onChange={(e) => setRole(e.target.value as MineralRole)} size="small">
            <MenuItem value="sulfide">Сульфид</MenuItem>
            <MenuItem value="talc">Тальк</MenuItem>
            <MenuItem value="gangue">Нерудная матрица</MenuItem>
            <MenuItem value="other">Другое</MenuItem>
          </Select>
          <TextField
            label="Цвет"
            type="color"
            value={color}
            onChange={(e) => setColor(e.target.value)}
            sx={{ width: 100 }}
          />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Отмена</Button>
        <Button
          variant="contained"
          disabled={!name.trim()}
          onClick={() => onSave(name.trim(), role, color)}
        >
          Добавить
        </Button>
      </DialogActions>
    </Dialog>
  );
}
