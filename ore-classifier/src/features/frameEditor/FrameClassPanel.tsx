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
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import AddIcon from '@mui/icons-material/Add';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import EditIcon from '@mui/icons-material/Edit';
import { OreClassBadge } from '../../components/OreClassBadge';
import { ORE_CLASS_META } from '../../theme/palette';
import type { Frame, OreClass, MineralRole } from '../../types/models';
import { formatPercent } from '../../utils/format';
import { useDepositsStore } from '../../stores/depositsStore';

interface FrameClassPanelProps {
  frame: Frame;
  depositId: string;
  onConfirm: () => void;
  onManualOverride: (oreClass: OreClass | undefined) => void;
}

export function FrameClassPanel({ frame, depositId, onConfirm, onManualOverride }: FrameClassPanelProps) {
  const [editing, setEditing] = useState(false);
  const [addMineralOpen, setAddMineralOpen] = useState(false);
  const addMineral = useDepositsStore((s) => s.addMineral);

  const effectiveClass = frame.manualClassOverride ?? frame.frameClass;

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Классификация кадра
      </Typography>
      {frame.metrics ? (
        <Stack spacing={1} sx={{
          mb: 2
        }}>
          <MetricRow label="Доля талька" value={frame.metrics.talcFraction} />
          <MetricRow label="Доля сульфидов" value={frame.metrics.sulfideFraction} />
          <MetricRow label="Крупные срастания" value={frame.metrics.coarseFraction} />
          <MetricRow label="Тонкие срастания" value={frame.metrics.fineFraction} />
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
          {frame.classReason && (
            <Typography variant="body2" sx={{
              color: "text.secondary"
            }}>
              {frame.classReason}
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
      <Stack
        direction="row"
        spacing={1}
        sx={{
          mt: 2,
          flexWrap: "wrap"
        }}>
        <Button size="small" variant="contained" startIcon={<CheckCircleIcon />} onClick={onConfirm} disabled={!frame.frameClass && !frame.manualClassOverride}>
          Подтвердить
        </Button>
        <Button size="small" variant="outlined" startIcon={<EditIcon />} onClick={() => setEditing((v) => !v)}>
          Исправить класс
        </Button>
        {frame.manualClassOverride && (
          <Button size="small" color="inherit" onClick={() => onManualOverride(undefined)}>
            Сбросить правку
          </Button>
        )}
      </Stack>
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
