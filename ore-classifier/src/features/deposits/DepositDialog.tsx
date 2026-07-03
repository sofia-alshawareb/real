import { useEffect, useState } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import { MineralProfileEditor } from './MineralProfileEditor';
import type { Deposit, Mineral } from '../../types/models';
import { formatPercent } from '../../utils/format';

interface DepositDialogProps {
  open: boolean;
  deposit: Deposit | null;
  onClose: () => void;
  onSave: (data: { name: string; code: string; talcThreshold: number; notes?: string; minerals: Mineral[] }) => void;
}

export function DepositDialog({ open, deposit, onClose, onSave }: DepositDialogProps) {
  const [name, setName] = useState('');
  const [code, setCode] = useState('');
  const [talcThreshold, setTalcThreshold] = useState(0.1);
  const [notes, setNotes] = useState('');
  const [minerals, setMinerals] = useState<Mineral[]>([]);

  useEffect(() => {
    if (open) {
      setName(deposit?.name ?? '');
      setCode(deposit?.code ?? '');
      setTalcThreshold(deposit?.talcThreshold ?? 0.1);
      setNotes(deposit?.notes ?? '');
      setMinerals(deposit?.minerals ?? []);
    }
  }, [open, deposit]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{deposit ? 'Редактировать месторождение' : 'Новое месторождение'}</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{
          mt: 1
        }}>
          <Stack direction="row" spacing={2}>
            <TextField label="Название" value={name} onChange={(e) => setName(e.target.value)} fullWidth autoFocus />
            <TextField label="Код" value={code} onChange={(e) => setCode(e.target.value)} sx={{ width: 140 }} />
          </Stack>
          <Box>
            <Typography variant="body2" gutterBottom sx={{
              color: "text.secondary"
            }}>
              Порог доли талька для оталькованной руды: {formatPercent(talcThreshold)}
            </Typography>
            <input
              type="range"
              min={0}
              max={0.3}
              step={0.005}
              value={talcThreshold}
              onChange={(e) => setTalcThreshold(Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </Box>
          <TextField label="Заметки" value={notes} onChange={(e) => setNotes(e.target.value)} multiline minRows={2} fullWidth />
          <Typography variant="subtitle2">Профиль минералов</Typography>
          <MineralProfileEditor minerals={minerals} onChange={setMinerals} />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Отмена</Button>
        <Button
          variant="contained"
          disabled={!name.trim() || !code.trim()}
          onClick={() => onSave({ name: name.trim(), code: code.trim(), talcThreshold, notes: notes.trim() || undefined, minerals })}
        >
          Сохранить
        </Button>
      </DialogActions>
    </Dialog>
  );
}
