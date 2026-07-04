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
import Divider from '@mui/material/Divider';
import { MineralProfileEditor } from './MineralProfileEditor';
import type { Deposit, DepositReserves, Mineral, MetalGrades } from '../../types/models';
import { formatPercent } from '../../utils/format';

interface DepositDialogProps {
  open: boolean;
  deposit: Deposit | null;
  onClose: () => void;
  onSave: (data: {
    name: string;
    code: string;
    talcThreshold: number;
    notes?: string;
    minerals: Mineral[];
    oreCluster?: string;
    region?: string;
    oreTypes?: string[];
    reserves?: DepositReserves;
    metalGrades?: MetalGrades;
  }) => void;
}

function parseOreTypes(text: string): string[] | undefined {
  const list = text
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  return list.length ? list : undefined;
}

export function DepositDialog({ open, deposit, onClose, onSave }: DepositDialogProps) {
  const [name, setName] = useState('');
  const [code, setCode] = useState('');
  const [talcThreshold, setTalcThreshold] = useState(0.1);
  const [notes, setNotes] = useState('');
  const [minerals, setMinerals] = useState<Mineral[]>([]);
  const [oreCluster, setOreCluster] = useState('');
  const [region, setRegion] = useState('');
  const [oreTypesText, setOreTypesText] = useState('');
  const [provenProbable, setProvenProbable] = useState('');
  const [measuredIndicated, setMeasuredIndicated] = useState('');
  const [balance, setBalance] = useState('');
  const [nickel, setNickel] = useState('');
  const [copper, setCopper] = useState('');
  const [mpg, setMpg] = useState('');
  const [gold, setGold] = useState('');
  const [silver, setSilver] = useState('');
  const [iron, setIron] = useState('');
  const [cobalt, setCobalt] = useState('');

  useEffect(() => {
    if (open) {
      setName(deposit?.name ?? '');
      setCode(deposit?.code ?? '');
      setTalcThreshold(deposit?.talcThreshold ?? 0.1);
      setNotes(deposit?.notes ?? '');
      setMinerals(deposit?.minerals ?? []);
      setOreCluster(deposit?.oreCluster ?? '');
      setRegion(deposit?.region ?? '');
      setOreTypesText(deposit?.oreTypes?.join(', ') ?? '');
      setProvenProbable(deposit?.reserves?.provenProbable ?? '');
      setMeasuredIndicated(deposit?.reserves?.measuredIndicated ?? '');
      setBalance(deposit?.reserves?.balance ?? '');
      setNickel(deposit?.metalGrades?.nickel ?? '');
      setCopper(deposit?.metalGrades?.copper ?? '');
      setMpg(deposit?.metalGrades?.mpg ?? '');
      setGold(deposit?.metalGrades?.gold ?? '');
      setSilver(deposit?.metalGrades?.silver ?? '');
      setIron(deposit?.metalGrades?.iron ?? '');
      setCobalt(deposit?.metalGrades?.cobalt ?? '');
    }
  }, [open, deposit]);

  const handleSave = () => {
    const reserves: DepositReserves | undefined =
      provenProbable.trim() || measuredIndicated.trim() || balance.trim()
        ? {
            provenProbable: provenProbable.trim() || undefined,
            measuredIndicated: measuredIndicated.trim() || undefined,
            balance: balance.trim() || undefined,
          }
        : undefined;
    const metalGrades: MetalGrades | undefined =
      nickel.trim() || copper.trim() || mpg.trim() || gold.trim() || silver.trim() || iron.trim() || cobalt.trim()
        ? {
            nickel: nickel.trim() || undefined,
            copper: copper.trim() || undefined,
            mpg: mpg.trim() || undefined,
            gold: gold.trim() || undefined,
            silver: silver.trim() || undefined,
            iron: iron.trim() || undefined,
            cobalt: cobalt.trim() || undefined,
          }
        : undefined;
    onSave({
      name: name.trim(),
      code: code.trim(),
      talcThreshold,
      notes: notes.trim() || undefined,
      minerals,
      oreCluster: oreCluster.trim() || undefined,
      region: region.trim() || undefined,
      oreTypes: parseOreTypes(oreTypesText),
      reserves,
      metalGrades,
    });
  };

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
          <Stack direction="row" spacing={2}>
            <TextField label="Рудный узел" value={oreCluster} onChange={(e) => setOreCluster(e.target.value)} fullWidth />
            <TextField label="Регион" value={region} onChange={(e) => setRegion(e.target.value)} fullWidth />
          </Stack>
          <TextField
            label="Типы руд (через запятую)"
            placeholder="богатые, медистые, вкрапленные"
            value={oreTypesText}
            onChange={(e) => setOreTypesText(e.target.value)}
            fullWidth
          />
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

          <Divider />
          <Typography variant="subtitle2">Запасы и ресурсы</Typography>
          <Stack direction="row" spacing={2}>
            <TextField
              label="Доказ. и вероятные (JORC)"
              placeholder="622,8 млн т"
              value={provenProbable}
              onChange={(e) => setProvenProbable(e.target.value)}
              fullWidth
            />
            <TextField
              label="Оценённые и выявленные (JORC)"
              placeholder="1 546,3 млн т"
              value={measuredIndicated}
              onChange={(e) => setMeasuredIndicated(e.target.value)}
              fullWidth
            />
            <TextField
              label="Балансовые запасы"
              placeholder="1 979,6 млн т"
              value={balance}
              onChange={(e) => setBalance(e.target.value)}
              fullWidth
            />
          </Stack>

          <Typography variant="subtitle2">Среднее содержание металлов</Typography>
          <Stack direction="row" spacing={2} useFlexGap sx={{ flexWrap: 'wrap' }}>
            <TextField label="Никель" placeholder="2,22%" value={nickel} onChange={(e) => setNickel(e.target.value)} sx={{ width: 130 }} />
            <TextField label="Медь" placeholder="3,54%" value={copper} onChange={(e) => setCopper(e.target.value)} sx={{ width: 130 }} />
            <TextField label="МПГ" placeholder="10,27 г/т" value={mpg} onChange={(e) => setMpg(e.target.value)} sx={{ width: 130 }} />
            <TextField label="Золото" placeholder="8,1 млн тр. ун." value={gold} onChange={(e) => setGold(e.target.value)} sx={{ width: 150 }} />
            <TextField label="Серебро" value={silver} onChange={(e) => setSilver(e.target.value)} sx={{ width: 130 }} />
            <TextField label="Железо" value={iron} onChange={(e) => setIron(e.target.value)} sx={{ width: 130 }} />
            <TextField label="Кобальт" value={cobalt} onChange={(e) => setCobalt(e.target.value)} sx={{ width: 130 }} />
          </Stack>

          <Divider />
          <Typography variant="subtitle2">Профиль минералов</Typography>
          <MineralProfileEditor minerals={minerals} onChange={setMinerals} />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Отмена</Button>
        <Button variant="contained" disabled={!name.trim() || !code.trim()} onClick={handleSave}>
          Сохранить
        </Button>
      </DialogActions>
    </Dialog>
  );
}
