import { useState } from 'react';
import Table from '@mui/material/Table';
import TableHead from '@mui/material/TableHead';
import TableBody from '@mui/material/TableBody';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';
import TextField from '@mui/material/TextField';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import Popover from '@mui/material/Popover';
import Box from '@mui/material/Box';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import { HexColorPicker } from 'react-colorful';
import type { Mineral, MineralRole } from '../../types/models';
import { genId } from '../../stores/experimentsStore';

export const ROLE_LABELS: Record<MineralRole, string> = {
  sulfide: 'Сульфид',
  talc: 'Тальк',
  gangue: 'Нерудная матрица',
  other: 'Другое',
};

interface MineralProfileEditorProps {
  minerals: Mineral[];
  onChange: (minerals: Mineral[]) => void;
}

export function MineralProfileEditor({ minerals, onChange }: MineralProfileEditorProps) {
  const [colorAnchor, setColorAnchor] = useState<{ el: HTMLElement; id: string } | null>(null);

  const update = (id: string, patch: Partial<Mineral>) => {
    onChange(minerals.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  };
  const remove = (id: string) => onChange(minerals.filter((m) => m.id !== id));
  const add = () => onChange([...minerals, { id: genId('min'), name: '', role: 'sulfide', colorHex: '#D9A441' }]);

  const editingMineral = minerals.find((m) => m.id === colorAnchor?.id);

  return (
    <Box>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Минерал</TableCell>
            <TableCell width={170}>Роль</TableCell>
            <TableCell width={80}>Цвет</TableCell>
            <TableCell>Заметка</TableCell>
            <TableCell width={40} />
          </TableRow>
        </TableHead>
        <TableBody>
          {minerals.map((m) => (
            <TableRow key={m.id}>
              <TableCell>
                <TextField
                  size="small"
                  variant="standard"
                  value={m.name}
                  onChange={(e) => update(m.id, { name: e.target.value })}
                  fullWidth
                />
              </TableCell>
              <TableCell>
                <Select
                  size="small"
                  variant="standard"
                  value={m.role}
                  onChange={(e) => update(m.id, { role: e.target.value as MineralRole })}
                  fullWidth
                >
                  {(Object.keys(ROLE_LABELS) as MineralRole[]).map((r) => (
                    <MenuItem key={r} value={r}>
                      {ROLE_LABELS[r]}
                    </MenuItem>
                  ))}
                </Select>
              </TableCell>
              <TableCell>
                <Box
                  onClick={(e) => setColorAnchor({ el: e.currentTarget, id: m.id })}
                  sx={{
                    width: 24,
                    height: 24,
                    borderRadius: '50%',
                    bgcolor: m.colorHex,
                    border: '1px solid rgba(0,0,0,0.2)',
                    cursor: 'pointer',
                  }}
                />
              </TableCell>
              <TableCell>
                <TextField
                  size="small"
                  variant="standard"
                  value={m.note ?? ''}
                  onChange={(e) => update(m.id, { note: e.target.value })}
                  fullWidth
                />
              </TableCell>
              <TableCell>
                <IconButton size="small" onClick={() => remove(m.id)}>
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Button startIcon={<AddIcon />} size="small" sx={{ mt: 1 }} onClick={add}>
        Добавить минерал
      </Button>
      <Popover
        open={Boolean(colorAnchor)}
        anchorEl={colorAnchor?.el}
        onClose={() => setColorAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Box sx={{
          p: 2
        }}>
          {editingMineral && (
            <>
              <HexColorPicker
                color={editingMineral.colorHex}
                onChange={(color) => update(editingMineral.id, { colorHex: color })}
              />
              <TextField
                size="small"
                value={editingMineral.colorHex}
                onChange={(e) => update(editingMineral.id, { colorHex: e.target.value })}
                sx={{ mt: 1 }}
                fullWidth
              />
            </>
          )}
        </Box>
      </Popover>
    </Box>
  );
}
