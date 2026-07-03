import { useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import IconButton from '@mui/material/IconButton';
import Chip from '@mui/material/Chip';
import Tooltip from '@mui/material/Tooltip';
import { DataGrid, type GridColDef } from '@mui/x-data-grid';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import UnarchiveIcon from '@mui/icons-material/Unarchive';
import ArchiveIcon from '@mui/icons-material/Archive';
import { useDepositsStore } from '../../stores/depositsStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { DepositDialog } from './DepositDialog';
import { ConfirmDialog } from '../../components/ConfirmDialog';
import { EmptyState } from '../../components/EmptyState';
import { formatPercent, formatDateTime } from '../../utils/format';
import { notify } from '../../utils/toast';
import type { Deposit } from '../../types/models';
import MapIcon from '@mui/icons-material/Map';

export function DepositsPage() {
  const deposits = useDepositsStore((s) => s.deposits);
  const addDeposit = useDepositsStore((s) => s.addDeposit);
  const updateDeposit = useDepositsStore((s) => s.updateDeposit);
  const archiveDeposit = useDepositsStore((s) => s.archiveDeposit);
  const deleteDeposit = useDepositsStore((s) => s.deleteDeposit);
  const hasLinkedExperiments = useDepositsStore((s) => s.hasLinkedExperiments);
  const author = useSettingsStore((s) => s.authorName) || 'Без имени';

  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingDeposit, setEditingDeposit] = useState<Deposit | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Deposit | null>(null);

  const rows = useMemo(
    () =>
      deposits.map((d) => ({
        id: d.id,
        name: d.name,
        code: d.code,
        talcThreshold: d.talcThreshold,
        mineralsCount: d.minerals.length,
        archived: d.archived,
        updatedAt: d.updatedAt,
        raw: d,
      })),
    [deposits],
  );

  const columns: GridColDef[] = [
    { field: 'name', headerName: 'Название', flex: 1, minWidth: 180 },
    { field: 'code', headerName: 'Код', width: 100 },
    {
      field: 'talcThreshold',
      headerName: 'Порог талька',
      width: 130,
      valueFormatter: (value: number) => formatPercent(value),
    },
    { field: 'mineralsCount', headerName: 'Минералов', width: 110, align: 'center', headerAlign: 'center' },
    {
      field: 'archived',
      headerName: 'Статус',
      width: 120,
      renderCell: (params) => (params.value ? <Chip label="Архив" size="small" /> : <Chip label="Активно" size="small" color="success" />),
    },
    { field: 'updatedAt', headerName: 'Изменён', width: 160, valueFormatter: (value: number) => formatDateTime(value) },
    {
      field: 'actions',
      headerName: '',
      width: 140,
      sortable: false,
      renderCell: (params) => {
        const deposit = params.row.raw as Deposit;
        return (
          <Stack direction="row">
            <Tooltip title="Редактировать">
              <IconButton
                size="small"
                onClick={() => {
                  setEditingDeposit(deposit);
                  setDialogOpen(true);
                }}
              >
                <EditIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title={deposit.archived ? 'Вернуть из архива' : 'Архивировать'}>
              <IconButton size="small" onClick={() => archiveDeposit(deposit.id, !deposit.archived)}>
                {deposit.archived ? <UnarchiveIcon fontSize="small" /> : <ArchiveIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Удалить">
              <IconButton size="small" onClick={() => setDeleteTarget(deposit)}>
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        );
      },
    },
  ];

  const handleSave = (data: { name: string; code: string; talcThreshold: number; notes?: string; minerals: Deposit['minerals'] }) => {
    if (editingDeposit) {
      updateDeposit(editingDeposit.id, { ...data, updatedBy: author });
      notify('Месторождение обновлено', 'success');
    } else {
      addDeposit({ ...data, updatedBy: author });
      notify('Месторождение добавлено', 'success');
    }
    setDialogOpen(false);
    setEditingDeposit(null);
  };

  const handleDelete = () => {
    if (!deleteTarget) return;
    const ok = deleteDeposit(deleteTarget.id);
    if (!ok) {
      notify('Нельзя удалить: есть связанные эксперименты. Используйте архивирование.', 'warning');
    } else {
      notify('Месторождение удалено', 'success');
    }
    setDeleteTarget(null);
  };

  return (
    <Box>
      <Stack
        direction="row"
        sx={{
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2
        }}>
        <Typography variant="h5">Справочник месторождений</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            setEditingDeposit(null);
            setDialogOpen(true);
          }}
        >
          Новое месторождение
        </Button>
      </Stack>
      {deposits.length > 0 ? (
        <Paper variant="outlined">
          <DataGrid rows={rows} columns={columns} autoHeight disableRowSelectionOnClick />
        </Paper>
      ) : (
        <Paper variant="outlined">
          <EmptyState icon={<MapIcon sx={{ fontSize: 56, opacity: 0.4 }} />} title="Справочник пуст" description="Добавьте первое месторождение." />
        </Paper>
      )}
      <DepositDialog open={dialogOpen} deposit={editingDeposit} onClose={() => setDialogOpen(false)} onSave={handleSave} />
      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="Удалить месторождение?"
        description={
          deleteTarget && hasLinkedExperiments(deleteTarget.id)
            ? 'У этого месторождения есть связанные эксперименты — удаление невозможно. Используйте архивирование.'
            : 'Действие необратимо.'
        }
        confirmLabel="Удалить"
        danger
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </Box>
  );
}
