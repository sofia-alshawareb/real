import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import AddIcon from '@mui/icons-material/Add';
import ScienceIcon from '@mui/icons-material/Science';
import { DataGrid, type GridColDef } from '@mui/x-data-grid';
import { useExperimentsStore } from '../../stores/experimentsStore';
import { useDepositsStore } from '../../stores/depositsStore';
import { OreClassBadge } from '../../components/OreClassBadge';
import { ExperimentStatusChip } from '../../components/StatusChip';
import { DiscrepancyIndicator } from '../../components/DiscrepancyIndicator';
import { EmptyState } from '../../components/EmptyState';
import { formatDateTime } from '../../utils/format';
import type { ExperimentStatus, OreClass } from '../../types/models';

export function ExperimentsListPage() {
  const navigate = useNavigate();
  const experiments = useExperimentsStore((s) => s.experiments);
  const deposits = useDepositsStore((s) => s.deposits);
  const [search, setSearch] = useState('');
  const [depositFilter, setDepositFilter] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<ExperimentStatus | null>(null);
  const [classFilter, setClassFilter] = useState<OreClass | null>(null);

  const depositById = useMemo(() => new Map(deposits.map((d) => [d.id, d])), [deposits]);

  const rows = useMemo(() => {
    return experiments
      .filter((e) => (search ? e.title.toLowerCase().includes(search.toLowerCase()) : true))
      .filter((e) => (depositFilter ? e.depositId === depositFilter : true))
      .filter((e) => (statusFilter ? e.status === statusFilter : true))
      .filter((e) => (classFilter ? e.experimentClass === classFilter : true))
      .map((e) => ({
        id: e.id,
        title: e.title,
        depositName: depositById.get(e.depositId)?.name ?? '—',
        experimentClass: e.experimentClass,
        hasDiscrepancies: e.status === 'has_discrepancies',
        frameCount: e.frames.length,
        author: e.author,
        updatedAt: e.updatedAt,
        status: e.status,
      }))
      .sort((a, b) => b.updatedAt - a.updatedAt);
  }, [experiments, depositById, search, depositFilter, statusFilter, classFilter]);

  const columns: GridColDef[] = [
    {
      field: 'title',
      headerName: 'Название',
      flex: 1.4,
      minWidth: 220,
      renderCell: (params) => (
        <Typography
          component="span"
          color="primary"
          sx={{ cursor: 'pointer', fontWeight: 500 }}
          onClick={() => navigate(`/experiments/${params.row.id}`)}
        >
          {params.value}
        </Typography>
      ),
    },
    { field: 'depositName', headerName: 'Месторождение', flex: 1, minWidth: 160 },
    {
      field: 'experimentClass',
      headerName: 'Класс руды',
      flex: 1,
      minWidth: 170,
      renderCell: (params) => (
        <Stack
          direction="row"
          spacing={0.5}
          sx={{
            alignItems: "center",
            height: "100%"
          }}>
          <OreClassBadge oreClass={params.value} />
          <DiscrepancyIndicator hasDiscrepancies={params.row.hasDiscrepancies} />
        </Stack>
      ),
    },
    { field: 'frameCount', headerName: 'Кадров', width: 90, align: 'center', headerAlign: 'center' },
    { field: 'author', headerName: 'Автор', width: 150 },
    {
      field: 'updatedAt',
      headerName: 'Изменён',
      width: 160,
      valueFormatter: (value: number) => formatDateTime(value),
    },
    {
      field: 'status',
      headerName: 'Статус',
      width: 170,
      renderCell: (params) => <ExperimentStatusChip status={params.value} />,
    },
  ];

  const hasAnyExperiments = experiments.length > 0;

  return (
    <Box>
      <Stack
        direction="row"
        sx={{
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2
        }}>
        <Typography variant="h5">Эксперименты</Typography>
        <Button variant="contained" startIcon={<AddIcon />} onClick={() => navigate('/experiments/new')}>
          Новый эксперимент
        </Button>
      </Stack>
      {hasAnyExperiments && (
        <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
          <Stack
            direction="row"
            spacing={2}
            useFlexGap
            sx={{
              flexWrap: "wrap",
              alignItems: "center"
            }}>
            <TextField
              size="small"
              placeholder="Поиск по названию..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              sx={{ minWidth: 240 }}
            />
            <Stack direction="row" spacing={1} useFlexGap sx={{
              flexWrap: "wrap"
            }}>
              {deposits.map((d) => (
                <Chip
                  key={d.id}
                  label={d.name}
                  size="small"
                  color={depositFilter === d.id ? 'primary' : 'default'}
                  onClick={() => setDepositFilter(depositFilter === d.id ? null : d.id)}
                />
              ))}
            </Stack>
            <Stack direction="row" spacing={1}>
              {(['routine', 'hard', 'talc'] as OreClass[]).map((c) => (
                <Chip
                  key={c}
                  label={c === 'routine' ? 'Рядовая' : c === 'hard' ? 'Труднообог.' : 'Оталькованная'}
                  size="small"
                  variant={classFilter === c ? 'filled' : 'outlined'}
                  onClick={() => setClassFilter(classFilter === c ? null : c)}
                />
              ))}
            </Stack>
            <Stack direction="row" spacing={1}>
              {(['has_discrepancies', 'in_progress', 'completed', 'reported', 'draft'] as ExperimentStatus[]).map(
                (s) => (
                  <Chip
                    key={s}
                    label={
                      { has_discrepancies: 'Расхождения', in_progress: 'В работе', completed: 'Завершён', reported: 'Отчёт', draft: 'Черновик' }[
                        s
                      ]
                    }
                    size="small"
                    variant={statusFilter === s ? 'filled' : 'outlined'}
                    onClick={() => setStatusFilter(statusFilter === s ? null : s)}
                  />
                ),
              )}
            </Stack>
          </Stack>
        </Paper>
      )}
      {hasAnyExperiments ? (
        <Paper variant="outlined">
          <DataGrid
            rows={rows}
            columns={columns}
            autoHeight
            disableRowSelectionOnClick
            initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
            pageSizeOptions={[10, 25, 50]}
          />
        </Paper>
      ) : (
        <Paper variant="outlined">
          <EmptyState
            icon={<ScienceIcon sx={{ fontSize: 56, opacity: 0.4 }} />}
            title="Пока нет ни одного эксперимента"
            description="Создайте эксперимент, чтобы загрузить панорамы шлифов и запустить классификацию руды."
            action={
              <Button variant="contained" startIcon={<AddIcon />} onClick={() => navigate('/experiments/new')}>
                Создать первый эксперимент
              </Button>
            }
          />
        </Paper>
      )}
    </Box>
  );
}
