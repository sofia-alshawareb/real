import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useDropzone } from 'react-dropzone';
import { DndContext, closestCenter, type DragEndEvent } from '@dnd-kit/core';
import { SortableContext, arrayMove, rectSortingStrategy } from '@dnd-kit/sortable';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { useDepositsStore } from '../../stores/depositsStore';
import { useExperimentsStore, genId } from '../../stores/experimentsStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { importImageFile, UnsupportedImageError } from '../../services/tiling/tileImporter';
import { enqueueFrame } from '../../services/mockMl/queueRunner';
import { SortableThumb } from './SortableThumb';
import type { Frame } from '../../types/models';
import { notify } from '../../utils/toast';

export interface PendingFrame {
  id: string;
  file: File;
  previewUrl: string;
  status: 'importing' | 'ready' | 'error';
  progress: number;
  imageId?: string;
  width?: number;
  height?: number;
  error?: string;
}

const schema = z.object({
  title: z.string().min(3, 'Минимум 3 символа'),
  depositId: z.string().min(1, 'Выберите месторождение'),
});

type FormValues = z.infer<typeof schema>;

const DEFAULT_PIXEL_SIZE_UM = 0.5;

export function CreateExperimentPage() {
  const navigate = useNavigate();
  const allDeposits = useDepositsStore((s) => s.deposits);
  const deposits = useMemo(() => allDeposits.filter((d) => !d.archived), [allDeposits]);
  const authorName = useSettingsStore((s) => s.authorName) || 'Без имени';
  const createExperiment = useExperimentsStore((s) => s.createExperiment);
  const addFrames = useExperimentsStore((s) => s.addFrames);

  const [frames, setFrames] = useState<PendingFrame[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const {
    control,
    handleSubmit,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { title: '', depositId: '' },
  });

  const onDrop = (accepted: File[]) => {
    const newEntries: PendingFrame[] = accepted.map((file) => ({
      id: genId('pending'),
      file,
      previewUrl: URL.createObjectURL(file),
      status: 'importing',
      progress: 0,
    }));
    setFrames((prev) => [...prev, ...newEntries]);

    newEntries.forEach((entry) => {
      importImageFile(entry.file, (share) => {
        setFrames((prev) => prev.map((f) => (f.id === entry.id ? { ...f, progress: share } : f)));
      })
        .then((result) => {
          setFrames((prev) =>
            prev.map((f) =>
              f.id === entry.id
                ? { ...f, status: 'ready', progress: 1, imageId: result.imageId, width: result.width, height: result.height }
                : f,
            ),
          );
          if (result.downscaled) {
            notify(`«${entry.file.name}» уменьшено до допустимого разрешения браузера`, 'info');
          }
        })
        .catch((err) => {
          const message = err instanceof UnsupportedImageError ? err.message : 'Не удалось обработать изображение';
          setFrames((prev) => prev.map((f) => (f.id === entry.id ? { ...f, status: 'error', error: message } : f)));
        });
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'], 'image/webp': ['.webp'] },
    maxSize: 500 * 1024 * 1024,
  });

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    setFrames((prev) => {
      const oldIndex = prev.findIndex((f) => f.id === active.id);
      const newIndex = prev.findIndex((f) => f.id === over.id);
      return arrayMove(prev, oldIndex, newIndex);
    });
  };

  const handleRemove = (id: string) => setFrames((prev) => prev.filter((f) => f.id !== id));

  const handleSetReference = (id: string) => {
    setFrames((prev) => {
      const idx = prev.findIndex((f) => f.id === id);
      if (idx <= 0) return prev;
      return arrayMove(prev, idx, 0);
    });
  };

  const readyFrames = frames.filter((f) => f.status === 'ready');
  const isImporting = frames.some((f) => f.status === 'importing');

  const onSubmit = async (values: FormValues) => {
    if (readyFrames.length === 0) {
      notify('Добавьте хотя бы один обработанный кадр', 'warning');
      return;
    }
    setSubmitting(true);
    try {
      const experiment = createExperiment({ title: values.title, depositId: values.depositId, author: authorName });
      const modelFrames: Frame[] = readyFrames.map((f, idx) => ({
        id: genId('frame'),
        index: idx,
        name: f.file.name,
        source: { kind: 'dexie', imageId: f.imageId! },
        width: f.width!,
        height: f.height!,
        pixelSizeUm: DEFAULT_PIXEL_SIZE_UM,
        status: 'queued',
        isReference: idx === 0,
        manuallyEditedMask: false,
        updatedAt: Date.now(),
      }));
      addFrames(experiment.id, modelFrames);
      modelFrames.forEach((f) => enqueueFrame(experiment.id, f.id));
      notify('Эксперимент создан, кадры поставлены в очередь на анализ', 'success');
      navigate(`/experiments/${experiment.id}`);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box sx={{
      maxWidth: 900
    }}>
      <Typography variant="h5" sx={{
        mb: 2
      }}>
        Новый эксперимент
      </Typography>
      <Paper variant="outlined" sx={{ p: 3 }}>
        <Stack
          spacing={2}
          sx={{
            maxWidth: 480,
            mb: 3
          }}>
          <Controller
            name="title"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                label="Название эксперимента"
                placeholder="Партия 1042 — смена 2"
                error={Boolean(errors.title)}
                helperText={errors.title?.message}
                fullWidth
              />
            )}
          />
          <Controller
            name="depositId"
            control={control}
            render={({ field }) => (
              <Autocomplete
                options={deposits}
                getOptionLabel={(d) => d.name}
                value={deposits.find((d) => d.id === field.value) ?? null}
                onChange={(_, v) => field.onChange(v?.id ?? '')}
                renderInput={(params) => (
                  <TextField {...params} label="Месторождение" error={Boolean(errors.depositId)} helperText={errors.depositId?.message} />
                )}
              />
            )}
          />
          <TextField label="Автор" value={authorName} disabled fullWidth />
        </Stack>

        <Typography variant="subtitle1" sx={{
          mb: 1
        }}>
          Изображения шлифов
        </Typography>
        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'divider',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            cursor: 'pointer',
            bgcolor: isDragActive ? 'action.hover' : 'transparent',
            mb: 3,
          }}
        >
          <input {...getInputProps()} />
          <UploadFileIcon sx={{ fontSize: 40, color: 'text.secondary', mb: 1 }} />
          <Typography>Перетащите панорамы OM сюда или нажмите, чтобы выбрать</Typography>
          <Typography variant="caption" sx={{
            color: "text.secondary"
          }}>
            PNG, JPEG, WebP · до 500 МБ на файл · до 12 кадров на эксперимент
          </Typography>
        </Box>

        {frames.length > 0 && (
          <DndContext collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
            <SortableContext items={frames.map((f) => f.id)} strategy={rectSortingStrategy}>
              <Stack
                direction="row"
                sx={{
                  flexWrap: "wrap",
                  gap: 2,
                  mb: 2
                }}>
                {frames.map((f, idx) => (
                  <SortableThumb
                    key={f.id}
                    frame={f}
                    isReference={idx === 0}
                    onRemove={handleRemove}
                    onSetReference={handleSetReference}
                  />
                ))}
              </Stack>
            </SortableContext>
          </DndContext>
        )}

        {frames.some((f) => f.status === 'error') && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Часть файлов не удалось обработать браузером. Удалите их или замените на PNG/JPEG/WebP меньшего размера.
          </Alert>
        )}

        <Stack direction="row" spacing={2} sx={{
          justifyContent: "flex-end"
        }}>
          <Button onClick={() => navigate('/')}>Отмена</Button>
          <Button
            variant="contained"
            disabled={submitting || isImporting || readyFrames.length === 0}
            onClick={handleSubmit(onSubmit)}
          >
            {isImporting ? 'Обработка изображений...' : 'Создать эксперимент'}
          </Button>
        </Stack>
      </Paper>
    </Box>
  );
}
