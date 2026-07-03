import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Chip from '@mui/material/Chip';
import LinearProgress from '@mui/material/LinearProgress';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import CloseIcon from '@mui/icons-material/Close';
import StarIcon from '@mui/icons-material/Star';
import StarBorderIcon from '@mui/icons-material/StarBorder';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutlineOutlined';
import type { PendingFrame } from './CreateExperimentPage';

interface SortableThumbProps {
  frame: PendingFrame;
  isReference: boolean;
  onRemove: (id: string) => void;
  onSetReference: (id: string) => void;
}

export function SortableThumb({ frame, isReference, onRemove, onSetReference }: SortableThumbProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: frame.id });

  return (
    <Paper
      ref={setNodeRef}
      variant="outlined"
      style={{ transform: CSS.Transform.toString(transform), transition }}
      sx={{
        width: 180,
        overflow: 'hidden',
        opacity: isDragging ? 0.5 : 1,
        borderColor: isReference ? 'primary.main' : undefined,
        borderWidth: isReference ? 2 : 1,
      }}
    >
      <Box sx={{
        position: "relative"
      }}>
        <Box
          component="img"
          src={frame.previewUrl}
          alt={frame.file.name}
          sx={{ width: '100%', height: 120, objectFit: 'cover', display: 'block', bgcolor: '#20242A' }}
        />
        <Box
          {...attributes}
          {...listeners}
          sx={{
            position: 'absolute',
            top: 4,
            left: 4,
            cursor: 'grab',
          }}
        >
          <DragIndicatorIcon fontSize="small" sx={{ color: '#fff', filter: 'drop-shadow(0 0 2px #000)' }} />
        </Box>
        <IconButton
          size="small"
          onClick={() => onRemove(frame.id)}
          sx={{ position: 'absolute', top: 2, right: 2, bgcolor: 'rgba(0,0,0,0.4)', color: '#fff', '&:hover': { bgcolor: 'rgba(0,0,0,0.6)' } }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
        {isReference && (
          <Chip
            icon={<StarIcon sx={{ color: '#fff !important' }} fontSize="small" />}
            label="Опорный"
            size="small"
            color="primary"
            sx={{ position: 'absolute', bottom: 4, left: 4 }}
          />
        )}
      </Box>
      <Box sx={{
        p: 1
      }}>
        <Typography variant="caption" noWrap title={frame.file.name} sx={{
          display: "block"
        }}>
          {frame.file.name}
        </Typography>
        {frame.status === 'importing' && <LinearProgress variant="determinate" value={frame.progress * 100} sx={{ mt: 0.5 }} />}
        {frame.status === 'error' && (
          <Typography
            variant="caption"
            color="error"
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 0.5
            }}>
            <ErrorOutlineIcon fontSize="inherit" /> {frame.error}
          </Typography>
        )}
        {frame.status === 'ready' && !isReference && (
          <IconButton size="small" onClick={() => onSetReference(frame.id)} title="Сделать опорным">
            <StarBorderIcon fontSize="small" />
          </IconButton>
        )}
      </Box>
    </Paper>
  );
}
