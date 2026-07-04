import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import ToggleButton from '@mui/material/ToggleButton';
import Tooltip from '@mui/material/Tooltip';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Slider from '@mui/material/Slider';
import Divider from '@mui/material/Divider';
import Box from '@mui/material/Box';
import PanToolIcon from '@mui/icons-material/PanTool';
import BrushIcon from '@mui/icons-material/Brush';
import AutoFixOffIcon from '@mui/icons-material/AutoFixOff';
import PentagonIcon from '@mui/icons-material/Pentagon';
import GestureIcon from '@mui/icons-material/Gesture';
import FormatColorFillIcon from '@mui/icons-material/FormatColorFill';
import UndoIcon from '@mui/icons-material/Undo';
import RedoIcon from '@mui/icons-material/Redo';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import CompareIcon from '@mui/icons-material/Compare';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import ImageIcon from '@mui/icons-material/Image';
import LayersIcon from '@mui/icons-material/Layers';
import VisibilityIcon from '@mui/icons-material/Visibility';
import type { EditorTool } from '../../stores/editorStore';
import { BRUSH_RADIUS_RANGE } from '../../stores/editorStore';

interface EditorToolbarProps {
  tool: EditorTool;
  onToolChange: (tool: EditorTool) => void;
  brushRadius: number;
  onBrushRadiusChange: (v: number) => void;
  overlayOpacity: number;
  onOverlayOpacityChange: (v: number) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onRevertToAuto: () => void;
  hasAutoMask: boolean;
  compareMode: boolean;
  onToggleCompare: () => void;
  viewMode: 'overlay' | 'original' | 'mask';
  onViewModeChange: (mode: 'overlay' | 'original' | 'mask') => void;
  onClearMask: () => void;
  disabled?: boolean;
}

export function EditorToolbar({
  tool,
  onToolChange,
  brushRadius,
  onBrushRadiusChange,
  overlayOpacity,
  onOverlayOpacityChange,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onRevertToAuto,
  hasAutoMask,
  compareMode,
  onToggleCompare,
  viewMode,
  onViewModeChange,
  onClearMask,
  disabled,
}: EditorToolbarProps) {
  const brushPercent = Math.round((brushRadius / BRUSH_RADIUS_RANGE.max) * 100);

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 1.5,
        width: 104,
        maxHeight: '100%',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 1.5,
      }}
    >
      <ToggleButtonGroup
        orientation="vertical"
        exclusive
        value={tool}
        onChange={(_, v) => v && onToolChange(v)}
        size="small"
        disabled={disabled}
      >
        <ToggleButton value="pan">
          <Tooltip title="Пан (V)" placement="right">
            <PanToolIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="brush">
          <Tooltip title="Кисть (B)" placement="right">
            <BrushIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="eraser">
          <Tooltip title="Ластик (E)" placement="right">
            <AutoFixOffIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="polygon">
          <Tooltip title="Полигон (P)" placement="right">
            <PentagonIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="fill">
          <Tooltip title="Заливка: клик закрашивает смежную область активным классом, Alt+клик — удаляет (G)" placement="right">
            <FormatColorFillIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="lasso">
          <Tooltip title="Лассо: обвести область от руки и залить/удалить обведённое (L)" placement="right">
            <GestureIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
      </ToggleButtonGroup>
      <Divider flexItem />
      <Box sx={{ width: '100%', px: 0.5 }}>
        <Typography variant="caption" align="center" sx={{ color: 'text.secondary', display: 'block' }}>
          Кисть {brushPercent}%
        </Typography>
        <Slider
          size="small"
          orientation="vertical"
          value={brushRadius}
          min={BRUSH_RADIUS_RANGE.min}
          max={BRUSH_RADIUS_RANGE.max}
          onChange={(_, v) => onBrushRadiusChange(v as number)}
          sx={{ height: 64, mx: 'auto', display: 'block' }}
        />
      </Box>
      <Box sx={{ width: '100%', px: 0.5 }}>
        <Typography variant="caption" align="center" sx={{ color: 'text.secondary', display: 'block' }}>
          Слои
        </Typography>
        <Slider
          size="small"
          orientation="vertical"
          value={overlayOpacity}
          min={0}
          max={1}
          step={0.05}
          onChange={(_, v) => onOverlayOpacityChange(v as number)}
          sx={{ height: 64, mx: 'auto', display: 'block' }}
        />
      </Box>
      <Divider flexItem />
      <Stack spacing={0.5}>
        <Tooltip title="Отменить (Ctrl+Z)" placement="right">
          <span>
            <IconButton size="small" disabled={!canUndo} onClick={onUndo}>
              <UndoIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Повторить (Ctrl+Y)" placement="right">
          <span>
            <IconButton size="small" disabled={!canRedo} onClick={onRedo}>
              <RedoIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Вернуть автосегментацию" placement="right">
          <span>
            <IconButton size="small" disabled={!hasAutoMask} onClick={onRevertToAuto}>
              <RestartAltIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Очистить маску" placement="right">
          <span>
            <IconButton size="small" onClick={onClearMask}>
              <DeleteSweepIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Сравнить с авторазметкой" placement="right">
          <span>
            <IconButton size="small" color={compareMode ? 'primary' : 'default'} disabled={!hasAutoMask} onClick={onToggleCompare}>
              <CompareIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>
      <Divider flexItem />
      <Stack spacing={0.5}>
        <Tooltip title="Только исходное изображение" placement="right">
          <span>
            <IconButton size="small" color={viewMode === 'original' ? 'primary' : 'default'} onClick={() => onViewModeChange('original')}>
              <ImageIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Изображение + маска (обычный режим)" placement="right">
          <span>
            <IconButton size="small" color={viewMode === 'overlay' ? 'primary' : 'default'} onClick={() => onViewModeChange('overlay')}>
              <VisibilityIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Только маска" placement="right">
          <span>
            <IconButton size="small" color={viewMode === 'mask' ? 'primary' : 'default'} onClick={() => onViewModeChange('mask')}>
              <LayersIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>
    </Paper>
  );
}
