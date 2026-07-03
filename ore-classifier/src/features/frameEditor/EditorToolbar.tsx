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
import UndoIcon from '@mui/icons-material/Undo';
import RedoIcon from '@mui/icons-material/Redo';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import CompareIcon from '@mui/icons-material/Compare';
import type { EditorTool, MaskClassKey } from '../../stores/editorStore';
import { MASK_CLASSES } from '../../theme/palette';

interface EditorToolbarProps {
  tool: EditorTool;
  onToolChange: (tool: EditorTool) => void;
  activeClass: MaskClassKey;
  onActiveClassChange: (cls: MaskClassKey) => void;
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
  disabled?: boolean;
}

export function EditorToolbar({
  tool,
  onToolChange,
  activeClass,
  onActiveClassChange,
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
  disabled,
}: EditorToolbarProps) {
  return (
    <Paper variant="outlined" sx={{ p: 1.5, width: 96, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
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
      </ToggleButtonGroup>
      <Divider flexItem />
      <Stack spacing={0.75} sx={{
        alignItems: "center"
      }}>
        {(Object.keys(MASK_CLASSES) as MaskClassKey[]).map((key, idx) => (
          <Tooltip key={key} title={`${MASK_CLASSES[key].label} (${idx + 1})`} placement="right">
            <Box
              onClick={() => onActiveClassChange(key)}
              sx={{
                width: 28,
                height: 28,
                borderRadius: '50%',
                bgcolor: MASK_CLASSES[key].color,
                cursor: 'pointer',
                border: activeClass === key ? '3px solid #3B5B7C' : '2px solid #fff',
                boxShadow: '0 0 0 1px rgba(0,0,0,0.2)',
              }}
            />
          </Tooltip>
        ))}
      </Stack>
      <Divider flexItem />
      <Box
        sx={{
          width: "100%",
          px: 0.5
        }}>
        <Typography
          variant="caption"
          align="center"
          sx={{
            color: "text.secondary",
            display: "block"
          }}>
          Кисть
        </Typography>
        <Slider
          size="small"
          orientation="vertical"
          value={brushRadius}
          min={2}
          max={128}
          onChange={(_, v) => onBrushRadiusChange(v as number)}
          sx={{ height: 70, mx: 'auto', display: 'block' }}
        />
      </Box>
      <Box
        sx={{
          width: "100%",
          px: 0.5
        }}>
        <Typography
          variant="caption"
          align="center"
          sx={{
            color: "text.secondary",
            display: "block"
          }}>
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
          sx={{ height: 70, mx: 'auto', display: 'block' }}
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
        <Tooltip title="Сравнить с авторазметкой" placement="right">
          <span>
            <IconButton size="small" color={compareMode ? 'primary' : 'default'} disabled={!hasAutoMask} onClick={onToggleCompare}>
              <CompareIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>
    </Paper>
  );
}
