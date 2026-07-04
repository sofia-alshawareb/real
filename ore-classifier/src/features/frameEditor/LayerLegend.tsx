import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Checkbox from '@mui/material/Checkbox';
import Radio from '@mui/material/Radio';
import Box from '@mui/material/Box';
import Tooltip from '@mui/material/Tooltip';
import { MASK_CLASSES, type MaskClassKey } from '../../theme/palette';
import { formatPercent } from '../../utils/format';
import type { FrameMetrics } from '../../types/models';

interface LayerLegendProps {
  visibleLayers: Record<MaskClassKey, boolean>;
  onToggle: (key: MaskClassKey) => void;
  metrics?: FrameMetrics;
  activeClass: MaskClassKey;
  onActiveClassChange: (key: MaskClassKey) => void;
}

const METRIC_KEY: Record<MaskClassKey, keyof FrameMetrics> = {
  coarse: 'coarseFraction',
  fine: 'fineFraction',
  talc: 'talcFraction',
  matrix: 'matrixFraction',
};

const MASK_CLASS_KEYS = Object.keys(MASK_CLASSES) as MaskClassKey[];

export function LayerLegend({ visibleLayers, onToggle, metrics, activeClass, onActiveClassChange }: LayerLegendProps) {
  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Слои маски и цвет кисти
      </Typography>
      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
        Выберите класс переключателем слева, чтобы им рисовать — цвет и название всегда рядом.
      </Typography>
      <Stack spacing={0.5}>
        {MASK_CLASS_KEYS.map((key) => {
          const metricKey = METRIC_KEY[key];
          const value = metrics ? metrics[metricKey] : undefined;
          return (
            <Stack
              key={key}
              direction="row"
              spacing={0.5}
              sx={{
                alignItems: 'center',
                borderRadius: 1,
                bgcolor: activeClass === key ? 'action.selected' : 'transparent',
                pr: 1,
              }}
            >
              <Tooltip title="Рисовать этим классом">
                <Radio
                  size="small"
                  checked={activeClass === key}
                  onChange={() => onActiveClassChange(key)}
                  sx={{ color: MASK_CLASSES[key].color, '&.Mui-checked': { color: MASK_CLASSES[key].color } }}
                />
              </Tooltip>
              <Tooltip title="Показать/скрыть слой">
                <Checkbox checked={visibleLayers[key]} onChange={() => onToggle(key)} size="small" />
              </Tooltip>
              <Box
                sx={{
                  width: 14,
                  height: 14,
                  borderRadius: '50%',
                  bgcolor: MASK_CLASSES[key].color,
                  flexShrink: 0,
                }}
              />
              <Typography variant="body2" sx={{ flexGrow: 1 }}>
                {MASK_CLASSES[key].label}
              </Typography>
              {value !== undefined && (
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  {formatPercent(value)}
                </Typography>
              )}
            </Stack>
          );
        })}
      </Stack>
    </Paper>
  );
}
