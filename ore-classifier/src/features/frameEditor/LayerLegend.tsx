import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Checkbox from '@mui/material/Checkbox';
import Box from '@mui/material/Box';
import { MASK_CLASSES } from '../../theme/palette';
import type { MaskClassKey } from '../../stores/editorStore';
import { formatPercent } from '../../utils/format';
import type { FrameMetrics } from '../../types/models';

interface LayerLegendProps {
  visibleLayers: Record<MaskClassKey, boolean>;
  onToggle: (key: MaskClassKey) => void;
  metrics?: FrameMetrics;
}

const METRIC_KEY: Record<MaskClassKey, keyof FrameMetrics | null> = {
  sulfide: 'sulfideFraction',
  gangue: 'gangueFraction',
  talc: 'talcFraction',
};

export function LayerLegend({ visibleLayers, onToggle, metrics }: LayerLegendProps) {
  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Слои маски
      </Typography>
      <Stack spacing={0.5}>
        {(Object.keys(MASK_CLASSES) as MaskClassKey[]).map((key) => {
          const metricKey = METRIC_KEY[key];
          const value = metricKey && metrics ? metrics[metricKey] : undefined;
          return (
            <Stack key={key} direction="row" spacing={1} sx={{
              alignItems: "center"
            }}>
              <Checkbox checked={visibleLayers[key]} onChange={() => onToggle(key)} size="small" />
              <Box
                sx={{
                  width: 14,
                  height: 14,
                  borderRadius: "50%",
                  bgcolor: MASK_CLASSES[key].color,
                  flexShrink: 0
                }} />
              <Typography variant="body2" sx={{
                flexGrow: 1
              }}>
                {MASK_CLASSES[key].label}
              </Typography>
              {value !== undefined && (
                <Typography variant="body2" sx={{
                  color: "text.secondary"
                }}>
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
