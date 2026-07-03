import Stack from '@mui/material/Stack';
import Paper from '@mui/material/Paper';
import Checkbox from '@mui/material/Checkbox';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import { FrameThumbnail } from '../../components/FrameThumbnail';
import { OreClassBadge } from '../../components/OreClassBadge';
import type { Frame } from '../../types/models';

interface FramePickerProps {
  frames: Frame[];
  includedIds: string[];
  onToggle: (frameId: string) => void;
}

export function FramePicker({ frames, includedIds, onToggle }: FramePickerProps) {
  return (
    <Stack spacing={1}>
      {frames.map((f) => (
        <Paper key={f.id} variant="outlined" sx={{ p: 1, display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Checkbox checked={includedIds.includes(f.id)} onChange={() => onToggle(f.id)} />
          <Box
            sx={{
              width: 80,
              flexShrink: 0
            }}>
            <FrameThumbnail frame={f} width={80} height={50} />
          </Box>
          <Box sx={{
            flexGrow: 1
          }}>
            <Typography variant="body2">{f.name}</Typography>
          </Box>
          <OreClassBadge oreClass={f.manualClassOverride ?? f.frameClass} />
        </Paper>
      ))}
    </Stack>
  );
}
