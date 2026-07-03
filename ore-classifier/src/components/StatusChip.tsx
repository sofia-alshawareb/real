import Chip from '@mui/material/Chip';
import type { ExperimentStatus, FrameStatus } from '../types/models';
import { EXPERIMENT_STATUS_META, FRAME_STATUS_META } from '../theme/palette';

export function ExperimentStatusChip({ status }: { status: ExperimentStatus }) {
  const meta = EXPERIMENT_STATUS_META[status];
  return <Chip label={meta.label} color={meta.color} size="small" variant={meta.color === 'default' ? 'outlined' : 'filled'} />;
}

export function FrameStatusChip({ status }: { status: FrameStatus }) {
  const meta = FRAME_STATUS_META[status];
  return <Chip label={meta.label} color={meta.color} size="small" variant={meta.color === 'default' ? 'outlined' : 'filled'} />;
}
