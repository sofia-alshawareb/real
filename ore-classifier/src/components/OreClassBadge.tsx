import Chip from '@mui/material/Chip';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import LayersIcon from '@mui/icons-material/Layers';
import HelpOutlineIcon from '@mui/icons-material/HelpOutlineOutlined';
import type { OreClass } from '../types/models';
import { ORE_CLASS_META } from '../theme/palette';

const ICONS = {
  check_circle: CheckCircleIcon,
  warning: WarningIcon,
  layers: LayersIcon,
};

interface OreClassBadgeProps {
  oreClass?: OreClass;
  size?: 'small' | 'medium';
}

export function OreClassBadge({ oreClass, size = 'small' }: OreClassBadgeProps) {
  if (!oreClass) {
    return <Chip icon={<HelpOutlineIcon />} label="Не определён" size={size} variant="outlined" />;
  }
  const meta = ORE_CLASS_META[oreClass];
  const Icon = ICONS[meta.icon];
  return (
    <Chip
      icon={<Icon style={{ color: meta.color }} />}
      label={meta.label}
      size={size}
      sx={{
        backgroundColor: meta.bg,
        color: meta.color,
        fontWeight: 600,
        '& .MuiChip-icon': { color: meta.color },
      }}
    />
  );
}
