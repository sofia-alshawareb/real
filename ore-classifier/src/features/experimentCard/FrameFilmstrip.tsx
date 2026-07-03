import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import StarIcon from '@mui/icons-material/Star';
import StarBorderIcon from '@mui/icons-material/StarBorder';
import RefreshIcon from '@mui/icons-material/Refresh';
import EditIcon from '@mui/icons-material/Edit';
import { useNavigate } from 'react-router-dom';
import { FrameThumbnail } from '../../components/FrameThumbnail';
import { OreClassBadge } from '../../components/OreClassBadge';
import { FrameStatusChip } from '../../components/StatusChip';
import type { Experiment, Frame } from '../../types/models';
import { ORE_CLASS_META } from '../../theme/palette';
import { retryFrame } from '../../services/mockMl/queueRunner';

interface FrameFilmstripProps {
  experiment: Experiment;
  onSetReference: (frameId: string) => void;
}

export function FrameFilmstrip({ experiment, onSetReference }: FrameFilmstripProps) {
  const navigate = useNavigate();

  return (
    <Stack direction="row" spacing={2} sx={{ overflowX: 'auto', pb: 1 }}>
      {experiment.frames.map((frame: Frame) => {
        const effectiveClass = frame.manualClassOverride ?? frame.frameClass;
        const borderColor = effectiveClass ? ORE_CLASS_META[effectiveClass].color : 'divider';
        return (
          <Paper
            key={frame.id}
            variant="outlined"
            sx={{
              width: 190,
              flexShrink: 0,
              borderColor,
              borderWidth: effectiveClass ? 2 : 1,
              cursor: 'pointer',
              transition: 'transform .15s',
              '&:hover': { transform: 'translateY(-2px)' },
            }}
            onClick={() => navigate(`/experiments/${experiment.id}/frames/${frame.id}`)}
          >
            <Box sx={{
              position: "relative"
            }}>
              <FrameThumbnail frame={frame} width={190} height={120} />
              {frame.isReference && (
                <Tooltip title="Опорный кадр">
                  <StarIcon
                    fontSize="small"
                    sx={{ position: 'absolute', top: 4, left: 4, color: '#FFD54F', filter: 'drop-shadow(0 0 2px #000)' }}
                  />
                </Tooltip>
              )}
              {frame.manuallyEditedMask && (
                <Tooltip title="Сегментация отредактирована вручную">
                  <EditIcon
                    fontSize="small"
                    sx={{ position: 'absolute', top: 4, right: 4, color: '#fff', filter: 'drop-shadow(0 0 2px #000)' }}
                  />
                </Tooltip>
              )}
            </Box>
            <Box sx={{
              p: 1
            }}>
              <Typography variant="caption" noWrap title={frame.name} sx={{
                display: "block"
              }}>
                {frame.name}
              </Typography>
              <Stack
                direction="row"
                spacing={0.5}
                sx={{
                  alignItems: "center",
                  mt: 0.5,
                  flexWrap: "wrap"
                }}>
                <FrameStatusChip status={frame.status} />
              </Stack>
              <Stack
                direction="row"
                spacing={0.5}
                sx={{
                  alignItems: "center",
                  mt: 0.5,
                  justifyContent: "space-between"
                }}>
                <OreClassBadge oreClass={effectiveClass} />
                <Stack direction="row">
                  {frame.status === 'ml_unavailable' && (
                    <Tooltip title="Повторить обработку">
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          retryFrame(experiment.id, frame.id);
                        }}
                      >
                        <RefreshIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                  {!frame.isReference && (
                    <Tooltip title="Сделать опорным">
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSetReference(frame.id);
                        }}
                      >
                        <StarBorderIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                </Stack>
              </Stack>
            </Box>
          </Paper>
        );
      })}
    </Stack>
  );
}
