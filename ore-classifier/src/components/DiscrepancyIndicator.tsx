import Tooltip from '@mui/material/Tooltip';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutlineOutlined';
import Box from '@mui/material/Box';

export function DiscrepancyIndicator({ hasDiscrepancies }: { hasDiscrepancies: boolean }) {
  if (!hasDiscrepancies) return null;
  return (
    <Tooltip title="Классы кадров расходятся — требуется проверка">
      <Box
        sx={{
          display: "inline-flex",
          alignItems: "center",
          color: "warning.main"
        }}>
        <ErrorOutlineIcon fontSize="small" />
      </Box>
    </Tooltip>
  );
}
