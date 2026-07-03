import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import type { ReactNode } from 'react';

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        py: 8,
        px: 3,
        color: "text.secondary"
      }}>
      <Stack
        spacing={2}
        sx={{
          alignItems: "center",
          maxWidth: 420
        }}>
        {icon}
        <Typography variant="h6" sx={{
          color: "text.primary"
        }}>
          {title}
        </Typography>
        {description && <Typography variant="body2">{description}</Typography>}
        {action}
      </Stack>
    </Box>
  );
}
