import { useEffect, useState } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import { SnackbarProvider } from 'notistack';
import { RouterProvider } from 'react-router-dom';
import { theme } from './theme/theme';
import { router } from './routes';
import { bootstrapApp } from './services/bootstrap';
import { SnackbarBridge } from './components/SnackbarBridge';

function App() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    void bootstrapApp().then(() => setReady(true));
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SnackbarProvider maxSnack={3} anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}>
        <SnackbarBridge />
        {ready ? (
          <RouterProvider router={router} />
        ) : (
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              minHeight: "100vh",
              flexDirection: "column",
              gap: 2
            }}>
            <Stack spacing={2} sx={{
              alignItems: "center"
            }}>
              <CircularProgress />
              <Typography sx={{
                color: "text.secondary"
              }}>Инициализация демо-данных...</Typography>
            </Stack>
          </Box>
        )}
      </SnackbarProvider>
    </ThemeProvider>
  );
}

export default App;
