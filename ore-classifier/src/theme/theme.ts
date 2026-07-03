import { createTheme } from '@mui/material/styles';
import { ruRU as coreRuRU } from '@mui/material/locale';
import { ruRU as dataGridRuRU } from '@mui/x-data-grid/locales';

export const theme = createTheme(
  {
    palette: {
      mode: 'light',
      primary: { main: '#3B5B7C' },
      secondary: { main: '#00695C' },
      background: { default: '#F5F7F9', paper: '#FFFFFF' },
      success: { main: '#2E7D32' },
      warning: { main: '#E65100' },
      error: { main: '#C62828' },
    },
    shape: { borderRadius: 8 },
    typography: {
      fontFamily: [
        '"Inter"',
        '"Segoe UI"',
        'Roboto',
        '"Helvetica Neue"',
        'Arial',
        'sans-serif',
      ].join(','),
      h4: { fontWeight: 600 },
      h5: { fontWeight: 600 },
      h6: { fontWeight: 600 },
    },
    components: {
      MuiTextField: { defaultProps: { size: 'small' } },
      MuiButton: {
        defaultProps: { size: 'medium' },
        styleOverrides: { root: { textTransform: 'none', fontWeight: 500 } },
      },
      MuiChip: { defaultProps: { size: 'small' } },
      MuiTableCell: { defaultProps: { size: 'small' } },
      MuiAppBar: {
        styleOverrides: { root: { boxShadow: 'none', borderBottom: '1px solid #E0E0E0' } },
      },
    },
  },
  coreRuRU,
  dataGridRuRU,
);
