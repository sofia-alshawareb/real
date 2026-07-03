import { useEffect } from 'react';
import { useSnackbar } from 'notistack';
import { registerSnackbar } from '../utils/toast';

export function SnackbarBridge() {
  const { enqueueSnackbar } = useSnackbar();
  useEffect(() => {
    registerSnackbar(enqueueSnackbar);
  }, [enqueueSnackbar]);
  return null;
}
