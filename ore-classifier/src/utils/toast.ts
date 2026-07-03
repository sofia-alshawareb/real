import type { VariantType } from 'notistack';

type EnqueueFn = (message: string, options?: { variant?: VariantType }) => void;

let enqueueRef: EnqueueFn | null = null;

export function registerSnackbar(fn: EnqueueFn) {
  enqueueRef = fn;
}

export function notify(message: string, variant: VariantType = 'default') {
  enqueueRef?.(message, { variant });
}
