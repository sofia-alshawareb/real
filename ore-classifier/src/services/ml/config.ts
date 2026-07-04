/** Base URL for ML API (empty = same-origin, e.g. Vite proxy to localhost:8000). */
export function mlApiBaseUrl(): string {
  const url = import.meta.env.VITE_ML_API_URL as string | undefined;
  if (!url) return '';
  return url.replace(/\/$/, '');
}

export function mlApiUrl(path: string): string {
  const base = mlApiBaseUrl();
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${base}${p}`;
}
