#!/usr/bin/env bash
# Expose the Vite dev server (port 5173) on a temporary public HTTPS URL via Cloudflare.
# Requires: frontend running (npm run dev), backend running (python -m services.api.main)
set -euo pipefail

CLOUDFLARED="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
PORT="${PORT:-5173}"

if ! command -v "$CLOUDFLARED" >/dev/null 2>&1; then
  echo "Installing cloudflared to $HOME/.local/bin ..."
  mkdir -p "$HOME/.local/bin"
  curl -fsSL -o "$CLOUDFLARED" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$CLOUDFLARED"
fi

if ! curl -sf "http://127.0.0.1:${PORT}/" >/dev/null; then
  echo "Error: nothing listening on http://127.0.0.1:${PORT}/ — start the frontend first:"
  echo "  cd ore-classifier && npm run dev"
  exit 1
fi

echo "Starting public tunnel → http://127.0.0.1:${PORT}"
echo "Share the https://….trycloudflare.com URL printed below."
echo "Press Ctrl+C to stop."
exec "$CLOUDFLARED" tunnel --url "http://127.0.0.1:${PORT}" --protocol http2
