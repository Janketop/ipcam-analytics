#!/bin/sh
set -eu

cd /app

echo "[frontend] Ensuring npm dependencies are installed..."
if [ -d node_modules ]; then
  echo "[frontend] Existing node_modules directory detected, cleaning before reinstall"
  rm -rf node_modules
fi

if [ -f package-lock.json ]; then
  npm ci --no-audit --no-fund
else
  npm install --no-audit --no-fund
fi

if [ ! -x node_modules/.bin/next ]; then
  echo "[frontend] ERROR: Next.js binary not found after installation" >&2
  exit 1
fi

echo "[frontend] Dependencies ready, starting: $*"
set -e

cd /app

if [ ! -x node_modules/.bin/next ]; then
  echo "[frontend] node_modules missing or Next.js binary unavailable, installing dependencies..."
  rm -rf node_modules
  if [ -f package-lock.json ]; then
    npm ci
  else
    npm install
  fi
else
  echo "[frontend] Reusing existing node_modules cache"
fi

exec "$@"
