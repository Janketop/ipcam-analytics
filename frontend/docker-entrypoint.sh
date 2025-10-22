#!/bin/sh
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
