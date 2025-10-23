#!/bin/sh
set -eu

cd /app

log() {
  printf '[frontend] %s\n' "$1"
}

log "Обновляю зависимости npm перед запуском дев-сервера"
if [ -d node_modules ]; then
  log "Удаляю старый каталог node_modules, чтобы пересобрать зависимости"
  rm -rf node_modules
fi

if [ -f package-lock.json ]; then
  npm ci --no-audit --no-fund
else
  npm install --no-audit --no-fund
fi

if [ ! -x node_modules/.bin/next ]; then
  log "Не нашёл бинарник Next.js после установки зависимостей"
  exit 1
fi

log "Зависимости готовы, запускаю команду: $*"
exec "$@"
