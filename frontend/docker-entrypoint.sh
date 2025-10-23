#!/bin/sh
set -eu

cd /app

log() {
  printf '[frontend] %s\n' "$1"
}

log "Обновляю зависимости npm перед запуском дев-сервера"

LOCK_FILE="package-lock.json"
HASH_FILE="node_modules/.package-lock.hash"
NEEDS_INSTALL=0

if [ ! -d node_modules ]; then
  log "Каталог node_modules отсутствует — необходимо установить зависимости"
  NEEDS_INSTALL=1
elif [ -f "$LOCK_FILE" ]; then
  CURRENT_HASH=$(sha256sum "$LOCK_FILE" | awk '{print $1}')
  if [ -f "$HASH_FILE" ]; then
    SAVED_HASH=$(cat "$HASH_FILE")
  else
    SAVED_HASH=""
  fi

  if [ "$CURRENT_HASH" != "$SAVED_HASH" ]; then
    log "Обнаружены изменения в package-lock.json — обновляю зависимости"
    NEEDS_INSTALL=1
  else
    log "Хеш package-lock.json не изменился — переустановка зависимостей не требуется"
  fi
else
  log "Файл package-lock.json не найден — выполню установку зависимостей"
  NEEDS_INSTALL=1
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
  if [ -f "$LOCK_FILE" ]; then
    npm ci --no-audit --no-fund
    CURRENT_HASH=$(sha256sum "$LOCK_FILE" | awk '{print $1}')
    printf '%s' "$CURRENT_HASH" > "$HASH_FILE"
  else
    npm install --no-audit --no-fund
    rm -f "$HASH_FILE"
  fi
fi

if [ ! -x node_modules/.bin/next ]; then
  log "Не нашёл бинарник Next.js после установки зависимостей"
  exit 1
fi

log "Зависимости готовы, запускаю команду: $*"
exec "$@"
