#!/bin/sh
set -eu

cd /app

log() {
  # Пишем сообщения как в stdout, так и в stderr — это гарантирует,
  # что Docker не "проглотит" их, даже если stdout уже занят npm.
  printf '[frontend] %s\n' "$1"
  printf '[frontend] %s\n' "$1" >&2
}

log "Проверяю, нужно ли обновлять npm-зависимости перед запуском дев-сервера"

LOCK_FILE="package-lock.json"
HASH_FILE="node_modules/.package-lock.hash"
NEXT_BIN="node_modules/.bin/next"
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

if [ "$NEEDS_INSTALL" -eq 0 ] && [ -d node_modules ] && [ ! -x "$NEXT_BIN" ]; then
  log "Бинарник Next.js отсутствует, несмотря на кеш — переустанавливаю зависимости"
  NEEDS_INSTALL=1
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
  if [ -f "$LOCK_FILE" ]; then
    log "Запускаю npm ci для установки зависимостей"
    if npm ci --no-audit --no-fund; then
      CURRENT_HASH=$(sha256sum "$LOCK_FILE" | awk '{print $1}')
      printf '%s' "$CURRENT_HASH" > "$HASH_FILE"
    else
      log "npm ci завершился с ошибкой, попробую npm install"
      npm install --no-audit --no-fund
      rm -f "$HASH_FILE"
    fi
  else
    log "Файл package-lock.json отсутствует, выполняю npm install"
    npm install --no-audit --no-fund
    rm -f "$HASH_FILE"
  fi
fi

if [ ! -x "$NEXT_BIN" ]; then
  log "Не нашёл бинарник Next.js после установки зависимостей"
  exit 1
fi

log "Зависимости готовы, запускаю команду: $*"
exec "$@"
