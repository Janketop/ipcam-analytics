# Аналитика IP-камер (Телефон/Не работает) — Starter (RU)

Готовый стартовый проект: FastAPI (Python), PostgreSQL, Next.js (React).
- Детекция человека/телефона и позы (YOLOv8 + pose)
- События `PHONE_USAGE` и снимки с размытием лица
- MJPEG live-поток с наложенной разметкой (рамки, скелет)
- Дашборд на русском, статистика за 24 часа

## Запуск
```
cp .env.example .env
# укажите RTSP_SOURCES, например:
# RTSP_SOURCES=cam1|rtsp://user:pass@ip:554/stream1
docker compose up --build
```

- API: http://localhost:8000
- Дашборд: http://localhost:3000
- Live MJPEG: http://localhost:8000/stream/<имя_камеры>

### Приватность
- Размытие лиц включено по умолчанию (FACE_BLUR=true)
- Хранение снимков ограничено, события в БД
- Вся терминология и UI — на русском
