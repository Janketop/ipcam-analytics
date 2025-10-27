import { useCallback, useEffect, useState } from 'react';
import { Camera, CameraStatus } from '../types/api';

const DEFAULT_STATUS: CameraStatus = 'unknown';
const KNOWN_STATUSES: CameraStatus[] = ['online', 'offline', 'starting', 'stopping', 'no_signal', 'unknown'];

type FetchOptions = {
  silent?: boolean;
};

type WsStatusPayload = {
  cameraId?: unknown;
  camera_id?: unknown;
  id?: unknown;
  camera?: unknown;
  name?: unknown;
  status?: unknown;
  fps?: unknown;
  lastFrameTs?: unknown;
  last_frame_ts?: unknown;
  uptimeSec?: unknown;
  uptime_sec?: unknown;
};

const toNumberOrNull = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return null;
};

const toBoolean = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    return value !== 0;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes', 'y', 'on'].includes(normalized)) {
      return true;
    }
    if (['false', '0', 'no', 'n', 'off'].includes(normalized)) {
      return false;
    }
  }
  return fallback;
};

const parseStatus = (value: unknown): CameraStatus | undefined => {
  if (typeof value !== 'string') {
    return undefined;
  }
  return (KNOWN_STATUSES as readonly string[]).includes(value) ? (value as CameraStatus) : undefined;
};

const mapStatusPayload = (payload: WsStatusPayload | null | undefined) => {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  const cameraId = toNumberOrNull(payload.cameraId ?? payload.camera_id ?? payload.id);
  const cameraName =
    typeof payload.camera === 'string'
      ? payload.camera
      : typeof payload.name === 'string'
        ? payload.name
        : null;
  const status = parseStatus(payload.status);
  const fpsValue = payload.fps !== undefined ? toNumberOrNull(payload.fps) : undefined;
  const uptimeValue = payload.uptimeSec !== undefined || payload.uptime_sec !== undefined
    ? toNumberOrNull(payload.uptimeSec ?? payload.uptime_sec)
    : undefined;
  const lastFrameTs =
    typeof payload.lastFrameTs === 'string'
      ? payload.lastFrameTs
      : typeof payload.last_frame_ts === 'string'
        ? payload.last_frame_ts
        : undefined;

  return {
    cameraId,
    cameraName,
    status,
    fps: fpsValue,
    uptimeSec: uptimeValue,
    lastFrameTs,
  } as const;
};

const normalizeCamera = (raw: unknown): Camera | null => {
  if (!raw || typeof raw !== 'object') {
    return null;
  }

  const obj = raw as Record<string, unknown>;
  const id = obj.id;
  const name = obj.name;
  if (typeof id !== 'number' || typeof name !== 'string') {
    return null;
  }

  const status =
    typeof obj.status === 'string' && (KNOWN_STATUSES as readonly string[]).includes(obj.status)
      ? (obj.status as CameraStatus)
      : DEFAULT_STATUS;
  const fps = toNumberOrNull(obj.fps);
  const uptimeSec = toNumberOrNull(obj.uptimeSec);
  const lastFrameTs = typeof obj.lastFrameTs === 'string' ? obj.lastFrameTs : null;
  const rtspUrl =
    typeof obj.rtspUrl === 'string'
      ? obj.rtspUrl
      : typeof (obj as Record<string, unknown>).rtsp_url === 'string'
        ? ((obj as Record<string, unknown>).rtsp_url as string)
        : '';
  const detectPerson = toBoolean(obj.detectPerson ?? (obj as Record<string, unknown>).detect_person, true);
  const detectCar = toBoolean(obj.detectCar ?? (obj as Record<string, unknown>).detect_car, true);
  const captureEntryTime = toBoolean(
    obj.captureEntryTime ?? (obj as Record<string, unknown>).capture_entry_time,
    true,
  );

  return {
    id,
    name,
    rtspUrl,
    detectPerson,
    detectCar,
    captureEntryTime,
    status,
    fps,
    uptimeSec,
    lastFrameTs,
  };
};

export const useCameras = (apiBase: string) => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCameras = useCallback(async (options?: FetchOptions) => {
    if (!apiBase) return;
    const silent = Boolean(options?.silent);
    if (!silent) {
      setLoading(true);
    }
    setError(null);
    try {
      const response = await fetch(`${apiBase}/cameras`);
      const data = await response.json();
      if (Array.isArray(data?.cameras)) {
        const normalized = data.cameras
          .map(normalizeCamera)
          .filter((camera): camera is Camera => Boolean(camera));
        setCameras(normalized);
      } else {
        setCameras([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось получить список камер.');
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, [apiBase]);

  useEffect(() => {
    fetchCameras();
  }, [fetchCameras]);

  useEffect(() => {
    if (!apiBase) return;

    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    let attempt = 0;

    const scheduleReconnect = () => {
      if (stopped) return;
      const delay = Math.min(30000, 1000 * 2 ** attempt);
      attempt += 1;
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, delay);
    };

    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data) as WsStatusPayload;
        const mapped = mapStatusPayload(data);
        if (!mapped || (mapped.cameraId == null && !mapped.cameraName)) {
          return;
        }

        setCameras(prev => {
          let updated = false;
          const next = prev.map(camera => {
            const matchesId = mapped.cameraId != null && camera.id === mapped.cameraId;
            const matchesName = mapped.cameraName && camera.name === mapped.cameraName;
            if (!matchesId && !matchesName) {
              return camera;
            }

            updated = true;
            return {
              ...camera,
              status: mapped.status ?? camera.status ?? DEFAULT_STATUS,
              fps: mapped.fps !== undefined ? mapped.fps : camera.fps ?? null,
              uptimeSec: mapped.uptimeSec !== undefined ? mapped.uptimeSec : camera.uptimeSec ?? null,
              lastFrameTs: mapped.lastFrameTs !== undefined ? mapped.lastFrameTs : camera.lastFrameTs ?? null,
            };
          });

          if (updated) {
            return next;
          }

          if (mapped.cameraId != null && mapped.cameraName) {
            return [...prev, {
              id: mapped.cameraId,
              name: mapped.cameraName,
              rtspUrl: '',
              detectPerson: true,
              detectCar: true,
              captureEntryTime: true,
              status: mapped.status ?? DEFAULT_STATUS,
              fps: mapped.fps ?? null,
              uptimeSec: mapped.uptimeSec ?? null,
              lastFrameTs: mapped.lastFrameTs ?? null,
            }].sort((a, b) => a.id - b.id);
          }

          return prev;
        });
      } catch (err) {
        console.error('Не удалось обработать статус по WebSocket:', err);
      }
    };

    const connect = () => {
      if (stopped) return;

      try {
        const wsUrl = new URL('/ws/statuses', `${apiBase}/`);
        wsUrl.protocol = wsUrl.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(wsUrl.toString());
      } catch (err) {
        console.error('Не удалось сформировать адрес WebSocket статусов:', err);
        scheduleReconnect();
        return;
      }

      ws.onopen = () => {
        attempt = 0;
        fetchCameras({ silent: true }).catch(err => {
          console.error('Не удалось синхронизировать камеры после подключения WS:', err);
        });
      };

      ws.onmessage = handleMessage;

      ws.onclose = () => {
        if (stopped) return;
        scheduleReconnect();
      };

      ws.onerror = event => {
        console.error('Ошибка WebSocket статусов камер:', event);
        ws?.close();
      };
    };

    const connectTimer = setTimeout(connect, 0);

    return () => {
      stopped = true;
      if (connectTimer) {
        clearTimeout(connectTimer);
      }
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      ws?.close();
    };
  }, [apiBase, fetchCameras]);

  return { cameras, setCameras, loading, error, reload: () => fetchCameras() };
};
