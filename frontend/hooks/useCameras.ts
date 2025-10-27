import { useCallback, useEffect, useState } from 'react';
import { Camera, CameraStatus } from '../types/api';

const DEFAULT_STATUS: CameraStatus = 'unknown';
const KNOWN_STATUSES: CameraStatus[] = ['online', 'offline', 'starting', 'stopping', 'no_signal', 'unknown'];

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

  return {
    id,
    name,
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

  const fetchCameras = useCallback(async () => {
    if (!apiBase) return;
    setLoading(true);
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
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchCameras();
  }, [fetchCameras]);

  return { cameras, setCameras, loading, error, reload: fetchCameras };
};
