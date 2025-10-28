import { useCallback, useEffect, useState } from 'react';
import { CleanupRunResponse, CleanupSettings, CleanupState, HealthResponse } from '../types/api';

export const useCleanupInfo = (apiBase: string) => {
  const [cleanup, setCleanup] = useState<CleanupState | null>(null);
  const [settings, setSettings] = useState<CleanupSettings | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runInProgress, setRunInProgress] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [runSuccess, setRunSuccess] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!apiBase) return;
    setLoading(true);
    try {
      const response = await fetch(`${apiBase}/health`);
      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(text && text.trim() ? text.trim() : `HTTP ${response.status}`);
      }
      const data = (await response.json()) as HealthResponse;
      setCleanup(data.cleanup ?? null);
      setSettings({
        retentionDays: data.retention_days,
        cleanupIntervalHours: data.cleanup_interval_hours,
        faceSampleRetentionDays: data.face_sample_unverified_retention_days,
        faceBlurEnabled: data.face_blur,
      });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось получить состояние очистки.');
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const runCleanup = useCallback(async () => {
    if (!apiBase) {
      setRunError('API не инициализирован. Проверьте настройки подключения.');
      return;
    }
    setRunInProgress(true);
    setRunError(null);
    setRunSuccess(null);
    try {
      const response = await fetch(`${apiBase}/cleanup/run`, { method: 'POST' });
      if (!response.ok) {
        let detail = response.statusText;
        try {
          const data = (await response.json()) as { detail?: string };
          if (data?.detail) {
            detail = data.detail;
          }
        } catch (err) {
          // ignore JSON parsing errors, detail already set
        }
        throw new Error(detail || 'Не удалось выполнить очистку.');
      }
      const data = (await response.json()) as CleanupRunResponse;
      setCleanup(data.cleanup ?? null);
      setRunSuccess('Очистка успешно завершена.');
      setError(null);
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Не удалось запустить очистку.');
    } finally {
      setRunInProgress(false);
    }
  }, [apiBase]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    cleanup,
    settings,
    loading,
    error,
    refresh,
    runCleanup,
    runInProgress,
    runError,
    runSuccess,
  };
};
