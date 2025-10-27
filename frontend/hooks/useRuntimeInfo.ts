import { useCallback, useEffect, useState } from 'react';
import { RuntimeInfo } from '../types/api';

export const useRuntimeInfo = (apiBase: string) => {
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchRuntime = useCallback(async () => {
    if (!apiBase) return;
    try {
      const response = await fetch(`${apiBase}/runtime`);
      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(text && text.trim() ? text.trim() : `HTTP ${response.status}`);
      }
      const data = (await response.json()) as RuntimeInfo;
      setRuntime(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось получить статус системы.');
    }
  }, [apiBase]);

  useEffect(() => {
    let active = true;

    const load = async () => {
      if (!active) return;
      await fetchRuntime();
    };

    load();
    const timer = setInterval(load, 10000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [fetchRuntime]);

  return { runtime, error, refresh: fetchRuntime };
};
