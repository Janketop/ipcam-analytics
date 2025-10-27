import { useCallback, useEffect, useState } from 'react';
import { Camera } from '../types/api';

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
      setCameras(Array.isArray(data?.cameras) ? data.cameras : []);
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
