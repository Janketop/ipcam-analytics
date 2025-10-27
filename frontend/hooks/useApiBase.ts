import { useCallback, useEffect, useMemo, useState } from 'react';

const trimTrailingSlash = (value: string) => value.replace(/\/$/, '');

export const resolveApiBase = () => {
  const envBase = process.env.NEXT_PUBLIC_API_BASE;
  if (envBase && envBase.trim()) {
    return trimTrailingSlash(envBase.trim());
  }

  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const host = window.location.hostname;
    const port = process.env.NEXT_PUBLIC_API_PORT || '8000';
    return `${protocol}//${host}${port ? `:${port}` : ''}`;
  }

  return 'http://localhost:8000';
};

export const useApiBase = () => {
  const computeBase = useCallback(() => resolveApiBase(), []);
  const [apiBase, setApiBase] = useState<string>(() => computeBase());

  useEffect(() => {
    setApiBase(computeBase());
  }, [computeBase]);

  const normalizedApiBase = useMemo(() => trimTrailingSlash(apiBase), [apiBase]);

  const buildAbsoluteUrl = useCallback(
    (path?: string | null) => {
      if (!path) return null;
      if (/^https?:\/\//i.test(path)) {
        return path;
      }
      const normalizedPath = path.startsWith('/') ? path : `/${path}`;
      return `${normalizedApiBase}${normalizedPath}`;
    },
    [normalizedApiBase]
  );

  const refreshApiBase = useCallback(() => {
    setApiBase(computeBase());
  }, [computeBase]);

  return { apiBase, normalizedApiBase, buildAbsoluteUrl, refreshApiBase };
};
