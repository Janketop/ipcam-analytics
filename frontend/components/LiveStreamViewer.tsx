import { ChangeEvent, useCallback, useEffect, useMemo, useState } from 'react';
import { Camera } from '../types/api';

type LiveStreamViewerProps = {
  cameras: Camera[];
  normalizedApiBase: string;
};

export const LiveStreamViewer = ({ cameras, normalizedApiBase }: LiveStreamViewerProps) => {
  const [liveStreamOverride, setLiveStreamOverride] = useState<string | null>(null);
  const [liveExpanded, setLiveExpanded] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(() => cameras[0]?.name ?? null);

  useEffect(() => {
    if (cameras.length === 0) {
      setSelectedCamera(null);
      return;
    }

    setSelectedCamera(prev => {
      if (prev && cameras.some(camera => camera.name === prev)) {
        return prev;
      }
      return cameras[0]?.name ?? null;
    });
  }, [cameras]);

  const defaultLiveStreamUrl = useMemo(() => {
    if (!selectedCamera) return null;
    return `${normalizedApiBase}/stream/${encodeURIComponent(selectedCamera)}`;
  }, [normalizedApiBase, selectedCamera]);

  const liveStreamUrl = liveStreamOverride ?? defaultLiveStreamUrl;

  useEffect(() => {
    setLiveStreamOverride(null);
  }, [defaultLiveStreamUrl]);

  const handleLiveError = useCallback(() => {
    if (!selectedCamera || typeof window === 'undefined') return;
    const fallbackBase = window.location.origin.replace(/\/$/, '');
    const fallbackUrl = `${fallbackBase}/stream/${encodeURIComponent(selectedCamera)}`;
    setLiveStreamOverride(prev => (prev === fallbackUrl ? prev : fallbackUrl));
  }, [selectedCamera]);

  const handleSelectCamera = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    setSelectedCamera(event.target.value);
  }, []);

  useEffect(() => {
    if (!liveExpanded || typeof document === 'undefined') return;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, [liveExpanded]);

  useEffect(() => {
    if (!liveExpanded || typeof document === 'undefined') return;
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setLiveExpanded(false);
      }
    };
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('keydown', handleKey);
    };
  }, [liveExpanded]);

  if (!selectedCamera) {
    return (
      <p style={{ margin: 0 }}>
        Камеры не настроены. Добавьте RTSP/HTTP поток в <code>.env</code>.
      </p>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <h4 style={{ margin: '4px 0 0' }}>Живой просмотр</h4>
          <label style={{ fontSize: 14, color: '#334155', display: 'flex', gap: 8, alignItems: 'center' }}>
            Камера:
            <select
              value={selectedCamera ?? ''}
              onChange={handleSelectCamera}
              style={{
                padding: '4px 8px',
                borderRadius: 6,
                border: '1px solid #cbd5f5',
                backgroundColor: '#fff',
              }}
            >
              {cameras.map(camera => (
                <option key={camera.name} value={camera.name}>
                  {camera.name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <button
          onClick={() => setLiveExpanded(true)}
          style={{
            padding: '6px 12px',
            borderRadius: 6,
            border: '1px solid #2563eb',
            backgroundColor: '#2563eb',
            color: '#fff',
            cursor: 'pointer',
          }}
        >
          Увеличить
        </button>
      </div>
      {liveStreamUrl ? (
        <img
          src={liveStreamUrl}
          alt="live"
          style={{ width: '100%', borderRadius: 8, cursor: 'zoom-in' }}
          onClick={() => setLiveExpanded(true)}
          onError={handleLiveError}
        />
      ) : (
        <p style={{ margin: 0 }}>Поток недоступен.</p>
      )}
      <p style={{ fontSize: 12, color: '#64748b' }}>MJPEG-поток с наложенной разметкой (рамки, скелет, подписи).</p>

      {liveExpanded && liveStreamUrl && (
        <div
          onClick={() => setLiveExpanded(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15,23,42,0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: 16,
          }}
        >
          <div
            onClick={event => event.stopPropagation()}
            style={{
              position: 'relative',
              maxWidth: '90vw',
              maxHeight: '90vh',
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
            }}
          >
            <button
              onClick={() => setLiveExpanded(false)}
              style={{
                alignSelf: 'flex-end',
                padding: '6px 12px',
                borderRadius: 6,
                border: 'none',
                background: '#1e293b',
                color: '#fff',
                cursor: 'pointer',
              }}
            >
              Закрыть
            </button>
            <img
              src={liveStreamUrl}
              alt="live expanded"
              style={{
                maxWidth: '90vw',
                maxHeight: '80vh',
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                borderRadius: 12,
                boxShadow: '0 18px 36px rgba(15,23,42,0.45)',
              }}
              onError={handleLiveError}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default LiveStreamViewer;
