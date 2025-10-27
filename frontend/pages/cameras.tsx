import { FormEvent, useCallback, useState } from 'react';
import Layout from '../components/Layout';
import { useApiBase } from '../hooks/useApiBase';
import { useCameras } from '../hooks/useCameras';
import type { CameraStatus } from '../types/api';

const CamerasPage = () => {
  const { normalizedApiBase } = useApiBase();
  const { cameras, setCameras, loading, error: loadError, reload } = useCameras(normalizedApiBase);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [newCameraName, setNewCameraName] = useState('');
  const [newCameraUrl, setNewCameraUrl] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [formSuccess, setFormSuccess] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [removingCameraId, setRemovingCameraId] = useState<number | null>(null);

  const toggleForm = useCallback(() => {
    setIsFormOpen(prev => !prev);
    setFormError(null);
    setFormSuccess(null);
  }, []);

  const handleAddCamera = useCallback(async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const name = newCameraName.trim();
    const url = newCameraUrl.trim();

    if (!name || !url) {
      setFormError('Введите название и RTSP-адрес камеры.');
      setFormSuccess(null);
      return;
    }

    setIsSubmitting(true);
    setFormError(null);
    setFormSuccess(null);

    try {
      const response = await fetch(`${normalizedApiBase}/api/cameras/add`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name,
          rtsp_url: url,
        }),
      });

      if (!response.ok) {
        let detail = 'Не удалось добавить камеру.';
        try {
          const data = await response.json();
          if (data?.detail) {
            detail = data.detail;
          }
        } catch (jsonError) {
          const text = await response.text();
          if (text && text.trim()) {
            detail = text.trim();
          }
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const created = data?.camera;
      if (!created?.id || !created?.name) {
        throw new Error('Сервер вернул неожиданный ответ.');
      }

      setCameras(prev => {
        const next = [
          ...prev,
          {
            id: Number(created.id),
            name: String(created.name),
            status: 'starting' as CameraStatus,
            fps: null,
            lastFrameTs: null,
            uptimeSec: null,
          },
        ];
        return next.sort((a, b) => a.id - b.id);
      });
      setNewCameraName('');
      setNewCameraUrl('');
      setFormSuccess(`Камера «${created.name}» успешно добавлена.`);
      setIsFormOpen(false);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Произошла ошибка при добавлении камеры.');
    } finally {
      setIsSubmitting(false);
    }
  }, [newCameraName, newCameraUrl, normalizedApiBase, setCameras]);

  const handleDeleteCamera = useCallback(async (cameraId: number, cameraName: string) => {
    if (typeof window !== 'undefined') {
      const confirmed = window.confirm(`Удалить камеру «${cameraName}»?`);
      if (!confirmed) {
        return;
      }
    }

    setRemovingCameraId(cameraId);
    setFormError(null);
    setFormSuccess(null);

    try {
      const response = await fetch(`${normalizedApiBase}/api/cameras/${cameraId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        let detail = 'Не удалось удалить камеру.';
        try {
          const data = await response.json();
          if (data?.detail) {
            detail = data.detail;
          }
        } catch (jsonError) {
          const text = await response.text();
          if (text && text.trim()) {
            detail = text.trim();
          }
        }
        throw new Error(detail);
      }

      setCameras(prev => prev.filter(camera => camera.id !== cameraId));
      setFormSuccess(`Камера «${cameraName}» удалена.`);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Произошла ошибка при удалении камеры.');
    } finally {
      setRemovingCameraId(null);
    }
  }, [normalizedApiBase, setCameras]);

  const formatFps = (fps?: number | null) => {
    if (typeof fps !== 'number' || !Number.isFinite(fps)) {
      return '—';
    }
    if (fps >= 10) {
      return fps.toFixed(0);
    }
    return fps.toFixed(1);
  };

  const formatLastFrame = (timestamp?: string | null) => {
    if (!timestamp) {
      return '—';
    }
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) {
      return '—';
    }

    const now = Date.now();
    const diffSec = Math.max(0, Math.round((now - date.getTime()) / 1000));
    let relative: string;
    if (diffSec < 5) {
      relative = 'только что';
    } else if (diffSec < 60) {
      relative = `${diffSec} с назад`;
    } else if (diffSec < 3600) {
      relative = `${Math.floor(diffSec / 60)} мин назад`;
    } else if (diffSec < 86400) {
      relative = `${Math.floor(diffSec / 3600)} ч назад`;
    } else {
      relative = `${Math.floor(diffSec / 86400)} дн назад`;
    }

    return `${date.toLocaleString('ru-RU', { hour12: false })}\n${relative}`;
  };

  const formatUptime = (seconds?: number | null) => {
    if (typeof seconds !== 'number' || !Number.isFinite(seconds)) {
      return '—';
    }

    const total = Math.max(0, Math.floor(seconds));
    const days = Math.floor(total / 86400);
    const hours = Math.floor((total % 86400) / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const parts: string[] = [];
    if (days) parts.push(`${days} д`);
    if (hours) parts.push(`${hours} ч`);
    if (minutes) parts.push(`${minutes} мин`);
    if (!parts.length) {
      parts.push(`${total % 60} с`);
    }
    return parts.join(' ');
  };

  const statusLabels: Record<CameraStatus, string> = {
    online: 'В работе',
    offline: 'Отключена',
    starting: 'Запускается',
    stopping: 'Останавливается',
    no_signal: 'Нет сигнала',
    unknown: 'Неизвестно',
  };

  const statusPalette: Record<CameraStatus, { background: string; color: string; border: string }> = {
    online: { background: '#dcfce7', color: '#166534', border: '#86efac' },
    offline: { background: '#f1f5f9', color: '#475569', border: '#cbd5f5' },
    starting: { background: '#e0f2fe', color: '#0c4a6e', border: '#7dd3fc' },
    stopping: { background: '#fef9c3', color: '#854d0e', border: '#fde68a' },
    no_signal: { background: '#fee2e2', color: '#b91c1c', border: '#fecaca' },
    unknown: { background: '#f8fafc', color: '#475569', border: '#cbd5f5' },
  };

  const renderStatusBadge = (status?: CameraStatus) => {
    const key: CameraStatus = status ?? 'unknown';
    const palette = statusPalette[key] ?? statusPalette.unknown;
    const label = statusLabels[key] ?? statusLabels.unknown;
    return (
      <span
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '4px 8px',
          borderRadius: 999,
          border: `1px solid ${palette.border}`,
          background: palette.background,
          color: palette.color,
          fontWeight: 600,
          fontSize: 13,
          minWidth: 110,
          textAlign: 'center',
          whiteSpace: 'nowrap',
        }}
      >
        {label}
      </span>
    );
  };

  return (
    <Layout title="IP-CAM Analytics — Камеры">
      <h1>Управление камерами</h1>
      <p style={{ maxWidth: 640, color: '#475569' }}>
        Здесь можно подключить новые RTSP-потоки и управлять существующими. Камеры запускаются и останавливаются
        автоматически, когда появляются в списке.
      </p>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '24px 0', flexWrap: 'wrap', gap: 12 }}>
        <strong>Всего камер: {cameras.length}</strong>
        <button
          onClick={toggleForm}
          style={{
            padding: '10px 16px',
            borderRadius: 8,
            border: 'none',
            background: '#2563eb',
            color: '#fff',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          {isFormOpen ? 'Закрыть форму' : 'Добавить камеру'}
        </button>
      </div>

      {isFormOpen && (
        <form
          onSubmit={handleAddCamera}
          style={{
            display: 'grid',
            gap: 12,
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            alignItems: 'end',
            background: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: 12,
            padding: 16,
            boxShadow: '0 8px 16px rgba(15,23,42,0.08)',
            marginBottom: 24,
          }}
        >
          <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span style={{ fontSize: 14, color: '#0f172a', fontWeight: 600 }}>Название камеры</span>
            <input
              type="text"
              value={newCameraName}
              onChange={event => {
                setNewCameraName(event.target.value);
                setFormError(null);
                setFormSuccess(null);
              }}
              placeholder="Например, Въезд №1"
              style={{
                padding: '8px 10px',
                borderRadius: 6,
                border: '1px solid #cbd5f5',
                fontSize: 14,
              }}
            />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span style={{ fontSize: 14, color: '#0f172a', fontWeight: 600 }}>RTSP URL</span>
            <input
              type="text"
              value={newCameraUrl}
              onChange={event => {
                setNewCameraUrl(event.target.value);
                setFormError(null);
                setFormSuccess(null);
              }}
              placeholder="rtsp://user:pass@host/stream"
              style={{
                padding: '8px 10px',
                borderRadius: 6,
                border: '1px solid #cbd5f5',
                fontSize: 14,
              }}
            />
          </label>
          <button
            type="submit"
            disabled={isSubmitting}
            style={{
              padding: '10px 16px',
              borderRadius: 8,
              border: 'none',
              background: isSubmitting ? '#94a3b8' : '#0f766e',
              color: '#fff',
              cursor: isSubmitting ? 'wait' : 'pointer',
              fontWeight: 600,
              minHeight: 40,
            }}
          >
            {isSubmitting ? 'Добавляем…' : 'Сохранить'}
          </button>
        </form>
      )}

      {formError && (
        <div
          role="alert"
          style={{
            background: '#fee2e2',
            border: '1px solid #fecaca',
            color: '#b91c1c',
            padding: '8px 12px',
            borderRadius: 8,
            fontSize: 14,
            marginBottom: 16,
          }}
        >
          {formError}
        </div>
      )}

      {formSuccess && (
        <div
          style={{
            background: '#dcfce7',
            border: '1px solid #bbf7d0',
            color: '#166534',
            padding: '8px 12px',
            borderRadius: 8,
            fontSize: 14,
            marginBottom: 16,
          }}
        >
          {formSuccess}
        </div>
      )}

      <section
        style={{
          background: '#fff',
          border: '1px solid #e2e8f0',
          borderRadius: 12,
          padding: 16,
          boxShadow: '0 8px 16px rgba(15,23,42,0.08)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h2 style={{ margin: 0 }}>Активные камеры</h2>
          <button
            onClick={reload}
            style={{
              padding: '6px 12px',
              borderRadius: 6,
              border: '1px solid #cbd5f5',
              background: '#f8fafc',
              color: '#0f172a',
              cursor: 'pointer',
            }}
          >
            Обновить
          </button>
        </div>

        {loading && <p style={{ color: '#475569' }}>Загружаем список камер…</p>}
        {loadError && <p style={{ color: '#b91c1c' }}>{loadError}</p>}

        {cameras.length === 0 ? (
          <p style={{ margin: 0, color: '#475569', fontSize: 14 }}>
            Пока нет активных камер. Добавьте поток, чтобы начать обработку.
          </p>
        ) : (
          <div style={{ marginTop: 16, overflowX: 'auto' }}>
            <table
              style={{
                width: '100%',
                borderCollapse: 'separate',
                borderSpacing: 0,
                minWidth: 720,
              }}
            >
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th
                    style={{
                      padding: '12px 16px',
                      textAlign: 'left',
                      fontSize: 13,
                      fontWeight: 700,
                      color: '#0f172a',
                      borderBottom: '1px solid #e2e8f0',
                    }}
                  >
                    Камера
                  </th>
                  <th
                    style={{
                      padding: '12px 16px',
                      textAlign: 'center',
                      fontSize: 13,
                      fontWeight: 700,
                      color: '#0f172a',
                      borderBottom: '1px solid #e2e8f0',
                    }}
                  >
                    FPS
                  </th>
                  <th
                    style={{
                      padding: '12px 16px',
                      textAlign: 'left',
                      fontSize: 13,
                      fontWeight: 700,
                      color: '#0f172a',
                      borderBottom: '1px solid #e2e8f0',
                    }}
                  >
                    Последний кадр
                  </th>
                  <th
                    style={{
                      padding: '12px 16px',
                      textAlign: 'center',
                      fontSize: 13,
                      fontWeight: 700,
                      color: '#0f172a',
                      borderBottom: '1px solid #e2e8f0',
                    }}
                  >
                    Аптайм
                  </th>
                  <th
                    style={{
                      padding: '12px 16px',
                      textAlign: 'center',
                      fontSize: 13,
                      fontWeight: 700,
                      color: '#0f172a',
                      borderBottom: '1px solid #e2e8f0',
                    }}
                  >
                    Состояние
                  </th>
                  <th
                    style={{
                      padding: '12px 16px',
                      textAlign: 'right',
                      fontSize: 13,
                      fontWeight: 700,
                      color: '#0f172a',
                      borderBottom: '1px solid #e2e8f0',
                    }}
                  >
                    Действия
                  </th>
                </tr>
              </thead>
              <tbody>
                {cameras.map(camera => {
                  const isRemoving = removingCameraId === camera.id;
                  return (
                    <tr key={camera.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '14px 16px', verticalAlign: 'middle' }}>
                        <div style={{ fontWeight: 600, color: '#0f172a' }}>{camera.name}</div>
                        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>ID: {camera.id}</div>
                      </td>
                      <td style={{ padding: '14px 16px', textAlign: 'center', fontVariantNumeric: 'tabular-nums' }}>
                        {formatFps(camera.fps)}
                      </td>
                      <td style={{ padding: '14px 16px', whiteSpace: 'pre-line', fontSize: 13, color: '#334155' }}>
                        {formatLastFrame(camera.lastFrameTs)}
                      </td>
                      <td style={{ padding: '14px 16px', textAlign: 'center', fontVariantNumeric: 'tabular-nums' }}>
                        {formatUptime(camera.uptimeSec)}
                      </td>
                      <td style={{ padding: '14px 16px', textAlign: 'center' }}>{renderStatusBadge(camera.status)}</td>
                      <td style={{ padding: '14px 16px', textAlign: 'right' }}>
                        <button
                          type="button"
                          onClick={() => handleDeleteCamera(camera.id, camera.name)}
                          disabled={isRemoving}
                          style={{
                            padding: '6px 12px',
                            borderRadius: 6,
                            border: '1px solid #fca5a5',
                            background: isRemoving ? '#fecaca' : '#fee2e2',
                            color: '#b91c1c',
                            cursor: isRemoving ? 'wait' : 'pointer',
                            fontWeight: 600,
                            minWidth: 100,
                          }}
                        >
                          {isRemoving ? 'Удаляем…' : 'Удалить'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </Layout>
  );
};

export default CamerasPage;
