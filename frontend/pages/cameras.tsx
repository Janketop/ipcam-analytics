import { FormEvent, useCallback, useState } from 'react';
import Layout from '../components/Layout';
import { useApiBase } from '../hooks/useApiBase';
import { useCameras } from '../hooks/useCameras';
import type { CameraStatus } from '../types/api';

const DEFAULT_IDLE_ALERT_TIME = 300;
const MAX_IDLE_ALERT_TIME = 86400;

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
  const [detectPersonEnabled, setDetectPersonEnabled] = useState(true);
  const [detectCarEnabled, setDetectCarEnabled] = useState(true);
  const [captureEntryTimeEnabled, setCaptureEntryTimeEnabled] = useState(true);
  const [editingCameraId, setEditingCameraId] = useState<number | null>(null);
  const [idleAlertTime, setIdleAlertTime] = useState<string>(String(DEFAULT_IDLE_ALERT_TIME));

  const resetFormFields = useCallback(() => {
    setNewCameraName('');
    setNewCameraUrl('');
    setDetectPersonEnabled(true);
    setDetectCarEnabled(true);
    setCaptureEntryTimeEnabled(true);
    setEditingCameraId(null);
    setIdleAlertTime(String(DEFAULT_IDLE_ALERT_TIME));
  }, []);

  const toggleForm = useCallback(() => {
    setFormError(null);
    setFormSuccess(null);
    setIsFormOpen(prev => {
      const next = !prev;
      resetFormFields();
      return next;
    });
  }, [resetFormFields]);

  const handleSubmitCamera = useCallback(async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const name = newCameraName.trim();
    const url = newCameraUrl.trim();
    const idleSeconds = Number.parseInt(idleAlertTime.trim(), 10);

    if (!name || !url) {
      setFormError('Введите название и RTSP-адрес камеры.');
      setFormSuccess(null);
      return;
    }

    if (Number.isNaN(idleSeconds)) {
      setFormError('Укажите время простоя в секундах.');
      setFormSuccess(null);
      return;
    }

    if (idleSeconds < 10) {
      setFormError('Порог простоя не может быть меньше 10 секунд.');
      setFormSuccess(null);
      return;
    }

    if (idleSeconds > MAX_IDLE_ALERT_TIME) {
      setFormError('Порог простоя не должен превышать 24 часов (86400 секунд).');
      setFormSuccess(null);
      return;
    }

    setIsSubmitting(true);
    setFormError(null);
    setFormSuccess(null);

    try {
      const endpoint = editingCameraId
        ? `${normalizedApiBase}/api/cameras/${editingCameraId}`
        : `${normalizedApiBase}/api/cameras/add`;
      const method = editingCameraId ? 'PATCH' : 'POST';
      const response = await fetch(endpoint, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name,
          rtsp_url: url,
          detect_person: detectPersonEnabled,
          detect_car: detectCarEnabled,
          capture_entry_time: captureEntryTimeEnabled,
          idle_alert_time: idleSeconds,
        }),
      });

      if (!response.ok) {
        let detail = 'Не удалось сохранить камеру.';
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

      const updatedCameraId = Number(created.id);
      const rtspUrl =
        typeof created.rtspUrl === 'string'
          ? created.rtspUrl
          : typeof created.rtsp_url === 'string'
            ? (created.rtsp_url as string)
            : '';
      const detectPerson =
        typeof created.detectPerson === 'boolean'
          ? created.detectPerson
          : typeof created.detect_person === 'boolean'
            ? created.detect_person
            : true;
      const detectCar =
        typeof created.detectCar === 'boolean'
          ? created.detectCar
          : typeof created.detect_car === 'boolean'
            ? created.detect_car
            : true;
      const captureEntryTime =
        typeof created.captureEntryTime === 'boolean'
          ? created.captureEntryTime
          : typeof created.capture_entry_time === 'boolean'
            ? created.capture_entry_time
            : true;
      const idleAlert =
        typeof created.idleAlertTime === 'number'
          ? created.idleAlertTime
          : typeof created.idle_alert_time === 'number'
            ? created.idle_alert_time
            : idleSeconds;

      setCameras(prev => {
        if (editingCameraId != null) {
          return prev.map(camera =>
            camera.id === updatedCameraId
              ? {
                  ...camera,
                  name: String(created.name),
                  rtspUrl,
                  detectPerson,
                  detectCar,
                  captureEntryTime,
                  idleAlertTime: idleAlert,
                }
              : camera,
          );
        }

        const next = [
          ...prev,
          {
            id: updatedCameraId,
            name: String(created.name),
            rtspUrl,
            detectPerson,
            detectCar,
            captureEntryTime,
            idleAlertTime: idleAlert,
            status: 'starting' as CameraStatus,
            fps: null,
            lastFrameTs: null,
            uptimeSec: null,
          },
        ];
        return next.sort((a, b) => a.id - b.id);
      });

      const successMessage =
        editingCameraId != null
          ? `Настройки камеры «${created.name}» обновлены.`
          : `Камера «${created.name}» успешно добавлена.`;
      setFormSuccess(successMessage);
      resetFormFields();
      setIsFormOpen(false);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Произошла ошибка при сохранении камеры.');
    } finally {
      setIsSubmitting(false);
    }
  }, [
    newCameraName,
    newCameraUrl,
    idleAlertTime,
    detectPersonEnabled,
    detectCarEnabled,
    captureEntryTimeEnabled,
    normalizedApiBase,
    setCameras,
    editingCameraId,
    resetFormFields,
  ]);

  const handleEditCamera = useCallback(
    (cameraId: number) => {
      const target = cameras.find(camera => camera.id === cameraId);
      if (!target) {
        return;
      }

      setFormError(null);
      setFormSuccess(null);
      setEditingCameraId(target.id);
      setNewCameraName(target.name);
      setNewCameraUrl(target.rtspUrl);
      setDetectPersonEnabled(Boolean(target.detectPerson));
      setDetectCarEnabled(Boolean(target.detectCar));
      setCaptureEntryTimeEnabled(Boolean(target.captureEntryTime));
      setIdleAlertTime(String(target.idleAlertTime ?? DEFAULT_IDLE_ALERT_TIME));
      setIsFormOpen(true);
    },
    [cameras],
  );

  const handleCancelEdit = useCallback(() => {
    resetFormFields();
    setIsFormOpen(false);
    setFormError(null);
    setFormSuccess(null);
  }, [resetFormFields]);

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
      if (editingCameraId === cameraId) {
        resetFormFields();
        setIsFormOpen(false);
      }
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Произошла ошибка при удалении камеры.');
    } finally {
      setRemovingCameraId(null);
    }
  }, [normalizedApiBase, setCameras, editingCameraId, resetFormFields]);

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

  const renderFeatureChip = (label: string, enabled: boolean) => {
    const palette = enabled
      ? { background: '#dcfce7', color: '#166534', border: '#bbf7d0' }
      : { background: '#fee2e2', color: '#b91c1c', border: '#fecaca' };
    const valueLabel = enabled ? 'ВКЛ' : 'ВЫКЛ';
    return (
      <span
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          padding: '4px 8px',
          borderRadius: 999,
          border: `1px solid ${palette.border}`,
          background: palette.background,
          color: palette.color,
          fontWeight: 600,
          fontSize: 12,
          letterSpacing: 0.4,
        }}
      >
        {label}: {valueLabel}
      </span>
    );
  };

  const formatIdleThreshold = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds} с`;
    }
    if (seconds % 3600 === 0) {
      return `${Math.floor(seconds / 3600)} ч`;
    }
    if (seconds >= 3600) {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      const hoursPart = hours ? `${hours} ч` : '';
      const minutesPart = minutes ? ` ${minutes} мин` : '';
      return `${hoursPart}${minutesPart}`.trim();
    }
    if (seconds % 60 === 0) {
      return `${Math.floor(seconds / 60)} мин`;
    }
    return `${(seconds / 60).toFixed(1)} мин`;
  };

  const renderIdleThresholdChip = (seconds?: number) => {
    if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds <= 0) {
      return null;
    }
    return (
      <span
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          padding: '4px 8px',
          borderRadius: 999,
          border: '1px solid #ddd6fe',
          background: '#ede9fe',
          color: '#5b21b6',
          fontWeight: 600,
          fontSize: 12,
          letterSpacing: 0.4,
        }}
      >
        Порог простоя: {formatIdleThreshold(seconds)}
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
          {isFormOpen
            ? editingCameraId != null
              ? 'Скрыть редактирование'
              : 'Закрыть форму'
            : 'Добавить камеру'}
        </button>
      </div>

      {isFormOpen && (
        <form
          onSubmit={handleSubmitCamera}
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
          {editingCameraId != null && (
            <div
              style={{
                gridColumn: '1 / -1',
                background: '#f1f5f9',
                borderRadius: 8,
                padding: '8px 12px',
                fontSize: 13,
                color: '#0f172a',
                fontWeight: 600,
              }}
            >
              Редактирование камеры №{editingCameraId}
            </div>
          )}
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
          <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span style={{ fontSize: 14, color: '#0f172a', fontWeight: 600 }}>Порог простоя, сек</span>
            <input
              type="number"
              min={10}
              max={MAX_IDLE_ALERT_TIME}
              value={idleAlertTime}
              onChange={event => {
                setIdleAlertTime(event.target.value.replace(/[^0-9]/g, ''));
                setFormError(null);
                setFormSuccess(null);
              }}
              placeholder={`Например, ${DEFAULT_IDLE_ALERT_TIME}`}
              style={{
                padding: '8px 10px',
                borderRadius: 6,
                border: '1px solid #cbd5f5',
                fontSize: 14,
              }}
            />
          </label>
          <div
            style={{
              gridColumn: '1 / -1',
              display: 'flex',
              flexWrap: 'wrap',
              gap: 12,
              alignItems: 'center',
            }}
          >
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, color: '#0f172a' }}>
              <input
                type="checkbox"
                checked={detectPersonEnabled}
                onChange={event => {
                  setDetectPersonEnabled(event.target.checked);
                  setFormError(null);
                  setFormSuccess(null);
                }}
              />
              Детектировать использование телефонов
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, color: '#0f172a' }}>
              <input
                type="checkbox"
                checked={detectCarEnabled}
                onChange={event => {
                  setDetectCarEnabled(event.target.checked);
                  setFormError(null);
                  setFormSuccess(null);
                }}
              />
              Отслеживать въезд автомобилей
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, color: '#0f172a' }}>
              <input
                type="checkbox"
                checked={captureEntryTimeEnabled}
                onChange={event => {
                  setCaptureEntryTimeEnabled(event.target.checked);
                  setFormError(null);
                  setFormSuccess(null);
                }}
              />
              Сохранять отметку времени въезда
            </label>
          </div>
          <div
            style={{
              gridColumn: '1 / -1',
              display: 'flex',
              gap: 12,
              flexWrap: 'wrap',
            }}
          >
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
              {isSubmitting
                ? editingCameraId != null
                  ? 'Сохраняем…'
                  : 'Добавляем…'
                : editingCameraId != null
                  ? 'Сохранить изменения'
                  : 'Сохранить'}
            </button>
            {editingCameraId != null && (
              <button
                type="button"
                onClick={handleCancelEdit}
                style={{
                  padding: '10px 16px',
                  borderRadius: 8,
                  border: '1px solid #cbd5f5',
                  background: '#f8fafc',
                  color: '#0f172a',
                  cursor: 'pointer',
                  fontWeight: 600,
                  minHeight: 40,
                }}
              >
                Отмена
              </button>
            )}
          </div>
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
                      textAlign: 'left',
                      fontSize: 13,
                      fontWeight: 700,
                      color: '#0f172a',
                      borderBottom: '1px solid #e2e8f0',
                    }}
                  >
                    Функции
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
                  const isEditingThis = editingCameraId === camera.id;
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
                      <td style={{ padding: '14px 16px' }}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                          {renderFeatureChip('Телефоны', Boolean(camera.detectPerson))}
                          {renderFeatureChip('Авто', Boolean(camera.detectCar))}
                          {renderFeatureChip('Время въезда', Boolean(camera.captureEntryTime))}
                          {renderIdleThresholdChip(camera.idleAlertTime)}
                        </div>
                      </td>
                      <td style={{ padding: '14px 16px', textAlign: 'center' }}>{renderStatusBadge(camera.status)}</td>
                      <td style={{ padding: '14px 16px', textAlign: 'right' }}>
                        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, flexWrap: 'wrap' }}>
                          <button
                            type="button"
                            onClick={() => handleEditCamera(camera.id)}
                            disabled={isRemoving}
                            style={{
                              padding: '6px 12px',
                              borderRadius: 6,
                              border: '1px solid #2563eb',
                              background: isEditingThis ? '#2563eb' : '#e0f2fe',
                              color: isEditingThis ? '#fff' : '#0f172a',
                              cursor: isRemoving ? 'wait' : 'pointer',
                              fontWeight: 600,
                              minWidth: 120,
                            }}
                          >
                            {isEditingThis ? 'Редактируем' : 'Настроить'}
                          </button>
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
                              minWidth: 110,
                            }}
                          >
                            {isRemoving ? 'Удаляем…' : 'Удалить'}
                          </button>
                        </div>
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
