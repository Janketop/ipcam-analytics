import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ActivityChart from '../components/ActivityChart';
import Layout from '../components/Layout';
import { useApiBase } from '../hooks/useApiBase';
import type { Camera, EventItem, EventMeta } from '../types/api';

type WsEventPayload = {
  id?: number;
  type?: string;
  start_ts?: string;
  ts?: string;
  confidence?: number;
  snapshot_url?: string;
  camera?: string;
  meta?: EventMeta;
};

const mapWsEventPayload = (payload: WsEventPayload | null): EventItem | null => {
  if (!payload) return null;
  const startTs = payload.start_ts || payload.ts;
  if (!startTs) return null;

  return {
    id: typeof payload.id === 'number' ? payload.id : undefined,
    type: payload.type || 'UNKNOWN_EVENT',
    start_ts: startTs,
    confidence: typeof payload.confidence === 'number' ? payload.confidence : undefined,
    snapshot_url: payload.snapshot_url,
    camera: payload.camera,
    meta: payload.meta,
  };
};

const MAX_EVENTS = 200;

const convertLocalInputToIso = (value: string): string | null => {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  return parsed.toISOString();
};

const parseIsoTimestamp = (iso: string | null): number | undefined => {
  if (!iso) return undefined;
  const timestamp = Date.parse(iso);
  return Number.isNaN(timestamp) ? undefined : timestamp;
};

const formatActiveDateTime = (iso: string | null): string | null => {
  if (!iso) return null;
  const timestamp = Date.parse(iso);
  if (Number.isNaN(timestamp)) {
    return null;
  }
  return new Date(timestamp).toLocaleString();
};

const formatPlate = (meta?: EventMeta) => {
  const plate = meta?.plate;
  if (!plate) return '—';
  return plate;
};

const formatEmployeeName = (meta?: EventMeta) => {
  const name = meta?.employeeName;
  if (typeof name !== 'string') return null;
  const trimmed = name.trim();
  return trimmed.length > 0 ? trimmed : null;
};

const formatEntryTime = (meta?: EventMeta, fallback?: string) => {
  const ts = meta?.entry_ts || fallback;
  if (!ts) return '—';
  return new Date(ts).toLocaleString();
};

const normalizeNumber = (value: unknown): number | undefined => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
};

const getEventTimestamp = (event: EventItem): number => {
  const ts = Date.parse(event.start_ts);
  return Number.isNaN(ts) ? 0 : ts;
};

const sortEventsDesc = (items: EventItem[]): EventItem[] =>
  [...items].sort((a, b) => getEventTimestamp(b) - getEventTimestamp(a));

const formatPoseConfidence = (meta?: EventMeta) => {
  const raw =
    normalizeNumber(meta?.pose_confidence) ?? normalizeNumber((meta as Record<string, unknown> | undefined)?.poseConfidence);
  if (raw === undefined) return '—';
  const clamped = Math.min(Math.max(raw, 0), 1);
  const percent = Math.round(clamped * 100);
  return `${percent}%`;
};

const formatHeadAngle = (meta?: EventMeta) => {
  const raw =
    normalizeNumber(meta?.head_angle) ?? normalizeNumber((meta as Record<string, unknown> | undefined)?.headAngle);
  if (raw === undefined) return '—';
  const rounded = Math.round(raw * 10) / 10;
  const sign = rounded > 0 ? '+' : '';
  return `${sign}${rounded.toFixed(1)}°`;
};

const formatHandsMotion = (meta?: EventMeta) => {
  const raw =
    normalizeNumber(meta?.hands_motion) ?? normalizeNumber((meta as Record<string, unknown> | undefined)?.handMovement);
  if (raw === undefined) return '—';
  if (raw === 0) {
    return '0.000';
  }
  if (Math.abs(raw) >= 1) {
    return raw.toFixed(2);
  }
  return raw.toFixed(3);
};

const formatIdleDuration = (meta?: EventMeta) => {
  const raw =
    normalizeNumber(meta?.duration_idle_sec) ?? normalizeNumber((meta as Record<string, unknown> | undefined)?.idleSeconds);
  if (raw === undefined) return '—';
  if (raw < 1) return '<1 с';
  const totalSeconds = Math.floor(raw);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours} ч ${minutes.toString().padStart(2, '0')} мин`;
  }
  if (minutes > 0) {
    return `${minutes} мин ${seconds.toString().padStart(2, '0')} с`;
  }
  return `${seconds} с`;
};

const EventsPage = () => {
  const { apiBase, normalizedApiBase, buildAbsoluteUrl } = useApiBase();
  const [events, setEvents] = useState<EventItem[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [knownTypes, setKnownTypes] = useState<string[]>([]);
  const [filterState, setFilterState] = useState({ camera: '', type: '', from: '', to: '' });
  const [isLoading, setIsLoading] = useState(false);
  const [filterError, setFilterError] = useState<string | null>(null);

  const normalizedFilters = useMemo(() => {
    const fromIso = convertLocalInputToIso(filterState.from);
    const toIso = convertLocalInputToIso(filterState.to);
    return {
      camera: filterState.camera || undefined,
      type: filterState.type || undefined,
      fromIso,
      toIso,
      fromTimestamp: parseIsoTimestamp(fromIso),
      toTimestamp: parseIsoTimestamp(toIso),
    };
  }, [filterState]);

  const filtersSnapshotRef = useRef<{
    camera?: string;
    type?: string;
    fromTimestamp?: number;
    toTimestamp?: number;
  }>({});

  useEffect(() => {
    filtersSnapshotRef.current = {
      camera: normalizedFilters.camera,
      type: normalizedFilters.type,
      fromTimestamp: normalizedFilters.fromTimestamp,
      toTimestamp: normalizedFilters.toTimestamp,
    };
  }, [normalizedFilters.camera, normalizedFilters.type, normalizedFilters.fromTimestamp, normalizedFilters.toTimestamp]);

  const updateKnownTypes = useCallback((items: EventItem[]) => {
    setKnownTypes(prev => {
      const next = new Set(prev);
      items.forEach(item => {
        if (item.type) {
          next.add(item.type);
        }
      });
      return Array.from(next).sort((a, b) => a.localeCompare(b, 'ru-RU'));
    });
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    const loadCameras = async () => {
      try {
        const response = await fetch(`${normalizedApiBase}/cameras`, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Статус ответа ${response.status}`);
        }
        const data = await response.json();
        const items = Array.isArray(data?.cameras) ? (data.cameras as Camera[]) : [];
        setCameras(items);
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') {
          return;
        }
        console.error('Не удалось получить список камер:', err);
      }
    };

    loadCameras();

    return () => {
      controller.abort();
    };
  }, [normalizedApiBase]);

  useEffect(() => {
    const controller = new AbortController();

    const loadEvents = async () => {
      if (
        normalizedFilters.fromTimestamp !== undefined &&
        normalizedFilters.toTimestamp !== undefined &&
        normalizedFilters.fromTimestamp > normalizedFilters.toTimestamp
      ) {
        setFilterError('Дата начала не может быть позже даты окончания.');
        setIsLoading(false);
        return;
      }

      setFilterError(null);
      setIsLoading(true);

      try {
        const params = new URLSearchParams({ limit: String(MAX_EVENTS) });
        if (normalizedFilters.type) params.set('type', normalizedFilters.type);
        if (normalizedFilters.camera) params.set('camera', normalizedFilters.camera);
        if (normalizedFilters.fromIso) params.set('from_ts', normalizedFilters.fromIso);
        if (normalizedFilters.toIso) params.set('to_ts', normalizedFilters.toIso);

        const response = await fetch(`${normalizedApiBase}/events?${params.toString()}`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          let errorDetail = `Статус ответа ${response.status}`;
          try {
            const errorPayload = await response.json();
            if (typeof errorPayload?.detail === 'string') {
              errorDetail = errorPayload.detail;
            }
          } catch {
            // игнорируем ошибку парсинга
          }
          throw new Error(errorDetail);
        }
        const data = await response.json();
        const rawEvents = Array.isArray(data?.events) ? (data.events as EventItem[]) : [];
        updateKnownTypes(rawEvents);
        setEvents(sortEventsDesc(rawEvents).slice(0, MAX_EVENTS));
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') {
          return;
        }
        console.error('Не удалось получить события:', err);
        setFilterError((err as Error)?.message || 'Не удалось получить события.');
      } finally {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      }
    };

    loadEvents();

    return () => {
      controller.abort();
    };
  }, [
    normalizedApiBase,
    normalizedFilters.camera,
    normalizedFilters.type,
    normalizedFilters.fromIso,
    normalizedFilters.toIso,
    normalizedFilters.fromTimestamp,
    normalizedFilters.toTimestamp,
    updateKnownTypes,
  ]);

  const matchesCurrentFilters = useCallback((event: EventItem) => {
    const snapshot = filtersSnapshotRef.current;
    if (!snapshot) return true;

    if (snapshot.camera && event.camera !== snapshot.camera) {
      return false;
    }
    if (snapshot.type && event.type !== snapshot.type) {
      return false;
    }

    const eventTimestamp = Date.parse(event.start_ts);
    if (!Number.isNaN(eventTimestamp)) {
      if (snapshot.fromTimestamp !== undefined && eventTimestamp < snapshot.fromTimestamp) {
        return false;
      }
      if (snapshot.toTimestamp !== undefined && eventTimestamp > snapshot.toTimestamp) {
        return false;
      }
    }

    return true;
  }, []);

  const wsQuery = useMemo(() => {
    const params = new URLSearchParams();
    if (normalizedFilters.type) params.set('type', normalizedFilters.type);
    if (normalizedFilters.camera) params.set('camera', normalizedFilters.camera);
    if (normalizedFilters.fromIso) params.set('from_ts', normalizedFilters.fromIso);
    if (normalizedFilters.toIso) params.set('to_ts', normalizedFilters.toIso);
    return params.toString();
  }, [normalizedFilters.camera, normalizedFilters.type, normalizedFilters.fromIso, normalizedFilters.toIso]);

  useEffect(() => {
    const base = apiBase;
    if (!base) return;

    const wsUrl = new URL('/ws/events', `${base}/`);
    if (wsQuery) {
      wsUrl.search = wsQuery;
    }
    wsUrl.protocol = wsUrl.protocol === 'https:' ? 'wss:' : 'ws:';

    const ws = new WebSocket(wsUrl.toString());
    ws.onmessage = message => {
      try {
        const parsed = JSON.parse(message.data) as WsEventPayload;
        const nextEvent = mapWsEventPayload(parsed);
        if (!nextEvent) {
          return;
        }

        updateKnownTypes([nextEvent]);
        if (!matchesCurrentFilters(nextEvent)) {
          return;
        }

        setEvents(prev => {
          const deduped = nextEvent.id != null
            ? prev.filter(event => event.id !== nextEvent.id)
            : prev.filter(event => event.start_ts !== nextEvent.start_ts);
          const merged = [nextEvent, ...deduped].filter(matchesCurrentFilters);
          return sortEventsDesc(merged).slice(0, MAX_EVENTS);
        });
      } catch (err) {
        console.error('Не удалось обработать событие по WebSocket:', err);
      }
    };

    return () => {
      ws.close();
    };
  }, [apiBase, wsQuery, matchesCurrentFilters, updateKnownTypes]);

  const cameraOptions = useMemo(() => {
    const names = new Set<string>();
    cameras.forEach(camera => {
      if (camera?.name) {
        names.add(camera.name);
      }
    });
    if (filterState.camera) {
      names.add(filterState.camera);
    }
    return Array.from(names).sort((a, b) => a.localeCompare(b, 'ru-RU'));
  }, [cameras, filterState.camera]);

  const typeOptions = useMemo(() => {
    const types = new Set<string>(knownTypes);
    if (filterState.type) {
      types.add(filterState.type);
    }
    return Array.from(types).sort((a, b) => a.localeCompare(b, 'ru-RU'));
  }, [knownTypes, filterState.type]);

  const activeFilters = useMemo(() => {
    const result: string[] = [];
    if (normalizedFilters.camera) {
      result.push(`Камера: ${normalizedFilters.camera}`);
    }
    if (normalizedFilters.type) {
      result.push(`Тип: ${normalizedFilters.type}`);
    }
    const fromDisplay = formatActiveDateTime(normalizedFilters.fromIso);
    if (fromDisplay) {
      result.push(`С: ${fromDisplay}`);
    }
    const toDisplay = formatActiveDateTime(normalizedFilters.toIso);
    if (toDisplay) {
      result.push(`По: ${toDisplay}`);
    }
    return result;
  }, [normalizedFilters.camera, normalizedFilters.type, normalizedFilters.fromIso, normalizedFilters.toIso]);

  const rows = useMemo(
    () =>
      events.map((event, index) => {
        const snapshotAbsoluteUrl = buildAbsoluteUrl(event.snapshot_url);
        const employeeName = formatEmployeeName(event.meta);
        return (
          <tr key={`${event.start_ts}-${index}`} style={{ background: index % 2 === 0 ? '#fff' : '#f8fafc' }}>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>
              {snapshotAbsoluteUrl ? (
                <img src={snapshotAbsoluteUrl} alt="snapshot" width={160} style={{ borderRadius: 8, display: 'block' }} />
              ) : (
                <span style={{ color: '#94a3b8' }}>—</span>
              )}
            </td>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0', minWidth: 160 }}>
              <div style={{ fontWeight: 600 }}>{event.type}</div>
              {employeeName ? (
                <div style={{ color: '#0f172a', fontSize: 13 }}>Сотрудник: {employeeName}</div>
              ) : null}
              <div style={{ color: '#475569', fontSize: 13 }}>{event.camera ? `Камера: ${event.camera}` : '—'}</div>
            </td>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>{new Date(event.start_ts).toLocaleString()}</td>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace', fontSize: 16 }}>
              {formatPlate(event.meta)}
            </td>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>{formatEntryTime(event.meta, event.start_ts)}</td>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>
              {event.confidence !== undefined ? event.confidence.toFixed(2) : '—'}
            </td>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>{formatPoseConfidence(event.meta)}</td>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>{formatHeadAngle(event.meta)}</td>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>{formatHandsMotion(event.meta)}</td>
            <td style={{ padding: 10, borderBottom: '1px solid #e2e8f0' }}>{formatIdleDuration(event.meta)}</td>
          </tr>
        );
      }),
    [events, buildAbsoluteUrl]
  );

  const resetFilters = () => {
    setFilterState({ camera: '', type: '', from: '', to: '' });
    setFilterError(null);
  };

  return (
    <Layout title="IP-CAM Analytics — События">
      <h1>Лента событий</h1>
      <p style={{ maxWidth: 640, color: '#475569' }}>
        Здесь отображаются последние детекции с камер видеонаблюдения. Новые события прилетают в режиме реального времени.
      </p>

      <section style={{ marginBottom: 24 }}>
        <div
          style={{
            background: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: 12,
            padding: 16,
            boxShadow: '0 8px 16px rgba(15,23,42,0.08)',
          }}
        >
          <h2 style={{ marginTop: 0 }}>Фильтры событий</h2>
          <p style={{ marginTop: 0, maxWidth: 560, color: '#475569' }}>
            Выберите интересующие параметры, чтобы сузить список событий и обновить график активности.
          </p>
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 16,
              alignItems: 'flex-end',
              marginBottom: 12,
            }}
          >
            <label style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 220 }}>
              <span style={{ fontSize: 13, color: '#475569' }}>Камера</span>
              <select
                value={filterState.camera}
                onChange={event => setFilterState(prev => ({ ...prev, camera: event.target.value }))}
                style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #cbd5f5', fontSize: 14 }}
              >
                <option value="">Все камеры</option>
                {cameraOptions.map(name => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>

            <label style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 220 }}>
              <span style={{ fontSize: 13, color: '#475569' }}>Тип события</span>
              <select
                value={filterState.type}
                onChange={event => setFilterState(prev => ({ ...prev, type: event.target.value }))}
                style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #cbd5f5', fontSize: 14 }}
              >
                <option value="">Все типы</option>
                {typeOptions.map(type => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>

            <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span style={{ fontSize: 13, color: '#475569' }}>С (дата и время)</span>
              <input
                type="datetime-local"
                value={filterState.from}
                onChange={event => setFilterState(prev => ({ ...prev, from: event.target.value }))}
                style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #cbd5f5', fontSize: 14 }}
              />
            </label>

            <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span style={{ fontSize: 13, color: '#475569' }}>По (дата и время)</span>
              <input
                type="datetime-local"
                value={filterState.to}
                onChange={event => setFilterState(prev => ({ ...prev, to: event.target.value }))}
                style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #cbd5f5', fontSize: 14 }}
              />
            </label>

            <button
              type="button"
              onClick={resetFilters}
              style={{
                padding: '10px 16px',
                borderRadius: 8,
                border: '1px solid #e2e8f0',
                background: '#f1f5f9',
                color: '#1e293b',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Сбросить фильтры
            </button>
          </div>

          {filterError && (
            <div style={{ color: '#dc2626', marginBottom: 12, fontSize: 14 }}>{filterError}</div>
          )}

          {activeFilters.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 4 }}>
              {activeFilters.map(label => (
                <span
                  key={label}
                  style={{
                    background: '#e0f2fe',
                    color: '#0369a1',
                    borderRadius: 999,
                    padding: '4px 10px',
                    fontSize: 13,
                    fontWeight: 600,
                  }}
                >
                  {label}
                </span>
              ))}
            </div>
          )}

          <div style={{ fontSize: 13, color: '#64748b' }}>
            Показано {events.length} из {MAX_EVENTS} последних событий.
          </div>
        </div>
      </section>

      <section style={{ marginBottom: 24 }}>
        <div
          style={{
            background: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: 12,
            padding: 16,
            boxShadow: '0 8px 16px rgba(15,23,42,0.08)',
          }}
        >
          <h2 style={{ marginTop: 0 }}>Активность за последние 24 часа</h2>
          <p style={{ marginTop: 0, maxWidth: 560, color: '#475569' }}>
            График показывает распределение количества зафиксированных событий по часам. На нём легко заметить пики активности
            и тихие периоды за последние сутки.
          </p>
          <ActivityChart events={events} />
        </div>
      </section>

      <div style={{ overflowX: 'auto', background: '#fff', borderRadius: 12, border: '1px solid #e2e8f0', boxShadow: '0 8px 16px rgba(15,23,42,0.08)' }}>
        <table style={{ width: '100%', minWidth: 720, borderCollapse: 'collapse' }}>
          <thead style={{ background: '#f8fafc' }}>
            <tr>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Превью</th>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Событие</th>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Время фиксации</th>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Госномер</th>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Время заезда</th>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Уверенность</th>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Поза</th>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Наклон головы</th>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Активность рук</th>
              <th style={{ padding: 10, textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Простой</th>
            </tr>
          </thead>
          <tbody>
            {rows}
            {isLoading && events.length === 0 && (
              <tr>
                <td colSpan={10} style={{ padding: 16, textAlign: 'center', color: '#64748b' }}>
                  Загружаем события...
                </td>
              </tr>
            )}
            {!isLoading && events.length === 0 && (
              <tr>
                <td colSpan={10} style={{ padding: 16, textAlign: 'center', color: '#64748b' }}>
                  Пока нет зафиксированных событий.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Layout>
  );
};

export default EventsPage;
