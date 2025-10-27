import { useEffect, useMemo, useState } from 'react';
import ActivityChart from '../components/ActivityChart';
import Layout from '../components/Layout';
import { useApiBase } from '../hooks/useApiBase';
import { EventItem, EventMeta } from '../types/api';

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

const formatPlate = (meta?: EventMeta) => {
  const plate = meta?.plate;
  if (!plate) return '—';
  return plate;
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

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch(`${normalizedApiBase}/events`);
        if (!response.ok) {
          throw new Error(`Статус ответа ${response.status}`);
        }
        const data = await response.json();
        const rawEvents = Array.isArray(data?.events) ? (data.events as EventItem[]) : [];
        setEvents(sortEventsDesc(rawEvents).slice(0, 200));
      } catch (err) {
        console.error('Не удалось получить события:', err);
      }
    };

    load();
  }, [normalizedApiBase]);

  useEffect(() => {
    const base = apiBase;
    if (!base) return;

    const wsUrl = new URL('/ws/events', `${base}/`);
    wsUrl.protocol = wsUrl.protocol === 'https:' ? 'wss:' : 'ws:';

    const ws = new WebSocket(wsUrl.toString());
    ws.onmessage = message => {
      try {
        const parsed = JSON.parse(message.data) as WsEventPayload;
        const nextEvent = mapWsEventPayload(parsed);
        if (!nextEvent) {
          return;
        }

        setEvents(prev => {
          const deduped = nextEvent.id != null
            ? prev.filter(event => event.id !== nextEvent.id)
            : prev.filter(event => event.start_ts !== nextEvent.start_ts);

          return sortEventsDesc([nextEvent, ...deduped]).slice(0, 200);
        });
      } catch (err) {
        console.error('Не удалось обработать событие по WebSocket:', err);
      }
    };

    return () => {
      ws.close();
    };
  }, [apiBase]);

  const rows = useMemo(
    () =>
      events.map((event, index) => {
        const snapshotAbsoluteUrl = buildAbsoluteUrl(event.snapshot_url);
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
            {events.length === 0 && (
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
