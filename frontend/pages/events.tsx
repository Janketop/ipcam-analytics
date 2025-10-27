import { useEffect, useMemo, useState } from 'react';
import Layout from '../components/Layout';
import { useApiBase } from '../hooks/useApiBase';
import { EventItem, EventMeta } from '../types/api';

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

const EventsPage = () => {
  const { apiBase, normalizedApiBase, buildAbsoluteUrl } = useApiBase();
  const [events, setEvents] = useState<EventItem[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch(`${normalizedApiBase}/events`);
        const data = await response.json();
        setEvents(Array.isArray(data?.events) ? data.events : []);
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
      const data = JSON.parse(message.data);
      setEvents(prev => [
        {
          type: data.type,
          start_ts: data.ts,
          confidence: data.confidence,
          snapshot_url: data.snapshot_url,
          camera: data.camera,
          meta: data.meta,
        },
        ...prev,
      ].slice(0, 200));
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
            </tr>
          </thead>
          <tbody>
            {rows}
            {events.length === 0 && (
              <tr>
                <td colSpan={6} style={{ padding: 16, textAlign: 'center', color: '#64748b' }}>
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
