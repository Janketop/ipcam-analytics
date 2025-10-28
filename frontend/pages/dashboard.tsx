import { useCallback, useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import CleanupStatus from '../components/CleanupStatus';
import Layout from '../components/Layout';
import LiveStreamViewer from '../components/LiveStreamViewer';
import RuntimeStatus from '../components/RuntimeStatus';
import SimulatedVisualization from '../components/SimulatedVisualization';
import StatsChart from '../components/StatsChart';
import { useApiBase } from '../hooks/useApiBase';
import { useCameras } from '../hooks/useCameras';
import { useCleanupInfo } from '../hooks/useCleanupInfo';
import { useRuntimeInfo } from '../hooks/useRuntimeInfo';
import { Stat } from '../types/api';

type SimEvent = {
  id: number;
  label: string;
  probability: number;
  camera: string;
  ts: number;
};

const DashboardPage = () => {
  const { normalizedApiBase } = useApiBase();
  const { cameras } = useCameras(normalizedApiBase);
  const { runtime, error: runtimeError } = useRuntimeInfo(normalizedApiBase);
  const {
    cleanup,
    settings: cleanupSettings,
    loading: cleanupLoading,
    error: cleanupError,
    refresh: refreshCleanup,
    runCleanup,
    runInProgress: cleanupInProgress,
    runError: cleanupRunError,
    runSuccess: cleanupRunSuccess,
    clearSnapshots,
    clearSnapshotsInProgress,
    clearSnapshotsError,
    clearSnapshotsSuccess,
    clearEvents,
    clearEventsInProgress,
    clearEventsError,
    clearEventsSuccess,
  } = useCleanupInfo(normalizedApiBase);
  const [stats, setStats] = useState<Stat[]>([]);
  const [viewMode, setViewMode] = useState<'live' | 'sim'>('live');
  const [simEvents, setSimEvents] = useState<SimEvent[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainSuccess, setTrainSuccess] = useState<string | null>(null);

  useEffect(() => {
    const loadStats = async () => {
      try {
        const response = await fetch(`${normalizedApiBase}/stats`);
        const data = await response.json();
        setStats(Array.isArray(data?.stats) ? data.stats : []);
      } catch (err) {
        console.error('Не удалось получить статистику:', err);
      }
    };

    loadStats();
  }, [normalizedApiBase]);

  const handleSimEvent = useCallback((event: SimEvent) => {
    setSimEvents(prev => [event, ...prev].slice(0, 15));
  }, []);

  const handleSelfTraining = useCallback(async () => {
    setTrainError(null);
    setTrainSuccess(null);
    setIsTraining(true);
    try {
      const response = await fetch(`${normalizedApiBase}/train/self`, { method: 'POST' });
      if (!response.ok) {
        let detail = response.statusText;
        try {
          const data = await response.json();
          if (data?.detail) {
            detail = data.detail;
          }
        } catch (err) {
          console.warn('Не удалось разобрать ответ API при запуске обучения', err);
        }
        throw new Error(detail || 'Не удалось запустить самообучение');
      }
      setTrainSuccess('Обучение модели запущено. Это может занять несколько минут.');
    } catch (err) {
      console.error('Ошибка запуска самообучения модели', err);
      setTrainError(err instanceof Error ? err.message : 'Не удалось запустить самообучение');
    } finally {
      setIsTraining(false);
    }
  }, [normalizedApiBase]);

  useEffect(() => {
    if (viewMode !== 'sim') {
      setSimEvents([]);
    }
  }, [viewMode]);

  const viewToggleStyle = useMemo(
    () => ({ display: 'flex', gap: 8, flexWrap: 'wrap' as const }),
    []
  );

  const spinnerStyle: CSSProperties = useMemo(
    () => ({
      width: 16,
      height: 16,
      borderRadius: '50%',
      border: '2px solid rgba(148, 163, 184, 0.5)',
      borderTopColor: '#2563eb',
      animation: 'dashboard-spin 0.8s linear infinite',
      marginRight: 8,
    }),
    []
  );

  return (
    <Layout title="IP-CAM Analytics — Дашборд">
      <h1>Общий дашборд</h1>
      <p style={{ maxWidth: 680, color: '#475569' }}>
        Здесь собраны ключевые метрики и визуализации: статистика по событиям, состояние вычислительных ресурсов и живой
        поток с камер или их симуляция.
      </p>

      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16, marginBottom: 24 }}>
        <div style={{ border: '1px solid #e2e8f0', borderRadius: 12, padding: 16, background: '#fff', boxShadow: '0 8px 16px rgba(15,23,42,0.08)' }}>
          <h2 style={{ marginTop: 0 }}>Статистика (последние 24 часа)</h2>
          <StatsChart stats={stats} />
        </div>
        <div style={{ border: '1px solid #e2e8f0', borderRadius: 12, padding: 16, background: '#fff', boxShadow: '0 8px 16px rgba(15,23,42,0.08)', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <h2 style={{ margin: 0 }}>Визуализация</h2>
            <div style={viewToggleStyle}>
              <button
                onClick={() => setViewMode('live')}
                style={{
                  padding: '6px 12px',
                  borderRadius: 6,
                  border: '1px solid',
                  borderColor: viewMode === 'live' ? '#2563eb' : '#cbd5f5',
                  backgroundColor: viewMode === 'live' ? '#2563eb' : '#f8fafc',
                  color: viewMode === 'live' ? '#fff' : '#0f172a',
                  cursor: 'pointer',
                }}
              >
                Реальный поток
              </button>
              <button
                onClick={() => setViewMode('sim')}
                style={{
                  padding: '6px 12px',
                  borderRadius: 6,
                  border: '1px solid',
                  borderColor: viewMode === 'sim' ? '#2563eb' : '#cbd5f5',
                  backgroundColor: viewMode === 'sim' ? '#2563eb' : '#f8fafc',
                  color: viewMode === 'sim' ? '#fff' : '#0f172a',
                  cursor: 'pointer',
                }}
              >
                Симулятор
              </button>
            </div>
          </div>

          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, display: 'flex', flexDirection: 'column', gap: 8, position: 'relative' }}>
            <style>
              {`
                @keyframes dashboard-spin {
                  from { transform: rotate(0deg); }
                  to { transform: rotate(360deg); }
                }
              `}
            </style>
            <h3 style={{ margin: '0 0 4px' }}>Статус вычислений</h3>
            <RuntimeStatus runtime={runtime} error={runtimeError} />
            <CleanupStatus
              cleanup={cleanup}
              settings={cleanupSettings}
              loading={cleanupLoading}
              error={cleanupError}
              onRefresh={refreshCleanup}
              onRun={runCleanup}
              runInProgress={cleanupInProgress}
              runError={cleanupRunError}
              runSuccess={cleanupRunSuccess}
              onClearSnapshots={clearSnapshots}
              clearSnapshotsInProgress={clearSnapshotsInProgress}
              clearSnapshotsError={clearSnapshotsError}
              clearSnapshotsSuccess={clearSnapshotsSuccess}
              onClearEvents={clearEvents}
              clearEventsInProgress={clearEventsInProgress}
              clearEventsError={clearEventsError}
              clearEventsSuccess={clearEventsSuccess}
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <button
                onClick={handleSelfTraining}
                disabled={isTraining}
                style={{
                  padding: '8px 12px',
                  borderRadius: 6,
                  border: 'none',
                  backgroundColor: isTraining ? '#94a3b8' : '#2563eb',
                  color: '#fff',
                  cursor: isTraining ? 'not-allowed' : 'pointer',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: 40,
                }}
              >
                {isTraining && <span style={spinnerStyle} role="status" aria-label="Запуск обучения" />}
                {isTraining ? 'Обновляем модель...' : 'Обновить модель'}
              </button>
              {isTraining && (
                <span style={{ fontSize: 12, color: '#475569' }}>
                  Обучение запущено, дождитесь завершения процесса.
                </span>
              )}
              {trainSuccess && !isTraining && (
                <span style={{ fontSize: 12, color: '#16a34a' }}>{trainSuccess}</span>
              )}
              {trainError && !isTraining && (
                <span style={{ fontSize: 12, color: '#dc2626' }}>{trainError}</span>
              )}
            </div>
          </div>

          {viewMode === 'live' ? (
            <LiveStreamViewer cameras={cameras} normalizedApiBase={normalizedApiBase} />
          ) : (
            <div>
              <h4 style={{ margin: '4px 0 12px' }}>Симулятор потоков для тестирования</h4>
              <SimulatedVisualization onEvent={handleSimEvent} />
              <div style={{ marginTop: 12, background: '#f1f5f9', padding: 12, borderRadius: 8 }}>
                <h5 style={{ margin: '0 0 8px' }}>Последние события симуляции</h5>
                {simEvents.length === 0 ? (
                  <p style={{ margin: 0, color: '#475569' }}>Пока нет сгенерированных детекций.</p>
                ) : (
                  <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid', gap: 8 }}>
                    {simEvents.map(event => (
                      <li key={event.id} style={{ background: '#fff', borderRadius: 6, padding: '8px 10px', border: '1px solid #e2e8f0' }}>
                        <div style={{ fontWeight: 600, display: 'flex', justifyContent: 'space-between' }}>
                          <span>{event.label}</span>
                          <span>{event.probability.toFixed(2)}</span>
                        </div>
                        <div style={{ fontSize: 12, color: '#475569' }}>камера: {event.camera}</div>
                        <div style={{ fontSize: 12, color: '#475569' }}>{new Date(event.ts).toLocaleTimeString()}</div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </div>
      </section>
    </Layout>
  );
};

export default DashboardPage;
