import type { CSSProperties } from 'react';
import { CleanupSettings, CleanupState } from '../types/api';
import { formatDateTime, formatNumber } from '../utils/format';

type CleanupStatusProps = {
  cleanup: CleanupState | null;
  settings: CleanupSettings | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => Promise<void> | void;
  onRun: () => Promise<void> | void;
  runInProgress: boolean;
  runError: string | null;
  runSuccess: string | null;
  onClearSnapshots: () => Promise<void> | void;
  clearSnapshotsInProgress: boolean;
  clearSnapshotsError: string | null;
  clearSnapshotsSuccess: string | null;
  onClearEvents: () => Promise<void> | void;
  clearEventsInProgress: boolean;
  clearEventsError: string | null;
  clearEventsSuccess: string | null;
};

const infoRowStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'minmax(140px, 1fr) minmax(100px, 1fr)',
  gap: 8,
  fontSize: 13,
  color: '#0f172a',
};

const CleanupStatus = ({
  cleanup,
  settings,
  loading,
  error,
  onRefresh,
  onRun,
  runInProgress,
  runError,
  runSuccess,
  onClearSnapshots,
  clearSnapshotsInProgress,
  clearSnapshotsError,
  clearSnapshotsSuccess,
  onClearEvents,
  clearEventsInProgress,
  clearEventsError,
  clearEventsSuccess,
}: CleanupStatusProps) => {
  return (
    <div
      style={{
        background: '#f8fafc',
        border: '1px solid #e2e8f0',
        borderRadius: 8,
        padding: 12,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <h3 style={{ margin: '0 0 4px' }}>Очистка истории</h3>
        <button
          onClick={() => onRefresh()}
          disabled={loading}
          style={{
            padding: '6px 10px',
            borderRadius: 6,
            border: '1px solid #cbd5f5',
            backgroundColor: loading ? '#e2e8f0' : '#fff',
            color: '#1e293b',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: 13,
          }}
        >
          {loading ? 'Обновляем…' : 'Обновить'}
        </button>
      </div>

      {error && (
        <p style={{ margin: 0, color: '#b91c1c', fontSize: 13 }}>Не удалось получить данные: {error}</p>
      )}

      {settings && (
        <p style={{ margin: 0, color: '#475569', fontSize: 13 }}>
          Хранение событий: {settings.retentionDays} дн. · Неразмеченных карточек: {settings.faceSampleRetentionDays} дн. ·
          Интервал фоновой очистки: {formatNumber(settings.cleanupIntervalHours, 1)} ч.
        </p>
      )}

      {cleanup ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={infoRowStyle}>
            <span style={{ color: '#475569' }}>Последний запуск:</span>
            <span>{formatDateTime(cleanup.last_run)}</span>
          </div>
          <div style={infoRowStyle}>
            <span style={{ color: '#475569' }}>Граница событий:</span>
            <span>{formatDateTime(cleanup.cutoff)}</span>
          </div>
          <div style={infoRowStyle}>
            <span style={{ color: '#475569' }}>Граница карточек:</span>
            <span>{formatDateTime(cleanup.face_sample_cutoff)}</span>
          </div>
          <div style={infoRowStyle}>
            <span style={{ color: '#475569' }}>Удалено событий:</span>
            <span>{cleanup.deleted_events}</span>
          </div>
          <div style={infoRowStyle}>
            <span style={{ color: '#475569' }}>Удалено снимков:</span>
            <span>{cleanup.deleted_snapshots}</span>
          </div>
          <div style={infoRowStyle}>
            <span style={{ color: '#475569' }}>Удалено копий в датасете:</span>
            <span>{cleanup.deleted_dataset_copies}</span>
          </div>
          <div style={infoRowStyle}>
            <span style={{ color: '#475569' }}>Удалено карточек лиц:</span>
            <span>{cleanup.deleted_face_samples}</span>
          </div>
          {cleanup.error && (
            <div style={{ color: '#b91c1c', fontSize: 13 }}>Последняя ошибка: {cleanup.error}</div>
          )}
          {cleanup.in_progress && (
            <div style={{ color: '#2563eb', fontSize: 13 }}>Очистка сейчас выполняется в фоне…</div>
          )}
        </div>
      ) : (
        <p style={{ margin: 0, color: '#475569', fontSize: 13 }}>
          Данные ещё не получены. Нажмите «Обновить», чтобы загрузить состояние очистки.
        </p>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <button
          onClick={() => onRun()}
          disabled={runInProgress || cleanup?.in_progress}
          style={{
            padding: '8px 12px',
            borderRadius: 6,
            border: 'none',
            backgroundColor: runInProgress || cleanup?.in_progress ? '#94a3b8' : '#2563eb',
            color: '#fff',
            cursor: runInProgress || cleanup?.in_progress ? 'not-allowed' : 'pointer',
            fontWeight: 600,
            minHeight: 40,
          }}
        >
          {runInProgress ? 'Удаляем…' : 'Удалить устаревшие данные сейчас'}
        </button>
        {runSuccess && <span style={{ fontSize: 12, color: '#16a34a' }}>{runSuccess}</span>}
        {runError && <span style={{ fontSize: 12, color: '#dc2626' }}>{runError}</span>}
      </div>

      <div
        style={{
          display: 'grid',
          gap: 10,
          borderTop: '1px solid #e2e8f0',
          paddingTop: 10,
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <button
            onClick={() => onClearSnapshots()}
            disabled={
              clearSnapshotsInProgress || runInProgress || cleanup?.in_progress || clearEventsInProgress
            }
            style={{
              padding: '8px 12px',
              borderRadius: 6,
              border: 'none',
              backgroundColor:
                clearSnapshotsInProgress || runInProgress || cleanup?.in_progress || clearEventsInProgress
                  ? '#94a3b8'
                  : '#0f766e',
              color: '#fff',
              cursor:
                clearSnapshotsInProgress || runInProgress || cleanup?.in_progress || clearEventsInProgress
                  ? 'not-allowed'
                  : 'pointer',
              fontWeight: 600,
              minHeight: 40,
            }}
          >
            {clearSnapshotsInProgress ? 'Удаляем кадры…' : 'Удалить все кадры и карточки'}
          </button>
          {clearSnapshotsSuccess && (
            <span style={{ fontSize: 12, color: '#16a34a' }}>{clearSnapshotsSuccess}</span>
          )}
          {clearSnapshotsError && (
            <span style={{ fontSize: 12, color: '#dc2626' }}>{clearSnapshotsError}</span>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <button
            onClick={() => onClearEvents()}
            disabled={clearEventsInProgress || runInProgress || cleanup?.in_progress || clearSnapshotsInProgress}
            style={{
              padding: '8px 12px',
              borderRadius: 6,
              border: 'none',
              backgroundColor:
                clearEventsInProgress || runInProgress || cleanup?.in_progress || clearSnapshotsInProgress
                  ? '#94a3b8'
                  : '#dc2626',
              color: '#fff',
              cursor:
                clearEventsInProgress || runInProgress || cleanup?.in_progress || clearSnapshotsInProgress
                  ? 'not-allowed'
                  : 'pointer',
              fontWeight: 600,
              minHeight: 40,
            }}
          >
            {clearEventsInProgress ? 'Удаляем события…' : 'Удалить все события'}
          </button>
          {clearEventsSuccess && <span style={{ fontSize: 12, color: '#16a34a' }}>{clearEventsSuccess}</span>}
          {clearEventsError && <span style={{ fontSize: 12, color: '#dc2626' }}>{clearEventsError}</span>}
        </div>
      </div>
    </div>
  );
};

export default CleanupStatus;
