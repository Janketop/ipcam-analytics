import { useCallback, useEffect, useMemo, useState } from 'react';
import type { MouseEvent } from 'react';
import type { CameraZone } from '../types/api';

type ZonesEditorProps = {
  value: CameraZone[];
  onChange: (zones: CameraZone[]) => void;
  disabled?: boolean;
};

const COLORS = ['#2563eb', '#16a34a', '#f97316', '#9333ea', '#0ea5e9'];

const generateZoneId = (): string => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `zone-${Math.random().toString(36).slice(2, 10)}`;
};

const clamp01 = (value: number): number => {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
};

const ZonesEditor = ({ value, onChange, disabled = false }: ZonesEditorProps) => {
  const [selectedId, setSelectedId] = useState<string | null>(() => value[0]?.id ?? null);

  useEffect(() => {
    if (value.length === 0) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !value.some(zone => zone.id === selectedId)) {
      setSelectedId(value[0]?.id ?? null);
    }
  }, [value, selectedId]);

  useEffect(() => {
    const missingId = value.some(zone => !zone.id);
    if (!missingId) {
      return;
    }
    const withIds = value.map(zone => (zone.id ? zone : { ...zone, id: generateZoneId() }));
    onChange(withIds);
  }, [value, onChange]);

  const selectedZone = useMemo(
    () => value.find(zone => zone.id === selectedId) ?? null,
    [value, selectedId],
  );

  const handleSelectZone = useCallback(
    (id: string | undefined) => {
      if (!id || disabled) {
        return;
      }
      setSelectedId(id);
    },
    [disabled],
  );

  const handleAddZone = useCallback(() => {
    if (disabled) {
      return;
    }
    const newZone: CameraZone = {
      id: generateZoneId(),
      name: `Зона ${value.length + 1}`,
      points: [],
    };
    onChange([...value, newZone]);
    setSelectedId(newZone.id ?? null);
  }, [disabled, onChange, value]);

  const handleRemoveZone = useCallback(() => {
    if (disabled || !selectedZone?.id) {
      return;
    }
    const next = value.filter(zone => zone.id !== selectedZone.id);
    onChange(next);
    setSelectedId(next[0]?.id ?? null);
  }, [disabled, onChange, selectedZone, value]);

  const handleUndoPoint = useCallback(() => {
    if (disabled || !selectedZone?.id || selectedZone.points.length === 0) {
      return;
    }
    onChange(
      value.map(zone =>
        zone.id === selectedZone.id
          ? { ...zone, points: zone.points.slice(0, zone.points.length - 1) }
          : zone,
      ),
    );
  }, [disabled, onChange, selectedZone, value]);

  const handleClearZone = useCallback(() => {
    if (disabled || !selectedZone?.id || selectedZone.points.length === 0) {
      return;
    }
    onChange(
      value.map(zone => (zone.id === selectedZone.id ? { ...zone, points: [] } : zone)),
    );
  }, [disabled, onChange, selectedZone, value]);

  const handleNameChange = useCallback(
    (id: string, name: string) => {
      onChange(
        value.map(zone =>
          zone.id === id
            ? {
                ...zone,
                name,
              }
            : zone,
        ),
      );
    },
    [onChange, value],
  );

  const handleCanvasClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (disabled || !selectedZone?.id) {
        return;
      }
      const rect = event.currentTarget.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        return;
      }
      const relativeX = (event.clientX - rect.left) / rect.width;
      const relativeY = (event.clientY - rect.top) / rect.height;
      if (!Number.isFinite(relativeX) || !Number.isFinite(relativeY)) {
        return;
      }
      const point = { x: clamp01(relativeX), y: clamp01(relativeY) };
      onChange(
        value.map(zone =>
          zone.id === selectedZone.id
            ? { ...zone, points: [...zone.points, point] }
            : zone,
        ),
      );
    },
    [disabled, onChange, selectedZone, value],
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button
          type="button"
          onClick={handleAddZone}
          disabled={disabled}
          style={{
            padding: '6px 12px',
            borderRadius: 6,
            border: '1px solid #2563eb',
            background: disabled ? '#e2e8f0' : '#2563eb',
            color: disabled ? '#94a3b8' : '#fff',
            fontWeight: 600,
            cursor: disabled ? 'not-allowed' : 'pointer',
            minWidth: 140,
          }}
        >
          Добавить зону
        </button>
        <button
          type="button"
          onClick={handleRemoveZone}
          disabled={disabled || !selectedZone}
          style={{
            padding: '6px 12px',
            borderRadius: 6,
            border: '1px solid #fca5a5',
            background: disabled || !selectedZone ? '#fee2e2' : '#f87171',
            color: '#7f1d1d',
            fontWeight: 600,
            cursor: disabled || !selectedZone ? 'not-allowed' : 'pointer',
            minWidth: 140,
          }}
        >
          Удалить зону
        </button>
        <button
          type="button"
          onClick={handleUndoPoint}
          disabled={disabled || !selectedZone || selectedZone.points.length === 0}
          style={{
            padding: '6px 12px',
            borderRadius: 6,
            border: '1px solid #cbd5f5',
            background: '#f8fafc',
            color: '#0f172a',
            fontWeight: 600,
            cursor:
              disabled || !selectedZone || selectedZone.points.length === 0
                ? 'not-allowed'
                : 'pointer',
          }}
        >
          Отменить точку
        </button>
        <button
          type="button"
          onClick={handleClearZone}
          disabled={disabled || !selectedZone || selectedZone.points.length === 0}
          style={{
            padding: '6px 12px',
            borderRadius: 6,
            border: '1px solid #cbd5f5',
            background: '#fff7ed',
            color: '#c2410c',
            fontWeight: 600,
            cursor:
              disabled || !selectedZone || selectedZone.points.length === 0
                ? 'not-allowed'
                : 'pointer',
          }}
        >
          Очистить зону
        </button>
      </div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div
          style={{
            flex: '0 0 240px',
            minWidth: 220,
            border: '1px solid #e2e8f0',
            borderRadius: 12,
            padding: 12,
            background: '#fff',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
          }}
        >
          {value.length === 0 ? (
            <p style={{ margin: 0, fontSize: 13, color: '#64748b' }}>Зон пока нет — добавьте первую.</p>
          ) : (
            value.map((zone, index) => {
              const zoneId = zone.id ?? `zone-${index}`;
              const isSelected = zoneId === selectedId;
              const indicatorColor = COLORS[index % COLORS.length];
              return (
                <div
                  key={zoneId}
                  onClick={() => handleSelectZone(zoneId)}
                  style={{
                    borderRadius: 10,
                    border: isSelected ? `2px solid ${indicatorColor}` : '1px solid #e2e8f0',
                    padding: '8px 10px',
                    background: isSelected ? '#eff6ff' : '#f8fafc',
                    cursor: disabled ? 'default' : 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        background: indicatorColor,
                        flexShrink: 0,
                      }}
                    />
                    <input
                      type="text"
                      value={zone.name ?? ''}
                      onChange={event => handleNameChange(zoneId, event.target.value)}
                      onClick={event => event.stopPropagation()}
                      disabled={disabled}
                      placeholder={`Зона ${index + 1}`}
                      style={{
                        flex: 1,
                        border: '1px solid #cbd5f5',
                        borderRadius: 6,
                        padding: '4px 6px',
                        fontSize: 13,
                        background: '#fff',
                      }}
                    />
                  </div>
                  <div style={{ fontSize: 12, color: zone.points.length >= 3 ? '#166534' : '#b91c1c' }}>
                    Точек: {zone.points.length}
                  </div>
                </div>
              );
            })
          )}
        </div>
        <div
          style={{
            flex: '1 1 320px',
            minWidth: 280,
            borderRadius: 12,
            border: '1px solid #e2e8f0',
            background: '#fff',
            padding: 12,
          }}
        >
          <div
            onClick={handleCanvasClick}
            role="presentation"
            style={{
              position: 'relative',
              borderRadius: 10,
              overflow: 'hidden',
              background: '#0f172a0d',
              border: '1px solid #cbd5f5',
              aspectRatio: '16 / 9',
              cursor: !selectedZone || disabled ? 'not-allowed' : 'crosshair',
            }}
          >
            <svg
              width="100%"
              height="100%"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              style={{ display: 'block', pointerEvents: 'none' }}
            >
              {value.map((zone, index) => {
                const pointsAttr = zone.points
                  .map(point => `${Math.round(clamp01(point.x) * 100)},${Math.round(clamp01(point.y) * 100)}`)
                  .join(' ');
                if (!pointsAttr) {
                  return null;
                }
                const color = COLORS[index % COLORS.length];
                const isSelected = zone.id === selectedZone?.id;
                return (
                  <g key={zone.id ?? `zone-${index}`}>
                    <polygon
                      points={pointsAttr}
                      fill={color}
                      fillOpacity={isSelected ? 0.3 : 0.15}
                      stroke={color}
                      strokeWidth={isSelected ? 2.5 : 1.5}
                    />
                    {isSelected &&
                      zone.points.map((point, pointIndex) => (
                        <circle
                          key={`pt-${pointIndex}`}
                          cx={clamp01(point.x) * 100}
                          cy={clamp01(point.y) * 100}
                          r={2.6}
                          fill={color}
                        />
                      ))}
                  </g>
                );
              })}
            </svg>
            {!selectedZone && (
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#64748b',
                  fontSize: 14,
                  fontWeight: 600,
                  textAlign: 'center',
                  padding: '0 16px',
                  background: 'rgba(248, 250, 252, 0.75)',
                }}
              >
                Выберите зону слева и кликните по полю, чтобы добавить точки.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ZonesEditor;
