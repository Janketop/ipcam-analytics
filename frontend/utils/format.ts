export const formatDateTime = (value?: string | null) => {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString('ru-RU', {
    hour12: false,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

export const formatDuration = (seconds?: number | null) => {
  if (seconds == null || Number.isNaN(seconds)) {
    return '—';
  }
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const parts: string[] = [];
  if (hours) parts.push(`${hours} ч`);
  if (minutes || hours) parts.push(`${minutes} мин`);
  parts.push(`${secs} с`);
  return parts.join(' ');
};

export const formatNumber = (value?: number | null, fractionDigits = 1) => {
  if (value == null || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(fractionDigits);
};
