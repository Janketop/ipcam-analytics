import { useEffect, useMemo, useRef } from 'react';
import {
  CategoryScale,
  Chart,
  ChartConfiguration,
  ChartData,
  Filler,
  Legend,
  LineController,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from 'chart.js';
import { EventItem } from '../types/api';

const ACTIVITY_STATES = ['WORKING', 'NOT_WORKING', 'AWAY'] as const;
type ActivityState = (typeof ACTIVITY_STATES)[number];

Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend, Filler);

type ActivitySeries = {
  labels: string[];
  datasets: Record<ActivityState, number[]>;
};

type ActivityChartProps = {
  events: EventItem[];
};

const buildActivitySeries = (events: EventItem[]): ActivitySeries => {
  const bucketDurationMs = 60 * 60 * 1000; // 1 час
  const now = new Date();
  now.setMinutes(0, 0, 0);
  const currentHourTs = now.getTime();
  const windowStart = currentHourTs - bucketDurationMs * 23;
  const windowEnd = currentHourTs + bucketDurationMs;

  const labels: string[] = [];
  const datasets: Record<ActivityState, number[]> = {
    WORKING: new Array(24).fill(0),
    NOT_WORKING: new Array(24).fill(0),
    AWAY: new Array(24).fill(0),
  };

  for (let i = 0; i < 24; i += 1) {
    const bucketDate = new Date(windowStart + i * bucketDurationMs);
    labels.push(bucketDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
  }

  events.forEach(event => {
    if (!ACTIVITY_STATES.includes(event.type as ActivityState)) {
      return;
    }

    const ts = new Date(event.start_ts).getTime();
    if (Number.isNaN(ts)) {
      return;
    }
    if (ts < windowStart || ts >= windowEnd) {
      return;
    }

    const index = Math.floor((ts - windowStart) / bucketDurationMs);
    const bucket = datasets[event.type as ActivityState];
    if (bucket && index >= 0 && index < bucket.length) {
      bucket[index] += 1;
    }
  });

  return { labels, datasets };
};

export const ActivityChart = ({ events }: ActivityChartProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<Chart<'line'> | null>(null);

  const chartData = useMemo<ChartData<'line'>>(() => {
    const series = buildActivitySeries(events);
    const baseDatasetOptions = {
      tension: 0.35,
      pointRadius: 3,
      pointHoverRadius: 6,
      fill: false as const,
    };

    return {
      labels: series.labels,
      datasets: [
        {
          label: 'Работает',
          data: series.datasets.WORKING,
          borderColor: 'rgba(34,197,94,1)',
          backgroundColor: 'rgba(34,197,94,0.25)',
          ...baseDatasetOptions,
        },
        {
          label: 'Не работает',
          data: series.datasets.NOT_WORKING,
          borderColor: 'rgba(250,204,21,1)',
          backgroundColor: 'rgba(250,204,21,0.25)',
          ...baseDatasetOptions,
        },
        {
          label: 'Нет на месте',
          data: series.datasets.AWAY,
          borderColor: 'rgba(148,163,184,1)',
          backgroundColor: 'rgba(148,163,184,0.25)',
          ...baseDatasetOptions,
        },
      ],
    };
  }, [events]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const config: ChartConfiguration<'line'> = {
      type: 'line',
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            stacked: false,
            ticks: {
              precision: 0,
            },
          },
          x: {
            ticks: {
              autoSkip: true,
              maxTicksLimit: 12,
            },
          },
        },
        plugins: {
          legend: { display: true },
          tooltip: {
            callbacks: {
              title(items) {
                return items[0]?.label ?? '';
              },
              label(context) {
                const value = context.parsed.y ?? 0;
                return `${context.dataset.label ?? 'События'}: ${value}`;
              },
            },
          },
        },
      },
    };

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    chartRef.current = new Chart(ctx, config);

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [chartData]);

  return (
    <div style={{ height: 240 }}>
      <canvas ref={canvasRef} />
    </div>
  );
};

export default ActivityChart;
