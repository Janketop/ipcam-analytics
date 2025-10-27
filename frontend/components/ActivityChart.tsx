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

Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend, Filler);

type ActivitySeries = {
  labels: string[];
  counts: number[];
};

type ActivityChartProps = {
  events: EventItem[];
};

const buildActivitySeries = (events: EventItem[]): ActivitySeries => {
  const hourMs = 60 * 60 * 1000;
  const currentHour = new Date();
  currentHour.setMinutes(0, 0, 0);
  const currentHourTs = currentHour.getTime();
  const windowStart = currentHourTs - hourMs * 23;
  const windowEnd = currentHourTs + hourMs;

  const labels: string[] = [];
  const counts = new Array(24).fill(0);

  for (let i = 0; i < 24; i += 1) {
    const bucketDate = new Date(windowStart + i * hourMs);
    labels.push(bucketDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
  }

  events.forEach(event => {
    const ts = new Date(event.start_ts).getTime();
    if (Number.isNaN(ts)) {
      return;
    }
    if (ts < windowStart || ts >= windowEnd) {
      return;
    }

    const index = Math.floor((ts - windowStart) / hourMs);
    if (index >= 0 && index < counts.length) {
      counts[index] += 1;
    }
  });

  return { labels, counts };
};

export const ActivityChart = ({ events }: ActivityChartProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<Chart<'line'> | null>(null);

  const chartData = useMemo<ChartData<'line'>>(() => {
    const series = buildActivitySeries(events);
    return {
      labels: series.labels,
      datasets: [
        {
          label: 'Количество событий',
          data: series.counts,
          borderColor: 'rgba(37, 99, 235, 1)',
          backgroundColor: 'rgba(37, 99, 235, 0.25)',
          fill: 'start',
          tension: 0.35,
          pointRadius: 3,
          pointHoverRadius: 6,
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
            ticks: {
              precision: 0,
            },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title(items) {
                return items[0]?.label ?? '';
              },
              label(context) {
                const value = context.parsed.y ?? 0;
                return `События: ${value}`;
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
