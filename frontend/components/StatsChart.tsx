import { useEffect, useRef } from 'react';
import {
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  ChartConfiguration,
  ChartData,
  Legend,
  LinearScale,
  Tooltip,
} from 'chart.js';
import { Stat } from '../types/api';

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend);

type StatsChartProps = {
  stats: Stat[];
};

export const StatsChart = ({ stats }: StatsChartProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const data: ChartData<'bar'> = {
      labels: stats.map(item => item.type),
      datasets: [
        {
          label: 'События за 24 часа',
          data: stats.map(item => item.cnt),
          backgroundColor: 'rgba(37, 99, 235, 0.65)',
        },
      ],
    };

    const config: ChartConfiguration<'bar'> = {
      type: 'bar',
      data,
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
          },
        },
        plugins: {
          legend: { display: true },
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
  }, [stats]);

  return <canvas ref={canvasRef} height={200} />;
};

export default StatsChart;
