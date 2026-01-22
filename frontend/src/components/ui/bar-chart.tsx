import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { useTheme } from '../../hooks/useTheme';

// 注册Chart.js组件
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
);

interface BarChartProps {
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      backgroundColor?: string;
      borderColor?: string;
      borderWidth?: number;
      borderRadius?: number;
      borderSkipped?: boolean;
    }[];
  };
  options?: any;
  className?: string;
}

export const BarChart: React.FC<BarChartProps> = ({ data, options, className }) => {
  const { theme } = useTheme();

  // 默认配置
  const defaultOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: theme === 'light' ? '#111827' : '#f3f4f6',
        },
      },
      tooltip: {
        backgroundColor: theme === 'light' ? '#ffffff' : '#1f2937',
        titleColor: theme === 'light' ? '#111827' : '#f3f4f6',
        bodyColor: theme === 'light' ? '#4b5563' : '#d1d5db',
        borderColor: theme === 'light' ? '#e5e7eb' : '#374151',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        grid: {
          color: theme === 'light' ? '#e5e7eb' : '#374151',
        },
        ticks: {
          color: theme === 'light' ? '#6b7280' : '#9ca3af',
        },
      },
      y: {
        grid: {
          color: theme === 'light' ? '#e5e7eb' : '#374151',
        },
        ticks: {
          color: theme === 'light' ? '#6b7280' : '#9ca3af',
        },
      },
    },
  };

  return (
    <div className={`w-full h-full ${className}`}>
      <Bar data={data} options={{ ...defaultOptions, ...options }} />
    </div>
  );
};
