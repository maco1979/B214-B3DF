import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
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
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface AreaChartProps {
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      borderColor?: string;
      backgroundColor?: string;
      fill?: boolean;
      tension?: number;
    }[];
  };
  options?: any;
  className?: string;
}

export const AreaChart: React.FC<AreaChartProps> = ({ data, options, className }) => {
  const { theme } = useTheme();

  // 默认配置 - 针对面积图优化
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
        mode: 'index' as const,
        intersect: false,
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
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
  };

  // 确保数据集设置了fill属性
  const processedData = {
    ...data,
    datasets: data.datasets.map(dataset => ({
      ...dataset,
      fill: dataset.fill !== undefined ? dataset.fill : true,
      tension: dataset.tension !== undefined ? dataset.tension : 0.4,
    }))
  };

  return (
    <div className={`w-full h-full ${className}`}>
      <Line data={processedData} options={{ ...defaultOptions, ...options }} />
    </div>
  );
};
