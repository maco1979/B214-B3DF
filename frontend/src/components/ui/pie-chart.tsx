import React from 'react';
import { Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { useTheme } from '../../hooks/useTheme';

// 注册Chart.js组件
ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
);

interface PieChartProps {
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      backgroundColor: string[];
      borderColor?: string[];
      borderWidth?: number;
    }[];
  };
  options?: any;
  className?: string;
}

export const PieChart: React.FC<PieChartProps> = ({ data, options, className }) => {
  const { theme } = useTheme();

  // 默认配置
  const defaultOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          color: theme === 'light' ? '#111827' : '#f3f4f6',
          padding: 20,
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
  };

  return (
    <div className={`w-full h-full ${className}`}>
      <Pie data={data} options={{ ...defaultOptions, ...options }} />
    </div>
  );
};
