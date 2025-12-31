import React from 'react';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { useTheme } from '../../hooks/useTheme';

// 注册Chart.js组件
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Title,
  Tooltip,
  Legend
);

interface RadarChartProps {
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      borderColor?: string;
      backgroundColor?: string;
      fill?: boolean;
      pointBackgroundColor?: string;
      pointBorderColor?: string;
      pointHoverBackgroundColor?: string;
      pointHoverBorderColor?: string;
    }[];
  };
  options?: any;
  className?: string;
}

export const RadarChart: React.FC<RadarChartProps> = ({ data, options, className }) => {
  const { theme } = useTheme();

  // 默认配置 - 针对雷达图优化
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
      r: {
        angleLines: {
          color: theme === 'light' ? '#e5e7eb' : '#374151',
        },
        grid: {
          color: theme === 'light' ? '#e5e7eb' : '#374151',
        },
        pointLabels: {
          color: theme === 'light' ? '#6b7280' : '#9ca3af',
        },
        ticks: {
          color: theme === 'light' ? '#6b7280' : '#9ca3af',
          backdropColor: 'transparent',
        },
        suggestedMin: 0,
        suggestedMax: 100,
      },
    },
  };

  // 确保数据集设置了必要的属性
  const processedData = {
    ...data,
    datasets: data.datasets.map(dataset => ({
      ...dataset,
      fill: dataset.fill !== undefined ? dataset.fill : true,
      pointBackgroundColor: dataset.pointBackgroundColor || dataset.borderColor,
      pointBorderColor: dataset.pointBorderColor || '#ffffff',
      pointHoverBackgroundColor: dataset.pointHoverBackgroundColor || '#ffffff',
      pointHoverBorderColor: dataset.pointHoverBorderColor || dataset.borderColor,
    }))
  };

  return (
    <div className={`w-full h-full ${className}`}>
      <Radar data={processedData} options={{ ...defaultOptions, ...options }} />
    </div>
  );
};
