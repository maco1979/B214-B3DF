/**
 * 注意力热力图组件
 * 使用ECharts实现，支持画布复用、虚拟渲染和大数据量优化
 */

import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import { useTheme } from './../hooks/useTheme';
import { DataProcessor } from './../utils/dataProcessor';

/**
 * 注意力热力图配置
 */
export interface AttentionHeatmapConfig {

  /** 注意力权重数据 */
  attentionData: number[][] | number[]

  /** 维度信息（如果数据是一维数组） */
  dimensions?: number[]

  /** 样本ID */
  sampleId?: string

  /** 层数 */
  layer?: number

  /** 注意力头 */
  head?: number

  /** 域类型 */
  domainType?: 'intra' | 'inter'

  /** 序列长度 */
  seqLen?: number

  /** 显示标签 */
  showLabels?: boolean

  /** 显示颜色刻度 */
  showColorScale?: boolean

  /** 交互类型 */
  interactionType?: 'hover' | 'click' | 'none'

  /** 是否启用数据降采样 */
  enableDownsampling?: boolean

  /** 降采样阈值，超过该值则进行降采样 */
  downsamplingThreshold?: number

  /** 标签显示间隔，0表示显示所有标签 */
  labelInterval?: number

  /** 是否启用缩放和平移 */
  enableZoom?: boolean
}

/**
 * 注意力热力图组件
 */
export const AttentionHeatmap: React.FC<AttentionHeatmapConfig & React.HTMLAttributes<HTMLDivElement>> = ({
  attentionData,
  dimensions,
  sampleId,
  layer,
  head,
  domainType,
  seqLen,
  showLabels = true,
  showColorScale = true,
  interactionType = 'hover',
  enableDownsampling = true,
  downsamplingThreshold = 200,
  labelInterval = 10,
  enableZoom = true,
  className,
  ...props
}) => {
  const { theme } = useTheme();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // 数据降采样函数
  const downsampleData = (data: number[][]): number[][] => {
    if (data.length <= downsamplingThreshold) {
      return data;
    }

    const step = Math.ceil(data.length / downsamplingThreshold);
    const result: number[][] = [];

    for (let i = 0; i < data.length; i += step) {
      const row: number[] = [];
      for (let j = 0; j < data[i].length; j += step) {
        // 计算区域平均值
        let sum = 0;
        let count = 0;
        for (let x = 0; x < step && i + x < data.length; x++) {
          for (let y = 0; y < step && j + y < data[i + x].length; y++) {
            sum += data[i + x][j + y];
            count++;
          }
        }
        row.push(count > 0 ? sum / count : 0);
      }
      result.push(row);
    }

    return result;
  };

  // 初始化图表
  useEffect(() => {
    if (!chartRef.current) {
 return;
}

    // 如果图表实例已存在，直接返回
    if (chartInstanceRef.current) {
      setIsLoading(false);
      return;
    }

    // 创建图表实例
    const chart = echarts.init(chartRef.current, theme === 'dark' ? 'dark' : undefined);
    chartInstanceRef.current = chart;
    setIsLoading(false);

    // 监听窗口大小变化，自动调整图表大小
    const handleResize = () => {
      chart.resize();
    };

    window.addEventListener('resize', handleResize);

    // 清理函数
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.dispose();
      chartInstanceRef.current = null;
    };
  }, [theme]);

  // 更新图表数据
  useEffect(() => {
    if (!chartInstanceRef.current || isLoading) {
 return;
}

    let processedData: number[][] = [];

    // 处理一维数组数据
    if (Array.isArray(attentionData) && typeof attentionData[0] === 'number' && dimensions) {
      // 将一维数组转换为二维数组
      processedData = DataProcessor.unflatten(attentionData, dimensions) as number[][];
    } else if (Array.isArray(attentionData) && Array.isArray(attentionData[0])) {
      // 二维数组直接使用
      processedData = attentionData as number[][];
    } else {
      console.error('Invalid attention data format');
      return;
    }

    // 确保数据是二维数组
    if (!Array.isArray(processedData) || !Array.isArray(processedData[0])) {
      console.error('Processed data is not a 2D array');
      return;
    }

    // 应用数据降采样
    const finalData = enableDownsampling ? downsampleData(processedData) : processedData;

    // 生成标签，根据降采样调整间隔
    const labels = Array.from({ length: finalData.length }, (_, i) => {
      // 只显示间隔为labelInterval的标签
      if (labelInterval > 0 && i % labelInterval !== 0) {
        return '';
      }
      // 计算原始索引
      const originalIndex = enableDownsampling ? i * Math.ceil(processedData.length / finalData.length) : i;
      return `Token ${originalIndex}`;
    });

    // 构建ECharts配置
    const option: echarts.EChartsOption = {
      title: {
        text: `注意力热力图 - 样本 ${sampleId} | 层 ${layer} | 头 ${head} | 域 ${domainType}`,
        left: 'center',
        textStyle: {
          fontSize: 14,
        },
      },
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          // 计算原始索引
          const step = enableDownsampling ? Math.ceil(processedData.length / finalData.length) : 1;
          const originalRowIndex = params.dataIndex * step;
          const originalColIndex = params.data[1] * step;

          return `
            <div style="padding: 8px;">
              <div>Token: ${originalRowIndex} → ${originalColIndex}</div>
              <div>注意力权重: ${params.data[2].toFixed(4)}</div>
            </div>
          `;
        },
      },
      grid: {
        height: '60%',
        top: '15%',
        containLabel: true,
      },
      // 启用缩放和平移
      dataZoom: enableZoom ?
[
        {
          type: 'inside',
          xAxisIndex: 0,
          start: 0,
          end: 100,
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
        {
          type: 'inside',
          yAxisIndex: 0,
          start: 0,
          end: 100,
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
      ] :
[],
      xAxis: {
        type: 'category',
        data: labels,
        splitArea: {
          show: true,
          areaStyle: {
            color: ['rgba(255, 255, 255, 0.3)', 'rgba(200, 200, 200, 0.3)'],
          },
        },
        axisLabel: {
          show: showLabels,
          rotate: 45,
          fontSize: 10,
          formatter: (value: string) =>
            // 只显示非空标签
             value || '',

        },
        silent: interactionType === 'none',
        axisLine: {
          show: true,
        },
        axisTick: {
          show: false,
        },
      },
      yAxis: {
        type: 'category',
        data: labels,
        splitArea: {
          show: true,
          areaStyle: {
            color: ['rgba(255, 255, 255, 0.3)', 'rgba(200, 200, 200, 0.3)'],
          },
        },
        axisLabel: {
          show: showLabels,
          fontSize: 10,
          formatter: (value: string) =>
            // 只显示非空标签
             value || '',

        },
        silent: interactionType === 'none',
        axisLine: {
          show: true,
        },
        axisTick: {
          show: false,
        },
      },
      visualMap: showColorScale ?
{
        min: 0,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '5%',
        inRange: {
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'],
        },
        itemWidth: 10,
        itemHeight: 100,
      } :
undefined,
      series: [
        {
          name: '注意力权重',
          type: 'heatmap',
          data: finalData.flatMap((row, rowIndex) =>
            row.map((value, colIndex) => [colIndex, rowIndex, value]),
          ),
          label: {
            show: false,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
            disabled: interactionType === 'none',
          },
          large: true, // 启用大数据量优化
          largeThreshold: 1000, // 大数据量阈值
          progressive: 5000, // 增加渐进式渲染阈值，提升大数据量性能
          progressiveThreshold: 10000, // 增加渐进式渲染阈值
          animation: false, // 禁用动画，提升大数据量渲染性能
          animationThreshold: 5000, // 动画阈值
        },
      ],
    };

    // 更新图表
    chartInstanceRef.current.setOption(option, { notMerge: true });
  }, [attentionData, dimensions, sampleId, layer, head, domainType, showLabels, showColorScale, interactionType, isLoading, enableDownsampling, downsamplingThreshold, labelInterval, enableZoom]);

  return (
    <div
      ref={chartRef}
      className={`w-full h-full ${className}`}
      style={{ minHeight: '400px' }}
      {...props}
    >
      {isLoading && (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600 dark:text-gray-300">加载注意力热力图...</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AttentionHeatmap;
