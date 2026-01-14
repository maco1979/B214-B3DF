import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Activity,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  BarChart3,
  Cpu,
  Network,
  Database,
  Clock,
  Zap,
  Target,
  Shield,
  Cloud,
  Server,
} from 'lucide-react';
import { usePerformanceMetricsQuery, useOptimizationStatusQuery } from '@/hooks/useSystemQueries';
import { AreaChart } from '@/components/ui/area-chart';
import { apiClient } from '@/services/api';


// 简化图表组件
const SimpleChart = ({ data, color = '#3b82f6', height = '60px' }: {
  data: number[],
  color?: string,
  height?: string
}) => {
  const maxValue = Math.max(...data, 1);

  return (
    <div className={'flex items-end space-x-1'} style={{ height }}>
      {data.map((value, index) => (
        <div
          key={index}
          className="flex-1 bg-gradient-to-t rounded-t transition-all duration-300 hover:opacity-80"
          style={{
            height: `${(value / maxValue) * 100}%`,
            backgroundColor: color,
          }}
        />
      ))}
    </div>
  );
};

// 性能指标卡片组件
const MetricCard = ({
  title,
  value,
  unit,
  trend,
  chartData,
  color,
  icon: Icon,
}: {
  title: string
  value: number
  unit: string
  trend: 'up' | 'down' | 'stable'
  chartData: number[]
  color: string
  icon: any
}) => {
  const getTrendColor = () => {
    switch (trend) {
      case 'up': return 'text-green-400';
      case 'down': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getTrendIcon = () => {
    switch (trend) {
      case 'up': return '↗';
      case 'down': return '↘';
      default: return '→';
    }
  };

  return (
    <Card className="glass-effect hover:neon-glow transition-all duration-300">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-gray-400">
          {title}
        </CardTitle>
        <Icon className={`w-4 h-4 ${color}`} />
      </CardHeader>
      <CardContent>
        <div className="flex items-end justify-between mb-2">
          <div className="text-2xl font-bold text-white">
            {value.toFixed(1)}{unit}
          </div>
          <div className={`text-sm ${getTrendColor()}`}>
            {getTrendIcon()} {trend}
          </div>
        </div>
        <SimpleChart data={chartData} color={color.replace('text-', '').replace('-400', '')} height="40px" />
      </CardContent>
    </Card>
  );
};

// 优化建议组件
const OptimizationCard = ({
  title,
  description,
  impact,
  status: initialStatus,
  estimatedImprovement,
}: {
  title: string
  description: string
  impact: 'high' | 'medium' | 'low'
  status: 'pending' | 'applied' | 'failed'
  estimatedImprovement: string
}) => {
  const [status, setStatus] = useState(initialStatus);
  const [loading, setLoading] = useState(false);

  const getImpactColor = () => {
    switch (impact) {
      case 'high': return 'text-red-400';
      case 'medium': return 'text-yellow-400';
      default: return 'text-green-400';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'applied': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'failed': return <AlertTriangle className="w-4 h-4 text-red-400" />;
      default: return <Clock className="w-4 h-4 text-yellow-400" />;
    }
  };

  const handleApply = async () => {
    try {
      setLoading(true);
      const response = await apiClient.applyOptimization('system', title);
      if (response.success) {
        setStatus('applied');
      } else {
        setStatus('failed');
      }
    } catch (error) {
      console.error('应用优化失败:', error);
      setStatus('failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="glass-effect border-l-4 border-l-tech-primary">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">{title}</CardTitle>
          <div className="flex items-center space-x-2">
            <span className={`text-sm ${getImpactColor()}`}>影响: {impact}</span>
            {getStatusIcon()}
          </div>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-400">预计改进: {estimatedImprovement}</span>
          <Button
            size="sm"
            variant="outline"
            onClick={handleApply}
            disabled={status !== 'pending' || loading}
          >
            {loading ? '处理中...' : status === 'pending' ? '应用优化' : status === 'applied' ? '已应用' : '重试'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export function PerformanceMonitoring() {
  const { data: performanceMetrics, isFetching: metricsLoading, refetch: refetchMetrics, error: metricsError } = usePerformanceMetricsQuery();
  const { data: optimizationStatus, refetch: refetchOptimization, isFetching: optimizationLoading, error: optimizationError } = useOptimizationStatusQuery();

  const [lastUpdate, setLastUpdate] = useState(new Date());


  // 模拟实时数据
  const [migrationAccuracy, setMigrationAccuracy] = useState<number[]>([85, 87, 89, 88, 90, 92, 91, 93]);
  const [edgeLatency, setEdgeLatency] = useState<number[]>([45, 42, 38, 35, 32, 30, 28, 25]);
  const [decisionTime, setDecisionTime] = useState<number[]>([120, 115, 110, 105, 100, 95, 90, 85]);
  const [integrationSuccess, setIntegrationSuccess] = useState<number[]>([92, 93, 94, 95, 96, 97, 98, 99]);

  // 模拟实时数据更新
  useEffect(() => {
    const interval = setInterval(() => {
      setMigrationAccuracy(prev => [...prev.slice(1), Math.random() * 10 + 85]);
      setEdgeLatency(prev => [...prev.slice(1), Math.random() * 20 + 20]);
      setDecisionTime(prev => [...prev.slice(1), Math.random() * 30 + 70]);
      setIntegrationSuccess(prev => [...prev.slice(1), Math.random() * 5 + 95]);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const refreshAll = () => {
    refetchMetrics();
    refetchOptimization();
    setLastUpdate(new Date());
  };

  if (metricsError || optimizationError) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-3">
        <div className="text-red-400 text-lg">数据加载失败</div>
        <div className="text-gray-400 text-sm">{(metricsError!).message || (optimizationError!).message}</div>
        <Button variant="tech" onClick={refreshAll}>重试</Button>
      </div>
    );
  }


  // 性能指标数据
  const performanceStats: Array<{
    title: string;
    value: number;
    unit: string;
    trend: 'up' | 'down' | 'stable';
    chartData: number[];
    color: string;
    icon: any;
  }> = [
    {
      title: '迁移学习精度',
      value: migrationAccuracy[migrationAccuracy.length - 1],
      unit: '%',
      trend: migrationAccuracy[migrationAccuracy.length - 1] > migrationAccuracy[0] ?
'up' :
             migrationAccuracy[migrationAccuracy.length - 1] < migrationAccuracy[0] ? 'down' : 'stable',
      chartData: migrationAccuracy,
      color: 'text-green-400',
      icon: Target,
    },
    {
      title: '边缘计算延迟',
      value: edgeLatency[edgeLatency.length - 1],
      unit: 'ms',
      trend: edgeLatency[edgeLatency.length - 1] < edgeLatency[0] ?
'up' :
             edgeLatency[edgeLatency.length - 1] > edgeLatency[0] ? 'down' : 'stable',
      chartData: edgeLatency,
      color: 'text-blue-400',
      icon: Zap,
    },
    {
      title: '决策时间',
      value: decisionTime[decisionTime.length - 1],
      unit: 'ms',
      trend: decisionTime[decisionTime.length - 1] < decisionTime[0] ?
'up' :
             decisionTime[decisionTime.length - 1] > decisionTime[0] ? 'down' : 'stable',
      chartData: decisionTime,
      color: 'text-purple-400',
      icon: Activity,
    },
    {
      title: '集成成功率',
      value: integrationSuccess[integrationSuccess.length - 1],
      unit: '%',
      trend: integrationSuccess[integrationSuccess.length - 1] > integrationSuccess[0] ?
'up' :
             integrationSuccess[integrationSuccess.length - 1] < integrationSuccess[0] ? 'down' : 'stable',
      chartData: integrationSuccess,
      color: 'text-yellow-400',
      icon: Shield,
    },
  ];

  // 优化建议数据
  const optimizationSuggestions = [
    {
      title: '迁移学习参数调优',
      description: '检测到迁移学习精度有提升空间，建议调整学习率和批次大小',
      impact: 'high' as const,
      status: 'pending' as const,
      estimatedImprovement: '精度提升3-5%',
    },
    {
      title: '边缘资源重分配',
      description: '部分边缘节点资源利用率不均衡，建议重新分配计算任务',
      impact: 'medium' as const,
      status: 'applied' as const,
      estimatedImprovement: '延迟降低10-15%',
    },
    {
      title: '决策模型优化',
      description: '决策引擎响应时间可以进一步优化，考虑模型压缩',
      impact: 'medium' as const,
      status: 'pending' as const,
      estimatedImprovement: '响应时间减少20-30%',
    },
  ];

  // 系统健康状态
  const systemHealth = {
    overall: 'healthy', // healthy, warning, critical
    components: [
      { name: '迁移学习模块', status: 'healthy', latency: '45ms', uptime: '99.8%' },
      { name: '边缘计算网关', status: 'healthy', latency: '28ms', uptime: '99.5%' },
      { name: '决策引擎', status: 'warning', latency: '85ms', uptime: '99.2%' },
      { name: '集成服务', status: 'healthy', latency: '60ms', uptime: '99.9%' },
    ],
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      default:
        return <AlertTriangle className="w-4 h-4 text-red-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-400';
      case 'warning':
        return 'text-yellow-400';
      default:
        return 'text-red-400';
    }
  };

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">性能监控与优化</h1>
          <p className="text-gray-300">实时监控迁移学习和边缘计算集成的性能指标</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-400">
            最后更新: {lastUpdate.toLocaleTimeString()}
          </div>
          <Button
            variant="outline"
            onClick={refreshAll}
            disabled={metricsLoading}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${(metricsLoading || optimizationLoading) ? 'animate-spin' : ''}`} />
            <span>刷新</span>
          </Button>
        </div>
      </div>


      {/* 性能指标卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {performanceStats.map((stat, index) => (
          <MetricCard key={index} {...stat} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 系统健康状态 */}
        <Card className="glass-effect lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Server className="w-5 h-5 text-tech-primary" />
              <span>系统健康状态</span>
            </CardTitle>
            <CardDescription>各模块的运行状态和性能指标</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {systemHealth.components.map((component, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-tech-dark/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(component.status)}
                  <div>
                    <div className="font-medium text-white">{component.name}</div>
                    <div className="text-sm text-gray-400">响应时间: {component.latency}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={getStatusColor(component.status)}>
                    {component.status === 'healthy' ?
'正常' :
                     component.status === 'warning' ? '警告' : '异常'}
                  </div>
                  <div className="text-xs text-gray-400">可用性: {component.uptime}</div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* 优化状态 */}
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-green-400" />
              <span>优化状态</span>
            </CardTitle>
            <CardDescription>自动优化系统的运行状态</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-tech-primary">
                {optimizationStatus?.total_recommendations || 12}
              </div>
              <div className="text-sm text-gray-400">总优化建议</div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-xl font-bold text-green-400">
                  {optimizationStatus?.applied_optimizations || 8}
                </div>
                <div className="text-xs text-gray-400">已应用</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-yellow-400">
                  {optimizationStatus?.total_recommendations - (optimizationStatus?.applied_optimizations || 0) || 4}
                </div>
                <div className="text-xs text-gray-400">待处理</div>
              </div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold text-white">
                性能提升: {optimizationStatus?.performance_improvement?.overall_performance_gain || 12.3}%
              </div>
              <div className="text-xs text-gray-400">自上次优化以来</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 优化建议 */}
      <Card className="glass-effect">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-purple-400" />
            <span>优化建议</span>
          </CardTitle>
          <CardDescription>基于性能数据的智能优化建议</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {optimizationSuggestions.map((suggestion, index) => (
            <OptimizationCard key={index} {...suggestion} />
          ))}
        </CardContent>
      </Card>

      {/* 详细性能图表 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="w-5 h-5 text-green-400" />
              <span>迁移学习精度趋势</span>
            </CardTitle>
            <CardDescription>最近迁移学习任务的精度变化</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <AreaChart
                data={{
                  labels: ['任务1', '任务2', '任务3', '任务4', '任务5', '任务6', '任务7', '任务8'],
                  datasets: [{
                    label: '迁移学习精度',
                    data: migrationAccuracy,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                  }],
                }}
              />
            </div>
            <div className="flex justify-between mt-4 text-sm text-gray-400">
              <span>最低: {Math.min(...migrationAccuracy).toFixed(1)}%</span>
              <span>平均: {(migrationAccuracy.reduce((a, b) => a + b) / migrationAccuracy.length).toFixed(1)}%</span>
              <span>最高: {Math.max(...migrationAccuracy).toFixed(1)}%</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-blue-400" />
              <span>边缘计算延迟趋势</span>
            </CardTitle>
            <CardDescription>边缘计算与云计算延迟对比</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <AreaChart
                data={{
                  labels: ['任务1', '任务2', '任务3', '任务4', '任务5', '任务6', '任务7', '任务8'],
                  datasets: [{
                    label: '边缘计算延迟',
                    data: edgeLatency,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                  }, {
                    label: '云计算延迟（基准）',
                    data: edgeLatency.map(x => x * 3),
                    borderColor: '#6b7280',
                    backgroundColor: 'rgba(107, 114, 128, 0.1)',
                  }],
                }}
              />
            </div>
            <div className="mt-4 text-center">
              <span className="text-sm text-green-400">
                延迟降低: {((1 - edgeLatency[edgeLatency.length - 1] / (edgeLatency[edgeLatency.length - 1] * 3)) * 100).toFixed(1)}%
              </span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
