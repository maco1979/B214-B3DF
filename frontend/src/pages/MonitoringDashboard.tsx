import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Activity,
  Server,
  Cpu,
  Database,
  Network,
  Users,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  BarChart3,
  Clock,
} from 'lucide-react';
import { useSystemMetricsRealtime, useBlockchainStatusQuery, useEdgeDevicesQuery } from '@/hooks/useSystemQueries';


// 图表组件（简化实现）
const SimpleChart = ({ data, color = '#3b82f6' }: { data: number[], color?: string }) => {
  const maxValue = Math.max(...data, 1);

  return (
    <div className="flex items-end h-16 space-x-1">
      {data.map((value, index) => (
        <div
          key={index}
          className="flex-1 bg-gradient-to-t from-blue-500 to-blue-600 rounded-t"
          style={{
            height: `${(value / maxValue) * 100}%`,
            backgroundColor: color,
          }}
        />
      ))}
    </div>
  );
};

export function MonitoringDashboard() {
  const { data: metrics, isFetching: metricsLoading, refetch: refetchMetrics, error: metricsError } = useSystemMetricsRealtime();

  const { data: blockchainStatus, refetch: refetchBlockchain, error: blockchainError } = useBlockchainStatusQuery();
  const { data: edgeDevices, refetch: refetchEdge, isFetching: edgeLoading, error: edgeError } = useEdgeDevicesQuery();

  const [lastUpdate, setLastUpdate] = useState(new Date());

  const [cpuUsage, setCpuUsage] = useState<number[]>([45, 52, 48, 60, 55, 58, 62, 50]);
  const [memoryUsage, setMemoryUsage] = useState<number[]>([65, 68, 72, 70, 75, 78, 80, 76]);
  const [networkTraffic, setNetworkTraffic] = useState<number[]>([120, 135, 110, 145, 160, 140, 155, 130]);

  // 模拟实时数据更新
  useEffect(() => {
    const interval = setInterval(() => {
      setCpuUsage(prev => [...prev.slice(1), Math.random() * 40 + 40]);
      setMemoryUsage(prev => [...prev.slice(1), Math.random() * 20 + 60]);
      setNetworkTraffic(prev => [...prev.slice(1), Math.random() * 100 + 100]);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const refreshAll = () => {
    refetchMetrics();
    refetchBlockchain();
    refetchEdge();
    setLastUpdate(new Date());
  };

  if (metricsError || blockchainError || edgeError) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-3">
        <div className="text-red-400 text-lg">数据加载失败</div>
        <div className="text-gray-400 text-sm">{(metricsError!).message || (blockchainError!).message || (edgeError!).message}</div>
        <Button variant="tech" onClick={refreshAll}>重试</Button>
      </div>
    );
  }


  const systemStats = [
    {
      label: 'CPU使用率',
      value: `${cpuUsage[cpuUsage.length - 1].toFixed(1)}%`,
      icon: Cpu,
      color: 'text-blue-400',
      trend: 'up',
    },
    {
      label: '内存使用',
      value: `${memoryUsage[memoryUsage.length - 1].toFixed(1)}%`,
      icon: Database,
      color: 'text-green-400',
      trend: 'stable',
    },
    {
      label: '网络流量',
      value: `${networkTraffic[networkTraffic.length - 1].toFixed(0)} MB/s`,
      icon: Network,
      color: 'text-purple-400',
      trend: 'up',
    },
    {
      label: '活跃连接',
      value: metrics?.active_connections || '0',
      icon: Users,
      color: 'text-yellow-400',
      trend: 'stable',
    },
  ];

  const serviceStatus = [
    {
      name: 'AI推理服务',
      status: metrics?.ai_service_status === 'healthy' ? 'healthy' : 'degraded',
      responseTime: '45ms',
      uptime: '99.8%',
    },
    {
      name: '区块链节点',
      status: blockchainStatus?.status === 'healthy' ? 'healthy' : 'degraded',
      responseTime: '120ms',
      uptime: '99.5%',
    },
    {
      name: '数据库服务',
      status: metrics?.database_status === 'healthy' ? 'healthy' : 'degraded',
      responseTime: '25ms',
      uptime: '99.9%',
    },
    {
      name: '边缘计算网关',
      status: edgeDevices && edgeDevices.length > 0 ? 'healthy' : 'degraded',
      responseTime: '80ms',
      uptime: '98.7%',
    },
  ];

  const recentAlerts = [
    {
      type: 'warning',
      message: 'CPU使用率超过80%',
      time: '5分钟前',
      service: '计算节点-01',
    },
    {
      type: 'info',
      message: '区块链同步完成',
      time: '10分钟前',
      service: '区块链网络',
    },
    {
      type: 'error',
      message: '边缘设备连接超时',
      time: '15分钟前',
      service: '边缘网关-02',
    },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'degraded':
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      default:
        return <AlertTriangle className="w-4 h-4 text-red-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-400';
      case 'degraded':
        return 'text-yellow-400';
      default:
        return 'text-red-400';
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'bg-red-500/20 border-red-500/30';
      case 'warning':
        return 'bg-yellow-500/20 border-yellow-500/30';
      case 'info':
        return 'bg-blue-500/20 border-blue-500/30';
      default:
        return 'bg-gray-500/20 border-gray-500/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">系统监控仪表盘</h1>
          <p className="text-gray-300">实时监控系统性能、服务状态和资源使用情况</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-400">
            最后更新: {lastUpdate.toLocaleTimeString()}
          </div>
          <Button
            variant="outline"
            onClick={refreshAll}
            disabled={metricsLoading || edgeLoading}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${(metricsLoading || edgeLoading) ? 'animate-spin' : ''}`} />
            <span>刷新</span>
          </Button>

        </div>
      </div>

      {/* 系统统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {systemStats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index} className="glass-effect hover:neon-glow transition-all duration-300">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  {stat.label}
                </CardTitle>
                <Icon className={`w-4 h-4 ${stat.color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">{stat.value}</div>
                <SimpleChart data={index === 0 ? cpuUsage : index === 1 ? memoryUsage : networkTraffic} />
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 服务状态 */}
        <Card className="glass-effect lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Server className="w-5 h-5 text-tech-primary" />
              <span>服务状态监控</span>
            </CardTitle>
            <CardDescription>实时监控各项服务的运行状态</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {serviceStatus.map((service, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-tech-dark/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(service.status)}
                  <div>
                    <div className="font-medium text-white">{service.name}</div>
                    <div className="text-sm text-gray-400">响应时间: {service.responseTime}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={getStatusColor(service.status)}>
                    {service.status === 'healthy' ? '正常' : '异常'}
                  </div>
                  <div className="text-xs text-gray-400">可用性: {service.uptime}</div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* 最近告警 */}
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
              <span>最近告警</span>
            </CardTitle>
            <CardDescription>系统最近的告警信息</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {recentAlerts.map((alert, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg border ${getAlertColor(alert.type)}`}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="text-sm font-medium text-white">{alert.service}</span>
                  <span className="text-xs text-gray-400">{alert.time}</span>
                </div>
                <div className="text-sm text-gray-300">{alert.message}</div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* 性能指标图表 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-tech-primary" />
              <span>CPU使用率趋势</span>
            </CardTitle>
            <CardDescription>最近几分钟的CPU使用情况</CardDescription>
          </CardHeader>
          <CardContent>
            <SimpleChart data={cpuUsage} color="#3b82f6" />
            <div className="flex justify-between mt-2 text-sm text-gray-400">
              <span>最低: {Math.min(...cpuUsage).toFixed(1)}%</span>
              <span>平均: {(cpuUsage.reduce((a, b) => a + b) / cpuUsage.length).toFixed(1)}%</span>
              <span>最高: {Math.max(...cpuUsage).toFixed(1)}%</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-green-400" />
              <span>内存使用趋势</span>
            </CardTitle>
            <CardDescription>最近几分钟的内存使用情况</CardDescription>
          </CardHeader>
          <CardContent>
            <SimpleChart data={memoryUsage} color="#10b981" />
            <div className="flex justify-between mt-2 text-sm text-gray-400">
              <span>最低: {Math.min(...memoryUsage).toFixed(1)}%</span>
              <span>平均: {(memoryUsage.reduce((a, b) => a + b) / memoryUsage.length).toFixed(1)}%</span>
              <span>最高: {Math.max(...memoryUsage).toFixed(1)}%</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 区块链状态 */}
      {blockchainStatus && (
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              <span>区块链网络状态</span>
            </CardTitle>
            <CardDescription>Hyperledger Fabric网络监控</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-tech-primary">
                  {blockchainStatus.latest_block?.block_number || '--'}
                </div>
                <div className="text-sm text-gray-400">区块高度</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">
                  {blockchainStatus.latest_block?.transaction_count || '--'}
                </div>
                <div className="text-sm text-gray-400">交易数量</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {blockchainStatus.initialized ? '是' : '否'}
                </div>
                <div className="text-sm text-gray-400">网络初始化</div>
              </div>
              <div className="text-center">
                <div className={`text-2xl font-bold ${
                  blockchainStatus.status === 'healthy' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {blockchainStatus.status === 'healthy' ? '正常' : '异常'}
                </div>
                <div className="text-sm text-gray-400">网络状态</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
