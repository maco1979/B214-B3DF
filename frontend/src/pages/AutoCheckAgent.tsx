import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { AutoCheckResult, ApiResponse } from '@/services/api';
import { apiClient } from '@/services/api';

// 移除所有不存在的UI组件导入

const AutoCheckAgent: React.FC = () => {
  const queryClient = useQueryClient();

  // 移除toast相关代码

  // 状态管理
  const [showCheckForm, setShowCheckForm] = useState(false);
  const [checkConfig, setCheckConfig] = useState({
    check_type: 'code_quality',
    target: './src',
    customTarget: '',
    scheduled: false,
    cron_expression: '0 0 * * *',
    check_depth: 'basic',
  });

  // 获取自动检查结果列表
  const {
    data: checkResults,
    isLoading: isResultsLoading,
    error: resultsError,
  } = useQuery<ApiResponse<AutoCheckResult[]>>({
    queryKey: ['autoCheckResults'],
    queryFn: () => apiClient.getAutoCheckResults(),
  });

  // 获取所有检查类型
  const {
    data: checkTypes,
    isLoading: isTypesLoading,
  } = useQuery<ApiResponse<string[]>>({
    queryKey: ['checkTypes'],
    queryFn: () => apiClient.getCheckTypes(),
  });

  // 执行自动检查
  const runAutoCheckMutation = useMutation<ApiResponse<AutoCheckResult>, Error, typeof checkConfig>({
    mutationFn: config => apiClient.runAutoCheck(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['autoCheckResults'] });
      setShowCheckForm(false);
      setCheckConfig({
        check_type: 'code_quality',
        target: '',
        scheduled: false,
        cron_expression: '0 0 * * *',
        check_depth: 'basic',
      });
      // 替换toast为alert
      alert('自动检查已执行');
    },
    onError: error => {
      // 替换toast为alert
      alert(`自动检查执行失败: ${error.message}`);
    },
  });

  // 处理执行检查
  const handleRunCheck = () => {
    if (!checkConfig.target) {
      // 替换toast为alert
      alert('检查目标不能为空');
      return;
    }

    runAutoCheckMutation.mutate(checkConfig);
  };

  // 获取检查状态的样式
  const getStatusStyle = (status: string) => {
    switch (status) {
      case 'passed':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'running':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'pending':
        return 'bg-muted text-muted-foreground';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  // 获取检查结果的样式
  const getResultStyle = (passed: boolean) => passed ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';

  return (
    <div className="space-y-6 p-4 md:p-6 bg-background text-foreground">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-foreground">自动检查智能体</h1>
        <button
          onClick={() => setShowCheckForm(!showCheckForm)}
          className="px-4 py-2 bg-primary text-primary-foreground hover:bg-primary/90 rounded-md transition-colors"
        >
          {showCheckForm ? '取消' : '执行自动检查'}
        </button>
      </div>

      {/* 执行检查表单 */}
      {showCheckForm && (
        <div className="border rounded-lg p-6 bg-card shadow-sm border border-border">
          <h2 className="text-xl font-semibold mb-4 text-foreground">配置自动检查</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label htmlFor="check-type" className="text-sm font-medium text-foreground">检查类型</label>
              <select
                value={checkConfig.check_type}
                onChange={e => setCheckConfig({ ...checkConfig, check_type: e.target.value })}
                className="w-full px-3 py-2 border border-border bg-input text-foreground rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="">选择检查类型</option>
                {isTypesLoading ?
(
                  <option value="loading" disabled>
                    加载中...
                  </option>
                ) :
checkTypes?.success && checkTypes.data ?
(
                  checkTypes.data.map(type => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))
                ) :
(
                  <>
                    <option value="code_quality">代码质量</option>
                    <option value="security">安全性</option>
                    <option value="performance">性能</option>
                    <option value="dependency">依赖检查</option>
                    <option value="style">代码风格</option>
                  </>
                )}
              </select>
            </div>

            <div className="space-y-2">
              <label htmlFor="check-depth" className="text-sm font-medium text-foreground">检查深度</label>
              <select
                value={checkConfig.check_depth}
                onChange={e => setCheckConfig({ ...checkConfig, check_depth: e.target.value })}
                className="w-full px-3 py-2 border border-border bg-input text-foreground rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="">选择检查深度</option>
                <option value="basic">基础检查</option>
                <option value="standard">标准检查</option>
                <option value="detailed">详细检查</option>
                <option value="comprehensive">全面检查</option>
              </select>
            </div>

            <div className="space-y-2 md:col-span-2">
              <label htmlFor="check-target" className="text-sm font-medium text-foreground">检查目标</label>
              <select
                id="check-target"
                value={checkConfig.target}
                onChange={e => setCheckConfig({ ...checkConfig, target: e.target.value })}
                className="w-full px-3 py-2 border border-border bg-input text-foreground rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="">选择检查目标</option>
                <option value="./src">当前项目源码目录 (./src)</option>
                <option value="./backend">后端代码目录 (./backend)</option>
                <option value="./frontend">前端代码目录 (./frontend)</option>
                <option value="./tests">测试目录 (./tests)</option>
                <option value="./docs">文档目录 (./docs)</option>
                <option value="custom">自定义路径...</option>
              </select>
              {checkConfig.target === 'custom' && (
                <input
                  className="mt-2 w-full px-3 py-2 border border-border bg-input text-foreground rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                  placeholder="输入自定义路径"
                  value={checkConfig.customTarget || ''}
                  onChange={e => setCheckConfig({ ...checkConfig, customTarget: e.target.value })}
                />
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label htmlFor="scheduled-check" className="text-sm font-medium text-foreground">定时检查</label>
                <input
                  type="checkbox"
                  id="scheduled-check"
                  checked={checkConfig.scheduled}
                  onChange={e => setCheckConfig({ ...checkConfig, scheduled: e.target.checked })}
                  className="ml-2 text-primary"
                />
              </div>
            </div>

            {checkConfig.scheduled && (
              <div className="space-y-2">
                <label htmlFor="cron-expression" className="text-sm font-medium text-foreground">Cron表达式</label>
                <input
                  id="cron-expression"
                  placeholder="输入Cron表达式，如：0 0 * * *"
                  value={checkConfig.cron_expression}
                  onChange={e => setCheckConfig({ ...checkConfig, cron_expression: e.target.value })}
                  className="w-full px-3 py-2 border border-border bg-input text-foreground rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
            )}

            <div className="flex items-center justify-end space-x-2 md:col-span-2">
              <button
                onClick={handleRunCheck}
                disabled={runAutoCheckMutation.isPending}
                className="px-4 py-2 bg-primary text-primary-foreground hover:bg-primary/90 rounded-md transition-colors disabled:bg-muted disabled:cursor-not-allowed"
              >
                {runAutoCheckMutation.isPending ?
(
                  <>
                    <span className="mr-2">⏳</span>
                    执行中...
                  </>
                ) :
(
                  '执行自动检查'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 检查结果列表 */}
      <div className="border rounded-lg p-6 bg-card shadow-sm border border-border">
        <h2 className="text-xl font-semibold mb-4 text-foreground">检查结果列表</h2>
        {isResultsLoading ?
(
          <div className="flex justify-center items-center py-8 text-foreground">
            <span className="text-xl">⏳</span>
            <span className="ml-2">加载检查结果中...</span>
          </div>
        ) :
resultsError ?
(
          <div className="p-4 bg-destructive/30 border border-destructive/50 rounded-md text-destructive">
            <h3 className="font-semibold text-destructive">加载失败</h3>
            <p className="text-destructive">{resultsError.message}</p>
          </div>
        ) :
!checkResults?.success ?
(
          <div className="p-4 bg-destructive/30 border border-destructive/50 rounded-md text-destructive">
            <h3 className="font-semibold text-destructive">加载失败</h3>
            <p className="text-destructive">{checkResults?.error || 'Request failed with status code 404'}</p>
          </div>
        ) :
checkResults.data?.length === 0 ?
(
          <div className="p-4 bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-800 rounded-md text-yellow-700 dark:text-yellow-300">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200">无检查结果</h3>
            <p className="text-yellow-700 dark:text-yellow-300">当前没有自动检查结果，请先执行自动检查。</p>
          </div>
        ) :
(
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted border-border">
                  <th className="px-4 py-2 text-left border-b border-border text-foreground">检查ID</th>
                  <th className="px-4 py-2 text-left border-b border-border text-foreground">检查类型</th>
                  <th className="px-4 py-2 text-left border-b border-border text-foreground">检查目标</th>
                  <th className="px-4 py-2 text-left border-b border-border text-foreground">状态</th>
                  <th className="px-4 py-2 text-left border-b border-border text-foreground">结果</th>
                  <th className="px-4 py-2 text-left border-b border-border text-foreground">开始时间</th>
                  <th className="px-4 py-2 text-left border-b border-border text-foreground">结束时间</th>
                  <th className="px-4 py-2 text-left border-b border-border text-foreground">发现问题数</th>
                  <th className="px-4 py-2 text-left border-b border-border text-foreground">操作</th>
                </tr>
              </thead>
              <tbody>
                {checkResults.data?.map(result => (
                  <tr key={result.check_id} className="hover:bg-muted/80 border-border">
                    <td className="px-4 py-2 border-b border-border font-mono text-sm text-foreground">{result.check_id}</td>
                    <td className="px-4 py-2 border-b border-border">
                      <span className="px-2 py-1 bg-muted rounded-full text-xs text-foreground">{result.check_type}</span>
                    </td>
                    <td className="px-4 py-2 border-b border-border max-w-[200px] truncate text-foreground">{result.target}</td>
                    <td className="px-4 py-2 border-b border-border">
                      <span className={`px-2 py-1 rounded-full text-xs ${getStatusStyle(result.status)}`}>
                        {result.status}
                      </span>
                    </td>
                    <td className="px-4 py-2 border-b border-border">
                      <span className={`px-2 py-1 rounded-full text-xs ${getResultStyle(result.passed)}`}>
                        {result.passed ? '通过' : '失败'}
                      </span>
                    </td>
                    <td className="px-4 py-2 border-b border-border text-sm text-foreground">
                      {new Date(result.start_time * 1000).toLocaleString()}
                    </td>
                    <td className="px-4 py-2 border-b border-border text-sm text-foreground">
                      {result.end_time ? new Date(result.end_time * 1000).toLocaleString() : '进行中'}
                    </td>
                    <td className="px-4 py-2 border-b border-border">
                      <span className={`px-2 py-1 rounded-full text-xs ${result.issues_count > 0 ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300' : 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300'}`}>
                        {result.issues_count}
                      </span>
                    </td>
                    <td className="px-4 py-2 border-b border-border">
                      <button className="px-3 py-1 bg-primary text-primary-foreground hover:bg-primary/90 rounded-md text-sm transition-colors">
                        查看详情
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default AutoCheckAgent;
