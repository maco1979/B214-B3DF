import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { Device, AutomationRule, ApiResponse } from '@/services/api';
import { apiClient } from '@/services/api';

const DeviceAutomation: React.FC = () => {
  const queryClient = useQueryClient();

  // 状态管理
  const [showControlForm, setShowControlForm] = useState(false);
  const [showRuleForm, setShowRuleForm] = useState(false);
  const [editingRuleId, setEditingRuleId] = useState<string | null>(null);
  const [selectedDevice, setSelectedDevice] = useState<Device | null>(null);
  const [controlAction, setControlAction] = useState('turn_on');
  const [controlParams, setControlParams] = useState<any>({ value: 100 });
  const [automationRule, setAutomationRule] = useState<AutomationRule>({
    name: '',
    description: '',
    device_id: 0,
    rule_id: '',
    trigger: { type: 'time', value: '0 0 * * *' },
    conditions: [],
    actions: [{ type: 'turn_on' }],
    enabled: true,
    created_at: '',
  });

  // 获取设备列表
  const {
    data: devices,
    isLoading: isDevicesLoading,
    error: devicesError,
    refetch: refetchDevices,
  } = useQuery<ApiResponse<Device[]>>({
    queryKey: ['devices'],
    queryFn: async () => apiClient.getDevices(),
  });

  // 扫描设备
  const scanDevicesMutation = useMutation<ApiResponse<Device[]>, Error>({
    mutationFn: async () => apiClient.scanDevices(),
    onSuccess: data => {
      queryClient.invalidateQueries({ queryKey: ['devices'] });
      alert(`成功扫描到 ${data.data?.length || 0} 台设备`);
    },
    onError: error => {
      alert(`扫描设备失败: ${error.message}`);
    },
  });

  // 控制设备
  const controlDeviceMutation = useMutation<ApiResponse<any>, Error, { deviceId: number, params: any }>({
    mutationFn: async ({ deviceId, params }) => apiClient.controlDevice(deviceId, params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['devices'] });
      setShowControlForm(false);
      alert('设备控制命令已发送');
    },
    onError: error => {
      alert(`设备控制失败: ${error.message}`);
    },
  });

  // 切换设备连接状态
  const toggleConnectionMutation = useMutation<ApiResponse<any>, Error, { deviceId: number, connect: boolean }>({
    mutationFn: async ({ deviceId, connect }) => apiClient.toggleDeviceConnection(deviceId, connect),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['devices'] });
      alert('设备连接状态已切换');
    },
    onError: error => {
      alert(`切换设备连接状态失败: ${error.message}`);
    },
  });

  // 获取设备状态
  const getDeviceStatusMutation = useMutation<ApiResponse<any>, Error, number>({
    mutationFn: async deviceId => apiClient.getDeviceStatus(deviceId),
    onSuccess: (data, deviceId) => {
      alert(`设备 ${deviceId} 状态: ${JSON.stringify(data.data)}`);
    },
    onError: error => {
      alert(`获取设备状态失败: ${error.message}`);
    },
  });

  // 处理设备控制
  const handleControlDevice = () => {
    if (!selectedDevice) {
      alert('请先选择设备');
      return;
    }

    const params = {
      action: controlAction,
      ...controlParams,
    };

    controlDeviceMutation.mutate({ deviceId: selectedDevice.id, params });
  };

  // 获取自动化规则列表
  const {
    data: automationRules,
    isLoading: isRulesLoading,
    error: rulesError,
    refetch: refetchRules,
  } = useQuery<ApiResponse<AutomationRule[]>>({
    queryKey: ['automationRules'],
    queryFn: async () => apiClient.getAutomationRules(),
  });

  // 创建自动化规则
  const createRuleMutation = useMutation<ApiResponse<AutomationRule>, Error, typeof automationRule>({
    mutationFn: async rule => apiClient.createAutomationRule(rule),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automationRules'] });
      alert('自动化规则已创建');
      setShowRuleForm(false);
      resetRuleForm();
    },
    onError: error => {
      alert(`创建自动化规则失败: ${error.message}`);
    },
  });

  // 删除自动化规则
  const deleteRuleMutation = useMutation<ApiResponse<any>, Error, string>({
    mutationFn: async ruleId => apiClient.deleteAutomationRule(ruleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automationRules'] });
      alert('自动化规则已删除');
    },
    onError: error => {
      alert(`删除自动化规则失败: ${error.message}`);
    },
  });

  // 启用/禁用自动化规则
  const toggleRuleStatusMutation = useMutation<ApiResponse<any>, Error, { ruleId: string; enabled: boolean }>({
    mutationFn: async ({ ruleId, enabled }) => {
      if (enabled) {
        return apiClient.enableAutomationRule(ruleId);
      }
        return apiClient.disableAutomationRule(ruleId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automationRules'] });
      alert('自动化规则状态已更新');
    },
    onError: error => {
      alert(`更新自动化规则状态失败: ${error.message}`);
    },
  });

  // 更新自动化规则
  const updateRuleMutation = useMutation<ApiResponse<AutomationRule>, Error, { ruleId: string; rule: typeof automationRule }>({
    mutationFn: async ({ ruleId, rule }) => apiClient.updateAutomationRule(ruleId, rule),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automationRules'] });
      alert('自动化规则已更新');
      setShowRuleForm(false);
      setEditingRuleId(null);
      resetRuleForm();
    },
    onError: error => {
      alert(`更新自动化规则失败: ${error.message}`);
    },
  });

  // 重置规则表单
  const resetRuleForm = () => {
    setAutomationRule({
      name: '',
      description: '',
      device_id: 0,
      rule_id: '',
      trigger: { type: 'time', value: '0 0 * * *' },
      conditions: [],
      actions: [{ type: 'turn_on' }],
      enabled: true,
      created_at: '',
    });
  };

  // 处理编辑规则
  const handleEditRule = (rule: AutomationRule) => {
    setEditingRuleId(rule.rule_id);
    setAutomationRule({
      name: rule.name,
      description: rule.description,
      device_id: rule.device_id,
      rule_id: rule.rule_id,
      trigger: rule.trigger || { type: 'time', value: '0 0 * * *' },
      conditions: rule.conditions || [],
      actions: rule.actions || [{ type: 'turn_on' }],
      enabled: rule.enabled,
      created_at: rule.created_at,
    });
    setShowRuleForm(true);
  };

  // 处理规则保存（创建或更新）
  const handleSaveRule = () => {
    if (!automationRule.device_id || !automationRule.name) {
      alert('规则名称和设备ID不能为空');
      return;
    }

    if (editingRuleId) {
      updateRuleMutation.mutate({ ruleId: editingRuleId, rule: automationRule });
    } else {
      createRuleMutation.mutate(automationRule);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">设备自动化管理</h1>
        <div className="flex gap-2">
          <button
            onClick={() => scanDevicesMutation.mutate()}
            disabled={scanDevicesMutation.isPending}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
          >
            {scanDevicesMutation.isPending ? '扫描中...' : '扫描设备'}
          </button>
          <button
            onClick={async () => refetchDevices()}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            刷新设备列表
          </button>
          <button
            onClick={() => setShowRuleForm(!showRuleForm)}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            {showRuleForm ? '取消' : '创建规则'}
          </button>
        </div>
      </div>

      {/* 创建/编辑自动化规则表单 */}
      {showRuleForm && (
        <div className="bg-card p-6 rounded-lg shadow-md border">
          <h2 className="text-xl font-semibold mb-4 text-foreground">{editingRuleId ? '编辑自动化规则' : '创建自动化规则'}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label htmlFor="rule-name" className="block text-sm font-medium text-foreground">规则名称</label>
              <input
                id="rule-name"
                placeholder="输入规则名称"
                value={automationRule.name}
                onChange={e => setAutomationRule({ ...automationRule, name: e.target.value })}
                className="border border-gray-300 rounded-md p-2 w-full"
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="rule-device" className="block text-sm font-medium text-foreground">选择设备</label>
              <select
                value={automationRule.device_id.toString()}
                onChange={e => setAutomationRule({ ...automationRule, device_id: parseInt(e.target.value) })}
                className="border border-gray-300 rounded-md p-2 w-full"
              >
                <option value="0">选择设备</option>
                {devices?.success && devices.data?.map(device => (
                  <option key={device.id} value={device.id.toString()}>
                    {device.name} ({device.type})
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2 md:col-span-2">
              <label htmlFor="rule-description" className="block text-sm font-medium text-foreground">规则描述</label>
              <textarea
                id="rule-description"
                placeholder="输入规则描述"
                value={automationRule.description}
                onChange={e => setAutomationRule({ ...automationRule, description: e.target.value })}
                className="border border-gray-300 rounded-md p-2 w-full min-h-[100px]"
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="rule-trigger" className="block text-sm font-medium text-foreground">触发条件类型</label>
              <select
                value={automationRule.trigger.type}
                onChange={e => setAutomationRule({
                  ...automationRule,
                  trigger: { ...automationRule.trigger, type: e.target.value },
                })}
                className="border border-gray-300 rounded-md p-2 w-full"
              >
                <option value="time">定时触发</option>
                <option value="device_status">设备状态变化</option>
                <option value="sensor_threshold">传感器阈值</option>
                <option value="scene_trigger">场景触发</option>
              </select>
            </div>

            <div className="space-y-2">
              <label htmlFor="trigger-value" className="block text-sm font-medium text-foreground">触发值</label>
              <input
                id="trigger-value"
                placeholder={automationRule.trigger.type === 'time' ? 'Cron表达式，如：0 0 * * *' : '触发值'}
                value={automationRule.trigger.value as string}
                onChange={e => setAutomationRule({
                  ...automationRule,
                  trigger: { ...automationRule.trigger, value: e.target.value },
                })}
                className="border border-gray-300 rounded-md p-2 w-full"
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="rule-action" className="block text-sm font-medium text-foreground">执行动作</label>
              <select
                value={automationRule.actions[0].type}
                onChange={e => setAutomationRule({
                  ...automationRule,
                  actions: [{ ...automationRule.actions[0], type: e.target.value }],
                })}
                className="border border-gray-300 rounded-md p-2 w-full"
              >
                <option value="turn_on">打开设备</option>
                <option value="turn_off">关闭设备</option>
                <option value="adjust">调整设备</option>
                <option value="toggle">切换状态</option>
              </select>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label htmlFor="rule-enabled" className="block text-sm font-medium text-foreground">启用规则</label>
                <input
                  id="rule-enabled"
                  type="checkbox"
                  checked={automationRule.enabled}
                  onChange={e => setAutomationRule({ ...automationRule, enabled: e.target.checked })}
                />
              </div>
            </div>

            <div className="flex items-center justify-end space-x-2 md:col-span-2">
              <button
                onClick={() => {
                  setShowRuleForm(false);
                  setEditingRuleId(null);
                  resetRuleForm();
                }}
                className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded"
              >
                取消
              </button>
              <button
                onClick={handleSaveRule}
                disabled={createRuleMutation.isPending || updateRuleMutation.isPending}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
              >
                {editingRuleId ? '更新规则' : '创建规则'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 设备控制表单 */}
      {showControlForm && selectedDevice && (
        <div className="bg-card p-6 rounded-lg shadow-md border">
          <h2 className="text-xl font-semibold mb-4 text-foreground">控制设备：{selectedDevice.name}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label htmlFor="control-action" className="block text-sm font-medium text-foreground">控制动作</label>
              <select
                value={controlAction}
                onChange={e => setControlAction(e.target.value)}
                className="border border-gray-300 rounded-md p-2 w-full"
              >
                <option value="turn_on">打开</option>
                <option value="turn_off">关闭</option>
                <option value="adjust_brightness">调整亮度</option>
                <option value="adjust_volume">调整音量</option>
                <option value="set_temperature">设置温度</option>
                <option value="toggle">切换状态</option>
              </select>
            </div>

            {(controlAction === 'adjust_brightness' || controlAction === 'adjust_volume' || controlAction === 'set_temperature') && (
              <div className="space-y-2">
                <label htmlFor="control-value" className="block text-sm font-medium text-foreground">控制值</label>
                <input
                  id="control-value"
                  type="number"
                  placeholder="输入控制值"
                  value={controlParams.value}
                  onChange={e => setControlParams({ value: parseInt(e.target.value) || 0 })}
                  className="border border-gray-300 rounded-md p-2 w-full"
                />
              </div>
            )}

            <div className="flex items-center justify-end space-x-2 md:col-span-2">
              <button
                onClick={handleControlDevice}
                disabled={controlDeviceMutation.isPending}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
              >
                {controlDeviceMutation.isPending ? '控制中...' : '执行控制'}
              </button>
              <button
                onClick={() => setShowControlForm(false)}
                className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded"
              >
                取消
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 设备列表 */}
      <div className="bg-card p-6 rounded-lg shadow-md border">
        <h2 className="text-xl font-semibold mb-4 text-foreground">设备列表</h2>
        {isDevicesLoading ?
(
          <div className="flex justify-center items-center py-8">
            <span>加载设备列表中...</span>
          </div>
        ) :
devicesError ?
(
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded">
            <strong className="font-bold text-red-800 dark:text-red-200">加载失败：</strong>
            <span>{devicesError.message}</span>
          </div>
        ) :
!devices?.success ?
(
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded">
            <strong className="font-bold text-red-800 dark:text-red-200">加载失败：</strong>
            <span>{devices?.error || '未知错误'}</span>
          </div>
        ) :
devices.data?.length === 0 ?
(
          <div className="bg-yellow-100 dark:bg-yellow-900/30 border border-yellow-400 dark:border-yellow-800 text-yellow-700 dark:text-yellow-300 px-4 py-3 rounded">
            <strong className="font-bold text-yellow-800 dark:text-yellow-200">无设备：</strong>
            <span>当前没有可用的设备，请先扫描设备。</span>
          </div>
        ) :
(
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-muted border">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-foreground uppercase tracking-wider">设备ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">名称</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">类型</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">状态</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">连接状态</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">信号强度</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">电量</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">位置</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">操作</th>
                </tr>
              </thead>
              <tbody className="bg-card divide-y divide-gray-200">
                {devices.data?.map(device => (
                  <tr key={device.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono">{device.id}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">{device.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-muted-foreground">{device.type}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${device.status === 'online' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        {device.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${device.connected ? 'bg-green-100 text-green-800' : 'bg-muted text-muted-foreground'}`}>
                        {device.connected ? '已连接' : '未连接'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">{device.signal}%</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">{device.battery}%</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">{device.location}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex gap-2">
                        <button
                          onClick={() => {
                            setSelectedDevice(device);
                            setShowControlForm(true);
                          }}
                          className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-1 px-2 rounded text-xs"
                        >
                          控制
                        </button>
                        <button
                          onClick={() => toggleConnectionMutation.mutate({
                            deviceId: device.id,
                            connect: !device.connected,
                          })}
                          className="bg-gray-200 hover:bg-gray-400 text-gray-800 font-bold py-1 px-2 rounded text-xs"
                        >
                          {device.connected ? '断开' : '连接'}
                        </button>
                        <button
                          onClick={() => getDeviceStatusMutation.mutate(device.id)}
                          className="bg-blue-200 hover:bg-blue-400 text-blue-800 font-bold py-1 px-2 rounded text-xs"
                        >
                          状态
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 自动化规则列表 */}
      <div className="bg-card p-6 rounded-lg shadow-md border">
        <h2 className="text-xl font-semibold mb-4 text-foreground">自动化规则列表</h2>
        {isRulesLoading ?
(
          <div className="flex justify-center items-center py-8">
            <span>加载规则列表中...</span>
          </div>
        ) :
rulesError ?
(
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded">
            <strong className="font-bold text-red-800 dark:text-red-200">加载失败：</strong>
            <span>{rulesError.message}</span>
          </div>
        ) :
!automationRules?.success ?
(
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded">
            <strong className="font-bold text-red-800 dark:text-red-200">加载失败：</strong>
            <span>{automationRules?.error}</span>
          </div>
        ) :
automationRules.data?.length === 0 ?
(
          <div className="bg-yellow-100 dark:bg-yellow-900/30 border border-yellow-400 dark:border-yellow-800 text-yellow-700 dark:text-yellow-300 px-4 py-3 rounded">
            <strong className="font-bold text-yellow-800 dark:text-yellow-200">无规则：</strong>
            <span>当前没有自动化规则，请先创建规则。</span>
          </div>
        ) :
(
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-muted border">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-foreground uppercase tracking-wider">规则ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">名称</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">设备</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">触发类型</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">状态</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">创建时间</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">上次执行</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">操作</th>
                </tr>
              </thead>
              <tbody className="bg-card divide-y divide-gray-200">
                {automationRules.data?.map(rule => (
                  <tr key={rule.rule_id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono">{rule.rule_id}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">{rule.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {devices?.success && devices.data?.find(device => device.id === rule.device_id)?.name || rule.device_id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-muted-foreground">
                        {rule.trigger?.type || '未设置'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${rule.enabled ? 'bg-green-100 text-green-800' : 'bg-muted text-muted-foreground'}`}>
                        {rule.enabled ? '启用' : '禁用'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {rule.created_at ? new Date(rule.created_at).toLocaleString() : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {rule.last_executed ? new Date(rule.last_executed).toLocaleString() : '从未执行'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleEditRule(rule)}
                          className="bg-gray-200 hover:bg-gray-400 text-gray-800 font-bold py-1 px-2 rounded text-xs"
                        >
                          编辑
                        </button>
                        <button
                          onClick={() => deleteRuleMutation.mutate(rule.rule_id)}
                          disabled={deleteRuleMutation.isPending}
                          className="bg-red-200 hover:bg-red-400 text-red-800 font-bold py-1 px-2 rounded text-xs disabled:opacity-50"
                        >
                          删除
                        </button>
                        <button
                          onClick={() => toggleRuleStatusMutation.mutate({ ruleId: rule.rule_id, enabled: !rule.enabled })}
                          disabled={toggleRuleStatusMutation.isPending}
                          className={`font-bold py-1 px-2 rounded text-xs disabled:opacity-50 ${rule.enabled ? 'bg-yellow-200 hover:bg-yellow-400 text-yellow-800' : 'bg-green-200 hover:bg-green-400 text-green-800'}`}
                        >
                          {rule.enabled ? '禁用' : '启用'}
                        </button>
                      </div>
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

export default DeviceAutomation;
