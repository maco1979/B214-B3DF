import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { HabitAnalysisResult, UserProfile, ApiResponse } from '@/services/api';
import { apiClient } from '@/services/api';

// 移除所有不存在的UI组件导入

const UserHabitAnalysis: React.FC = () => {
  const queryClient = useQueryClient();

  // 移除toast相关代码

  // 状态管理
  const [showRecordForm, setShowRecordForm] = useState(false);
  const [selectedUserId, setSelectedUserId] = useState('default_user');
  const [timeRange, setTimeRange] = useState(7);
  const [behaviorRecord, setBehaviorRecord] = useState({
    user_id: 'default_user',
    behavior_type: 'view',
    target: '',
    duration: 0,
    metadata: '{}',
  });

  // 获取用户习惯分析结果
  const {
    data: analysisResult,
    isLoading: isAnalysisLoading,
    error: analysisError,
  } = useQuery<ApiResponse<HabitAnalysisResult>>({
    queryKey: ['userHabits', selectedUserId, timeRange],
    queryFn: () => apiClient.analyzeUserHabits(selectedUserId, timeRange),
  });

  // 获取用户画像
  const {
    data: userProfile,
    isLoading: isProfileLoading,
    error: profileError,
  } = useQuery<ApiResponse<UserProfile>>({
    queryKey: ['userProfile', selectedUserId],
    queryFn: () => apiClient.getUserProfile(selectedUserId),
  });

  // 获取习惯建议
  const {
    data: recommendations,
    isLoading: isRecommendationsLoading,
  } = useQuery<ApiResponse<{ recommendations: any[] }>>({
    queryKey: ['habitRecommendations', selectedUserId],
    queryFn: () => apiClient.getHabitRecommendations(selectedUserId),
  });

  // 记录用户行为
  const recordBehaviorMutation = useMutation<ApiResponse<any>, Error, typeof behaviorRecord>({
    mutationFn: record => apiClient.recordUserBehavior({
      user_id: record.user_id,
      behavior_type: record.behavior_type,
      target: record.target,
      duration: record.duration,
      metadata: JSON.parse(record.metadata),
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userHabits'] });
      queryClient.invalidateQueries({ queryKey: ['userProfile'] });
      queryClient.invalidateQueries({ queryKey: ['habitRecommendations'] });
      setShowRecordForm(false);
      resetForm();
      // 替换toast为alert
      alert('用户行为已记录');
    },
    onError: error => {
      // 替换toast为alert
      alert(`记录用户行为失败: ${error.message}`);
    },
  });

  // 重置表单
  const resetForm = () => {
    setBehaviorRecord({
      user_id: selectedUserId,
      behavior_type: 'view',
      target: '',
      duration: 0,
      metadata: '{}',
    });
  };

  // 提交用户行为记录
  const handleSubmitRecord = () => {
    if (!behaviorRecord.behavior_type || !behaviorRecord.target) {
      // 替换toast为alert
      alert('行为类型和目标不能为空');
      return;
    }

    try {
      JSON.parse(behaviorRecord.metadata);
    } catch (error) {
      // 替换toast为alert
      alert('元数据必须是有效的JSON格式');
      return;
    }

    recordBehaviorMutation.mutate(behaviorRecord);
  };

  // 格式化行为分布数据，转换为图表所需格式
  const formatBehaviorDistribution = () => {
    if (!analysisResult?.success || !analysisResult.data) {
 return [];
}

    return Object.entries(analysisResult.data.behavior_distribution).map(([type, count]) => ({
      name: type,
      value: count,
    }));
  };

  // 获取活跃时间段的可读格式
  const getActiveHoursText = () => {
    if (!analysisResult?.success || !analysisResult.data) {
 return '无数据';
}

    const hours = analysisResult.data.active_hours;
    if (hours.length === 0) {
 return '无明显活跃时间段';
}

    // 将小时数组转换为连续时间段
    const sortedHours = hours.sort((a, b) => a - b);
    const periods: string[] = [];
    let start = sortedHours[0];

    for (let i = 1; i <= sortedHours.length; i++) {
      if (i === sortedHours.length || sortedHours[i] !== sortedHours[i - 1] + 1) {
        if (start === sortedHours[i - 1]) {
          periods.push(`${start}:00`);
        } else {
          periods.push(`${start}:00 - ${sortedHours[i - 1] + 1}:00`);
        }
        start = sortedHours[i];
      }
    }

    return periods.join(', ');
  };

  return (
    <div className="space-y-6 p-4 md:p-6 bg-cyber-black text-foreground">
      <div className="flex flex-wrap justify-between items-center gap-4">
        <h1 className="text-3xl font-bold text-foreground">用户习惯分析</h1>
        <div className="flex flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <label htmlFor="user-id" className="text-sm font-medium text-gray-300">用户ID</label>
            <input
              id="user-id"
              value={selectedUserId}
              onChange={e => setSelectedUserId(e.target.value)}
              className="w-32 px-3 py-2 border border-gray-800 bg-cyber-light text-foreground dark:text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="time-range" className="text-sm font-medium text-gray-300">时间范围（天）</label>
            <select
              value={timeRange.toString()}
              onChange={e => setTimeRange(parseInt(e.target.value))}
              className="w-24 px-3 py-2 border border-gray-800 bg-cyber-light text-foreground dark:text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="7">7天</option>
              <option value="14">14天</option>
              <option value="30">30天</option>
              <option value="90">90天</option>
            </select>
          </div>
          <button
            onClick={() => setShowRecordForm(!showRecordForm)}
            className="px-4 py-2 bg-blue-900 text-foreground dark:text-gray-200 hover:bg-blue-800 rounded-md transition-colors border border-blue-800"
          >
            {showRecordForm ? '取消' : '记录行为'}
          </button>
        </div>
      </div>

      {/* 记录用户行为表单 */}
      {showRecordForm && (
        <div className="border rounded-lg p-6 bg-cyber-deep shadow-sm border-gray-800">
          <h2 className="text-xl font-semibold mb-4 text-foreground dark:text-gray-200">记录用户行为</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label htmlFor="behavior-user-id" className="block text-sm font-medium text-gray-300">用户ID</label>
              <input
                id="behavior-user-id"
                value={behaviorRecord.user_id}
                onChange={e => setBehaviorRecord({ ...behaviorRecord, user_id: e.target.value })}
                className="w-full px-3 py-2 border border-gray-800 bg-cyber-light text-foreground dark:text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="behavior-type" className="block text-sm font-medium text-gray-300">行为类型</label>
              <select
                value={behaviorRecord.behavior_type}
                onChange={e => setBehaviorRecord({ ...behaviorRecord, behavior_type: e.target.value })}
                className="w-full px-3 py-2 border border-gray-800 bg-cyber-light text-foreground dark:text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">选择行为类型</option>
                <option value="view">查看</option>
                <option value="click">点击</option>
                <option value="create">创建</option>
                <option value="edit">编辑</option>
                <option value="delete">删除</option>
                <option value="search">搜索</option>
                <option value="share">分享</option>
                <option value="download">下载</option>
              </select>
            </div>

            <div className="space-y-2">
              <label htmlFor="behavior-target" className="block text-sm font-medium text-gray-300">目标</label>
              <input
                id="behavior-target"
                placeholder="输入行为目标"
                value={behaviorRecord.target}
                onChange={e => setBehaviorRecord({ ...behaviorRecord, target: e.target.value })}
                className="w-full px-3 py-2 border border-gray-800 bg-cyber-light text-foreground dark:text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-500"
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="behavior-duration" className="block text-sm font-medium text-gray-300">持续时间（秒）</label>
              <input
                id="behavior-duration"
                type="number"
                placeholder="输入持续时间"
                value={behaviorRecord.duration}
                onChange={e => setBehaviorRecord({ ...behaviorRecord, duration: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 border border-gray-800 bg-cyber-light text-foreground dark:text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-500"
              />
            </div>

            <div className="space-y-2 md:col-span-2">
              <label htmlFor="behavior-metadata" className="block text-sm font-medium text-gray-300">元数据（JSON格式）</label>
              <input
                id="behavior-metadata"
                placeholder="输入元数据，JSON格式"
                value={behaviorRecord.metadata}
                onChange={e => setBehaviorRecord({ ...behaviorRecord, metadata: e.target.value })}
                className="w-full px-3 py-2 border border-gray-800 bg-cyber-light text-foreground dark:text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm placeholder-gray-500"
              />
            </div>

            <div className="flex items-center justify-end space-x-2 md:col-span-2">
              <button
                onClick={handleSubmitRecord}
                disabled={recordBehaviorMutation.isPending}
                className="px-4 py-2 bg-blue-900 text-foreground dark:text-gray-200 hover:bg-blue-800 rounded-md transition-colors disabled:opacity-50 border border-blue-800"
              >
                {recordBehaviorMutation.isPending ?
(
                  <>
                    <span className="mr-2">⏳</span>
                    记录中...
                  </>
                ) :
(
                  '记录行为'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 用户画像 */}
        <div className="border rounded-lg p-6 bg-cyber-deep shadow-sm border-gray-800">
          <h2 className="text-xl font-semibold mb-4 text-foreground dark:text-gray-200">用户画像</h2>
          {isProfileLoading ?
(
            <div className="flex justify-center items-center py-8">
              <span className="text-xl text-foreground dark:text-gray-200">⏳</span>
              <span className="ml-2 text-gray-300">加载用户画像中...</span>
            </div>
          ) :
profileError ?
(
            <div className="p-4 bg-red-900/30 border border-red-800 rounded-md text-red-300">
              <h3 className="font-semibold text-red-800">加载失败</h3>
              <p className="text-red-700">{profileError.message}</p>
            </div>
          ) :
!userProfile?.success ?
(
            <div className="p-4 bg-red-900/30 border border-red-800 rounded-md text-red-300">
              <h3 className="font-semibold text-red-800">加载失败</h3>
              <p className="text-red-700">{userProfile?.error}</p>
            </div>
          ) :
(
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-400">用户ID</label>
                  <div className="font-semibold text-foreground dark:text-gray-200">{userProfile.data?.user_id}</div>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-400">创建时间</label>
                  <div className="font-semibold text-foreground dark:text-gray-200">
                    {userProfile.data?.created_at && new Date(userProfile.data.created_at).toLocaleString()}
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-400">最后更新</label>
                  <div className="font-semibold text-foreground dark:text-gray-200">
                    {userProfile.data?.last_updated && new Date(userProfile.data.last_updated).toLocaleString()}
                  </div>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-400 mb-2 block">行为统计</label>
                <div className="flex flex-wrap gap-2">
                  {userProfile.data?.behavior_counts && Object.entries(userProfile.data.behavior_counts).map(([type, count]) => (
                    <span key={type} className="px-2 py-1 bg-cyber-light text-gray-300 border border-gray-700 rounded-full text-xs">
                      {type}: {count}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-400 mb-2 block">活跃时间段</label>
                <div className="grid grid-cols-4 gap-2">
                  {userProfile.data?.active_hours && Object.entries(userProfile.data.active_hours).map(([hour, count]) => (
                    <div key={hour} className="text-center p-2 bg-cyber-light rounded border border-gray-700">
                      <div className="text-sm font-medium text-foreground dark:text-gray-200">{hour}:00</div>
                      <div className="text-xs text-gray-400">{count}次</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 习惯分析结果 */}
        <div className="border rounded-lg p-6 bg-cyber-deep shadow-sm border-gray-800">
          <h2 className="text-xl font-semibold mb-4 text-foreground dark:text-gray-200">习惯分析结果</h2>
          {isAnalysisLoading ?
(
            <div className="flex justify-center items-center py-8">
              <span className="text-xl text-foreground dark:text-gray-200">⏳</span>
              <span className="ml-2 text-gray-300">分析习惯数据中...</span>
            </div>
          ) :
analysisError ?
(
            <div className="p-4 bg-red-900/30 border border-red-800 rounded-md text-red-300">
              <h3 className="font-semibold text-red-800">分析失败</h3>
              <p className="text-red-700">{analysisError.message}</p>
            </div>
          ) :
!analysisResult?.success ?
(
            <div className="p-4 bg-red-900/30 border border-red-800 rounded-md text-red-300">
              <h3 className="font-semibold text-red-800">分析失败</h3>
              <p className="text-red-700">{analysisResult?.error}</p>
            </div>
          ) :
(
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-400">分析时间范围</label>
                  <div className="font-semibold text-foreground dark:text-gray-200">{analysisResult.data?.time_range}</div>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-400">行为总数</label>
                  <div className="font-semibold text-foreground dark:text-gray-200">{analysisResult.data?.behavior_count}</div>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-400 mb-2 block">行为分布</label>
                <div className="flex flex-wrap gap-2">
                  {analysisResult.data?.behavior_distribution && Object.entries(analysisResult.data.behavior_distribution).map(([type, count]) => (
                    <span key={type} className="px-2 py-1 bg-blue-900/50 text-blue-300 border border-blue-800 rounded-full text-xs">
                      {type}: {count}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-400 mb-2 block">活跃小时</label>
                <div className="flex flex-wrap gap-2">
                  {analysisResult.data?.active_hours.map(hour => (
                    <span key={hour} className="px-2 py-1 bg-blue-900/50 text-blue-300 border border-blue-800 rounded-full text-xs">
                      {hour}:00
                    </span>
                  ))}
                  {analysisResult.data?.active_hours.length === 0 && (
                    <span className="text-gray-400">无明显活跃时间段</span>
                  )}
                </div>
                <div className="text-sm text-gray-400 mt-2">
                  活跃时间段：{getActiveHoursText()}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-400 mb-2 block">设备偏好</label>
                <div className="flex flex-wrap gap-2">
                  {analysisResult.data?.device_preferences && Object.entries(analysisResult.data.device_preferences).map(([device, preference]) => (
                    <span key={device} className="px-2 py-1 bg-cyber-light text-gray-300 border border-gray-700 rounded-full text-xs">
                      {device}: {preference.toFixed(2)}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-400 mb-2 block">发现的模式</label>
                <div className="space-y-2">
                  {analysisResult.data?.patterns && analysisResult.data.patterns.length > 0 ?
(
                    analysisResult.data.patterns.map((pattern, index) => (
                      <div key={index} className="p-2 bg-cyber-light rounded text-sm text-foreground dark:text-gray-200 border border-gray-700">
                        {JSON.stringify(pattern)}
                      </div>
                    ))
                  ) :
(
                    <span className="text-gray-400">未发现明显模式</span>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 习惯建议 */}
      <div className="border rounded-lg p-6 bg-cyber-deep shadow-sm border-gray-800">
        <h2 className="text-xl font-semibold mb-4 text-foreground dark:text-gray-200">习惯建议</h2>
        {isRecommendationsLoading ?
(
          <div className="flex justify-center items-center py-8">
            <span className="text-xl text-foreground dark:text-gray-200">⏳</span>
            <span className="ml-2 text-gray-300">加载建议中...</span>
          </div>
        ) :
!recommendations?.success ?
(
          <div className="p-4 bg-red-900/30 border border-red-800 rounded-md text-red-300">
            <h3 className="font-semibold text-red-800">加载失败</h3>
            <p className="text-red-700">{recommendations?.error}</p>
          </div>
        ) :
recommendations.data?.recommendations.length === 0 ?
(
          <div className="p-4 bg-yellow-900/30 border border-yellow-800 rounded-md text-yellow-300">
            <h3 className="font-semibold text-yellow-800">无建议</h3>
            <p className="text-yellow-700">当前没有可用的习惯建议。</p>
          </div>
        ) :
(
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recommendations.data?.recommendations.map((recommendation, index) => (
              <div key={index} className="p-4 bg-cyber-light rounded border border-gray-700">
                <h3 className="font-medium mb-2 text-foreground dark:text-gray-200">建议 {index + 1}</h3>
                <p className="text-sm text-gray-300">{JSON.stringify(recommendation)}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default UserHabitAnalysis;
