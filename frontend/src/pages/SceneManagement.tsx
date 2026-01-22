import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { Scene, ApiResponse } from '@/services/api';
import { apiClient } from '@/services/api';

const SceneManagement: React.FC = () => {
  const queryClient = useQueryClient();

  // 状态管理
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedScene, setSelectedScene] = useState<Scene | null>(null);
  const [scene, setScene] = useState<{
    name: string;
    description: string;
    status: string;
    actions: any[];
    conditions: any[];
  }>({
    name: '',
    description: '',
    status: 'inactive',
    actions: [],
    conditions: [],
  });

  // 获取场景列表
  const {
    data: scenes,
    isLoading: isScenesLoading,
    error: scenesError,
    refetch: refetchScenes,
  } = useQuery<ApiResponse<Scene[]>>({
    queryKey: ['scenes'],
    queryFn: async () => apiClient.getScenes(),
  });

  // 触发场景
  const triggerSceneMutation = useMutation<ApiResponse<any>, Error, string>({
    mutationFn: async scene_id => apiClient.triggerScene(scene_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenes'] });
      alert('场景已触发');
    },
    onError: error => {
      alert(`触发场景失败: ${error.message}`);
    },
  });

  // 创建场景
  const createSceneMutation = useMutation<ApiResponse<Scene>, Error, typeof scene>({
    mutationFn: async sceneData => apiClient.createScene(sceneData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenes'] });
      setShowCreateForm(false);
      resetForm();
      alert('场景已创建');
    },
    onError: error => {
      alert(`创建场景失败: ${error.message}`);
    },
  });

  // 更新场景
  const updateSceneMutation = useMutation<ApiResponse<Scene>, Error, { scene_id: string, sceneData: typeof scene }>({
    mutationFn: async ({ scene_id, sceneData }) => apiClient.updateScene(scene_id, sceneData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenes'] });
      setSelectedScene(null);
      resetForm();
      alert('场景已更新');
    },
    onError: error => {
      alert(`更新场景失败: ${error.message}`);
    },
  });

  // 删除场景
  const deleteSceneMutation = useMutation<ApiResponse<any>, Error, string>({
    mutationFn: async scene_id => apiClient.deleteScene(scene_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenes'] });
      alert('场景已删除');
    },
    onError: error => {
      alert(`删除场景失败: ${error.message}`);
    },
  });

  // 重置表单
  const resetForm = () => {
    setScene({
      name: '',
      description: '',
      status: 'inactive',
      actions: [],
      conditions: [],
    });
  };

  // 处理场景提交
  const handleSubmitScene = () => {
    if (!scene.name) {
      alert('场景名称不能为空');
      return;
    }

    if (selectedScene) {
      updateSceneMutation.mutate({ scene_id: selectedScene.scene_id, sceneData: scene });
    } else {
      createSceneMutation.mutate(scene);
    }
  };

  // 处理场景编辑
  const handleEditScene = (sceneToEdit: Scene) => {
    setSelectedScene(sceneToEdit);
    setScene({
      name: sceneToEdit.name,
      description: sceneToEdit.description,
      status: sceneToEdit.status,
      actions: sceneToEdit.actions,
      conditions: sceneToEdit.conditions,
    });
    setShowCreateForm(true);
  };

  // 处理场景删除
  const handleDeleteScene = (scene_id: string) => {
    if (confirm('确定要删除这个场景吗？')) {
      deleteSceneMutation.mutate(scene_id);
    }
  };

  // 处理场景触发
  const handleTriggerScene = (scene_id: string) => {
    triggerSceneMutation.mutate(scene_id);
  };

  return (
    <div className="space-y-6 p-4 md:p-6 bg-background text-foreground">
        <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold text-foreground">场景管理</h1>
        <div className="flex gap-2">
          <button
            onClick={async () => refetchScenes()}
            className="bg-blue-600 hover:bg-blue-700 text-white dark:text-gray-200 font-bold py-2 px-4 rounded border border-blue-700"
          >
            刷新场景列表
          </button>
          <button
            onClick={() => {
              setSelectedScene(null);
              resetForm();
              setShowCreateForm(!showCreateForm);
            }}
            className="bg-blue-600 hover:bg-blue-700 text-white dark:text-gray-200 font-bold py-2 px-4 rounded border border-blue-700"
          >
            {showCreateForm ? '取消' : '创建场景'}
          </button>
        </div>
      </div>

      {/* 创建/编辑场景表单 */}
      {showCreateForm && (
        <div className="bg-card p-6 rounded-lg shadow-md border">
          <h2 className="text-xl font-semibold mb-4 text-foreground">
            {selectedScene ? '编辑场景' : '创建新场景'}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label htmlFor="scene-name" className="block text-sm font-medium text-foreground">场景名称</label>
              <input
                id="scene-name"
                placeholder="输入场景名称"
                value={scene.name}
                onChange={e => setScene({ ...scene, name: e.target.value })}
                className="border bg-card text-foreground rounded-md p-2 w-full placeholder:text-muted-foreground"
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="scene-status" className="block text-sm font-medium text-foreground">场景状态</label>
              <select
                value={scene.status}
                onChange={e => setScene({ ...scene, status: e.target.value })}
                className="border bg-card text-foreground rounded-md p-2 w-full"
              >
                <option value="active">激活</option>
                <option value="inactive">停用</option>
              </select>
            </div>

            <div className="space-y-2 md:col-span-2">
              <label htmlFor="scene-description" className="block text-sm font-medium text-foreground">场景描述</label>
              <textarea
                id="scene-description"
                placeholder="输入场景描述"
                value={scene.description}
                onChange={e => setScene({ ...scene, description: e.target.value })}
                className="border bg-card text-foreground rounded-md p-2 w-full min-h-[100px] placeholder:text-muted-foreground"
              />
            </div>

            <div className="space-y-2 md:col-span-2">
              <label htmlFor="scene-actions" className="block text-sm font-medium text-foreground">执行动作 (JSON格式)</label>
              <textarea
                id="scene-actions"
                placeholder='例如: [{"type": "turn_on", "device_id": 1}, {"type": "adjust_brightness", "device_id": 1, "value": 50}]'
                value={JSON.stringify(scene.actions, null, 2)}
                onChange={e => {
                  try {
                    setScene({ ...scene, actions: JSON.parse(e.target.value) });
                  } catch (error) {
                    // 忽略无效的JSON，不更新状态
                  }
                }}
                className="border bg-card text-foreground rounded-md p-2 w-full min-h-[150px] font-mono text-sm placeholder:text-muted-foreground"
              />
            </div>

            <div className="space-y-2 md:col-span-2">
              <label htmlFor="scene-conditions" className="block text-sm font-medium text-foreground">触发条件 (JSON格式)</label>
              <textarea
                id="scene-conditions"
                placeholder='例如: [{"type": "time", "value": "18:00"}, {"type": "device_status", "device_id": 2, "status": "online"}]'
                value={JSON.stringify(scene.conditions, null, 2)}
                onChange={e => {
                  try {
                    setScene({ ...scene, conditions: JSON.parse(e.target.value) });
                  } catch (error) {
                    // 忽略无效的JSON，不更新状态
                  }
                }}
                className="border bg-card text-foreground rounded-md p-2 w-full min-h-[150px] font-mono text-sm placeholder:text-muted-foreground"
              />
            </div>

            <div className="flex items-center justify-end space-x-2 md:col-span-2">
              <button
                onClick={handleSubmitScene}
                disabled={
                  createSceneMutation.isPending ||
                  updateSceneMutation.isPending
                }
                className="bg-blue-600 hover:bg-blue-700 text-white dark:text-gray-200 font-bold py-2 px-4 rounded disabled:opacity-50 border border-blue-700"
              >
                {(createSceneMutation.isPending || updateSceneMutation.isPending) ? '处理中...' : (selectedScene ? '更新场景' : '创建场景')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 场景列表 */}
      <div className="bg-card p-6 rounded-lg shadow-md border">
        <h2 className="text-xl font-semibold mb-4 text-foreground">场景列表</h2>
        {isScenesLoading ?
(
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <span className="text-muted-foreground">加载场景列表中...</span>
          </div>
        ) :
scenesError ?
(
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded">
            <strong className="font-bold text-red-800 dark:text-red-200">加载失败：</strong>
            <span>{scenesError.message}</span>
          </div>
        ) :
!scenes?.success ?
(
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded">
            <strong className="font-bold text-red-800 dark:text-red-200">加载失败：</strong>
            <span>{scenes?.error || '未知错误'}</span>
          </div>
        ) :
scenes.data?.length === 0 ?
(
          <div className="bg-yellow-100 dark:bg-yellow-900/30 border border-yellow-400 dark:border-yellow-800 text-yellow-700 dark:text-yellow-300 px-4 py-3 rounded">
            <strong className="font-bold text-yellow-800 dark:text-yellow-200">无场景：</strong>
            <span>当前没有可用的场景，请先创建场景。</span>
          </div>
        ) :
(
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-muted border">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-foreground uppercase tracking-wider">场景ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-foreground uppercase tracking-wider">名称</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">描述</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">状态</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">创建时间</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">上次触发</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">操作</th>
                </tr>
              </thead>
              <tbody className="bg-card divide-y divide-gray-200">
                {scenes.data?.map(scene => (
                  <tr key={scene.scene_id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-foreground">{scene.scene_id}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-foreground">{scene.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm max-w-[200px] truncate text-muted-foreground">{scene.description}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${scene.status === 'active' ? 'bg-green-100 text-green-800 border border-green-300' : 'bg-muted text-muted-foreground border'}`}>
                        {scene.status === 'active' ? '激活' : '停用'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-muted-foreground">
                      {new Date(scene.created_at).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-muted-foreground">
                      {scene.last_triggered ? new Date(scene.last_triggered).toLocaleString() : '从未触发'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleTriggerScene(scene.scene_id)}
                          disabled={triggerSceneMutation.isPending}
                          className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-1 px-2 rounded text-xs disabled:opacity-50 border border-gray-300"
                        >
                          {triggerSceneMutation.isPending ? '触发中...' : '触发'}
                        </button>
                        <button
                          onClick={() => handleEditScene(scene)}
                          className="bg-blue-100 hover:bg-blue-200 text-blue-800 font-bold py-1 px-2 rounded text-xs border border-blue-300"
                        >
                          编辑
                        </button>
                        <button
                          onClick={() => handleDeleteScene(scene.scene_id)}
                          disabled={deleteSceneMutation.isPending}
                          className="bg-red-100 hover:bg-red-200 text-red-800 font-bold py-1 px-2 rounded text-xs disabled:opacity-50 border border-red-300"
                        >
                          {deleteSceneMutation.isPending ? '删除中...' : '删除'}
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

export default SceneManagement;
