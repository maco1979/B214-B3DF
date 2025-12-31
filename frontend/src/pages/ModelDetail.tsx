import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ArrowLeft, Edit, Trash2, Play, Pause, Download, Share2, ChevronDown } from 'lucide-react'
import { useModelQuery, useModelVersionsQuery } from '@/hooks/useModelsQuery'
import { apiClient, TrainingStatusResponse, Model } from '@/services/api'

interface TrainingHistoryItem {
  id: string
  start_time: string
  end_time?: string
  status: string
  accuracy?: number
  loss?: number
}

export function ModelDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const modelId = id ? decodeURIComponent(id) : ''
  const { data: model, isLoading: loading, refetch } = useModelQuery(modelId)
  
  const [isTraining, setIsTraining] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatusResponse | null>(null)
  const [trainingError, setTrainingError] = useState<string | null>(null)
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null)
  
  // 使用useModelVersions hook获取模型版本
  const { data: versions = [], isLoading: isLoadingVersions, refetch: refetchVersions } = useModelVersionsQuery(modelId)
  
  // 版本控制相关状态
  const [selectedVersion, setSelectedVersion] = useState<string>('')
  const [isVersionDropdownOpen, setIsVersionDropdownOpen] = useState(false)
  
  // 启动模型训练
  const startTraining = async () => {
    if (!model || !modelId) return
    
    try {
      setIsTraining(true)
      setTrainingError(null)
      
      const response = await apiClient.startModelTraining(modelId, {
        // 这里可以根据实际需求添加训练参数
        dataset: 'default-dataset',
        epochs: 100,
        batch_size: 32
      })
      
      if (response.success && response.data) {
        const { task_id } = response.data
        setTrainingStatus(response.data)
        
        // 开始轮询训练状态
        const interval = setInterval(() => {
          checkTrainingStatus(task_id)
        }, 2000)
        setPollingInterval(interval)
      } else {
        setTrainingError(response.error || '训练启动失败')
      }
    } catch (err) {
      setTrainingError('训练启动失败: ' + (err as Error).message)
    } finally {
      setIsTraining(false)
    }
  }
  
  // 检查训练状态
  const checkTrainingStatus = async (taskId: string) => {
    try {
      const response = await apiClient.getModelTrainingStatus(taskId)
      
      if (response.success && response.data) {
        setTrainingStatus(response.data)
        
        // 如果训练完成或失败，停止轮询
        if (response.data.status === 'completed' || response.data.status === 'failed') {
          if (pollingInterval) {
            clearInterval(pollingInterval)
            setPollingInterval(null)
          }
          // 刷新模型数据
          refetch()
        }
      }
    } catch (err) {
      console.error('获取训练状态失败:', err)
    }
  }

  // 切换版本
  const handleVersionChange = async (version: string) => {
    setSelectedVersion(version)
    setIsVersionDropdownOpen(false)
    
    // 找到对应版本的模型ID
    const versionModel = versions.find(v => v.version === version)
    if (versionModel) {
      // 导航到对应版本的模型详情页
      navigate(`/models/${encodeURIComponent(versionModel.id)}`)
    }
  }

  // 创建新版本
  const handleCreateVersion = async () => {
    if (!model || !modelId) return
    
    try {
      const response = await apiClient.createModelVersion(modelId, {
        name: model.name,
        // 可以根据实际需求添加更多版本创建参数
      })
      
      if (response.success && response.data) {
        // 创建成功后刷新版本列表并导航到新版本
        await refetchVersions()
        navigate(`/models/${response.data.id}`)
      }
    } catch (err) {
      console.error('创建新版本失败:', err)
    }
  }

  // 设置默认版本
  useEffect(() => {
    if (model && versions && versions.length > 0) {
      setSelectedVersion(model.version)
    }
  }, [model, versions])

  // 组件卸载时清除轮询
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval)
      }
    }
  }, [pollingInterval])
  
  // 模拟训练历史数据
  const trainingHistory: TrainingHistoryItem[] = [
    {
      id: '1',
      start_time: '2025-12-24T10:00:00Z',
      end_time: '2025-12-24T12:30:00Z',
      status: 'completed',
      accuracy: 0.982,
      loss: 0.056
    },
    {
      id: '2',
      start_time: '2025-12-23T15:45:00Z',
      end_time: '2025-12-23T18:20:00Z',
      status: 'completed',
      accuracy: 0.975,
      loss: 0.068
    },
    {
      id: '3',
      start_time: '2025-12-22T09:15:00Z',
      end_time: '2025-12-22T11:45:00Z',
      status: 'error',
      accuracy: 0.921,
      loss: 0.189
    }
  ]

  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <div className="text-tech-primary text-xl">加载中...</div>
      </div>
    )
  }

  if (!model) {
    return (
      <div className="flex flex-col justify-center items-center h-96 space-y-4">
        <div className="text-red-400 text-xl">加载失败</div>
        <Button variant="tech" onClick={() => refetch()}>重试</Button>
      </div>
    )
  }

  // 格式化大小显示
  const formatSize = (bytes?: number): string => {
    if (!bytes) return '--'
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
  }

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex items-center space-x-4">
          <Button variant="outline" size="sm" onClick={() => navigate('/models')}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            返回列表
          </Button>
          <div>
            <h1 className="text-3xl font-bold gradient-text mb-2">{model.name}</h1>
            <p className="text-gray-300">模型详情 - 版本 {model.version}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {/* 版本选择下拉菜单 */}
          <div className="relative">
            <Button 
              variant="outline" 
              size="sm" 
              className="flex items-center space-x-1"
              onClick={() => setIsVersionDropdownOpen(!isVersionDropdownOpen)}
            >
              <span>选择版本</span>
              <ChevronDown className="w-4 h-4" />
            </Button>
            
            {isVersionDropdownOpen && (
              <div className="absolute mt-2 w-48 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-10">
                {isLoadingVersions ? (
                  <div className="p-2 text-gray-400">加载中...</div>
                ) : versions.length > 0 ? (
                  versions.map((version) => (
                    <button
                      key={version.id}
                      className={`w-full text-left px-4 py-2 text-sm ${version.version === selectedVersion ? 'bg-gray-700 text-white' : 'text-gray-300 hover:bg-gray-700'}`}
                      onClick={() => handleVersionChange(version.version)}
                    >
                      版本 {version.version} - {version.status}
                    </button>
                  ))
                ) : (
                  <div className="p-2 text-gray-400">暂无版本历史</div>
                )}
              </div>
            )}
          </div>
          
          {/* 创建新版本按钮 */}
          <Button 
            variant="outline" 
            className="flex items-center space-x-2"
            onClick={handleCreateVersion}
          >
            <Edit className="w-4 h-4" />
            <span>创建版本</span>
          </Button>
          
          <Button 
            variant="tech" 
            className="flex items-center space-x-2"
            onClick={startTraining}
            disabled={isTraining || model?.status === 'training'}
          >
            <Play className="w-4 h-4" />
            <span>{isTraining ? '启动中...' : '开始训练'}</span>
          </Button>
          <Button variant="outline" className="flex items-center space-x-2" disabled={!isTraining && model.status !== 'training'}>
            <Pause className="w-4 h-4" />
            <span>暂停</span>
          </Button>
          <Button variant="outline" className="flex items-center space-x-2">
            <Download className="w-4 h-4" />
            <span>下载</span>
          </Button>
          <Button variant="outline" className="flex items-center space-x-2">
            <Share2 className="w-4 h-4" />
            <span>分享</span>
          </Button>
          <Button variant="outline" className="flex items-center space-x-2 text-red-400">
            <Trash2 className="w-4 h-4" />
            <span>删除</span>
          </Button>
        </div>
      </div>

      {/* 模型基本信息 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="glass-effect lg:col-span-2">
          <CardHeader>
            <CardTitle>模型信息</CardTitle>
            <CardDescription>基本信息和配置</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">模型ID</h3>
                  <p className="text-white">{model.id}</p>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">创建时间</h3>
                  <p className="text-white">{new Date(model.created_at).toLocaleString()}</p>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">最后更新</h3>
                  <p className="text-white">{new Date(model.updated_at).toLocaleString()}</p>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">模型状态</h3>
                  <p className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${model.status === 'ready' ? 'bg-green-500 text-white' : model.status === 'training' ? 'bg-yellow-500 text-white' : 'bg-red-500 text-white'}`}>
                    {model.status === 'ready' ? '运行中' : model.status === 'training' ? '训练中' : model.status === 'deployed' ? '已部署' : '错误'}
                  </p>
                </div>
              </div>
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">模型大小</h3>
                  <p className="text-white">{formatSize(model.size)}</p>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">准确率</h3>
                  <p className="text-white">{model.accuracy ? `${(model.accuracy * 100).toFixed(2)}%` : '--'}</p>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">模型描述</h3>
                  <p className="text-white">{model.description}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 模型性能指标 */}
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle>性能指标</CardTitle>
            <CardDescription>最新训练结果</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium text-gray-400">准确率</span>
                  <span className="text-sm font-medium text-white">{model.accuracy ? `${(model.accuracy * 100).toFixed(2)}%` : '--'}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2.5">
                  <div 
                    className="bg-gradient-to-r from-green-400 to-blue-500 h-2.5 rounded-full" 
                    style={{ width: `${model.accuracy ? model.accuracy * 100 : 0}%` }}
                  ></div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h3 className="text-xs text-gray-500 mb-1">精确率</h3>
                  <p className="text-xl font-bold text-white">98.5%</p>
                </div>
                <div>
                  <h3 className="text-xs text-gray-500 mb-1">召回率</h3>
                  <p className="text-xl font-bold text-white">97.8%</p>
                </div>
                <div>
                  <h3 className="text-xs text-gray-500 mb-1">F1分数</h3>
                  <p className="text-xl font-bold text-white">98.1%</p>
                </div>
                <div>
                  <h3 className="text-xs text-gray-500 mb-1">训练时间</h3>
                  <p className="text-xl font-bold text-white">2.5h</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 训练进度展示 */}
      {(trainingStatus || trainingError) && (
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle>训练状态</CardTitle>
            <CardDescription>实时训练进度</CardDescription>
          </CardHeader>
          <CardContent>
            {trainingError ? (
              <div className="text-red-400">
                <p>训练失败: {trainingError}</p>
              </div>
            ) : trainingStatus ? (
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-2">训练进度</h3>
                  <div className="w-full bg-gray-700 rounded-full h-4">
                    <div 
                      className="bg-gradient-to-r from-yellow-400 to-orange-500 h-4 rounded-full transition-all duration-300 ease-in-out"
                      style={{ width: `${trainingStatus.progress}%` }}
                    ></div>
                  </div>
                  <div className="flex justify-between mt-1 text-sm">
                    <span className="text-gray-400">当前阶段</span>
                    <span className="text-white font-medium">{trainingStatus.stage}</span>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500 mb-1">训练任务ID</p>
                    <p className="text-white font-mono text-sm truncate">{trainingStatus.task_id}</p>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500 mb-1">当前步骤</p>
                    <p className="text-white">{trainingStatus.current_step} / {trainingStatus.total_steps}</p>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500 mb-1">训练状态</p>
                    <p className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${trainingStatus.status === 'completed' ? 'bg-green-500 text-white' : trainingStatus.status === 'training' ? 'bg-yellow-500 text-white' : 'bg-red-500 text-white'}`}>
                      {trainingStatus.status === 'completed' ? '已完成' : trainingStatus.status === 'training' ? '训练中' : '失败'}
                    </p>
                  </div>
                </div>
                
                {trainingStatus.metrics && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-2">训练指标</h3>
                    <div className="grid grid-cols-2 gap-4">
                      {Object.entries(trainingStatus.metrics).map(([key, value]) => (
                        <div key={key} className="bg-gray-800/50 rounded-lg p-3">
                          <p className="text-xs text-gray-500 mb-1">{key}</p>
                          <p className="text-white">{value.toFixed(4)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : null}
          </CardContent>
        </Card>
      )}

      {/* 模型训练历史 */}
      <Card className="glass-effect">
        <CardHeader>
          <CardTitle>训练历史</CardTitle>
          <CardDescription>模型训练记录</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">训练ID</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">开始时间</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">结束时间</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">状态</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">准确率</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">损失值</th>
                </tr>
              </thead>
              <tbody>
                {trainingHistory.map((item) => (
                  <tr key={item.id} className="border-b border-gray-700 hover:bg-gray-800/50">
                    <td className="py-3 px-4 text-sm text-white">{item.id}</td>
                    <td className="py-3 px-4 text-sm text-white">{new Date(item.start_time).toLocaleString()}</td>
                    <td className="py-3 px-4 text-sm text-white">{item.end_time ? new Date(item.end_time).toLocaleString() : '-'}</td>
                    <td className="py-3 px-4">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${item.status === 'completed' ? 'bg-green-500 text-white' : item.status === 'running' ? 'bg-yellow-500 text-white' : 'bg-red-500 text-white'}`}>
                        {item.status === 'completed' ? '完成' : item.status === 'running' ? '运行中' : '失败'}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm text-white">{item.accuracy ? `${(item.accuracy * 100).toFixed(2)}%` : '-'}</td>
                    <td className="py-3 px-4 text-sm text-white">{item.loss ? item.loss.toFixed(4) : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* 模型版本历史 */}
      <Card className="glass-effect">
        <CardHeader>
          <CardTitle>版本历史</CardTitle>
          <CardDescription>所有模型版本记录</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">版本号</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">模型ID</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">创建时间</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">状态</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">准确率</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">大小</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">操作</th>
                </tr>
              </thead>
              <tbody>
                {isLoadingVersions ? (
                  <tr>
                    <td colSpan={7} className="py-6 text-center text-gray-400">加载中...</td>
                  </tr>
                ) : versions.length > 0 ? (
                  versions.map((version) => (
                    <tr key={version.id} className="border-b border-gray-700 hover:bg-gray-800/50">
                      <td className="py-3 px-4 text-sm text-white">{version.version}</td>
                      <td className="py-3 px-4 text-sm text-white truncate max-w-xs">{version.id}</td>
                      <td className="py-3 px-4 text-sm text-white">{new Date(version.created_at).toLocaleString()}</td>
                      <td className="py-3 px-4">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${version.status === 'ready' ? 'bg-green-500 text-white' : version.status === 'training' ? 'bg-yellow-500 text-white' : 'bg-red-500 text-white'}`}>
                          {version.status === 'ready' ? '运行中' : version.status === 'training' ? '训练中' : version.status === 'deployed' ? '已部署' : '错误'}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-sm text-white">{version.accuracy ? `${(version.accuracy * 100).toFixed(2)}%` : '-'}</td>
                      <td className="py-3 px-4 text-sm text-white">{formatSize(version.size)}</td>
                      <td className="py-3 px-4">
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="px-3"
                          onClick={() => handleVersionChange(version.version)}
                        >
                          查看详情
                        </Button>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7} className="py-6 text-center text-gray-400">暂无版本历史</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* 模型配置 */}
      <Card className="glass-effect">
        <CardHeader>
          <CardTitle>模型配置</CardTitle>
          <CardDescription>超参数和技术规格</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-2">超参数</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">学习率</p>
                  <p className="text-white">0.001</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">批次大小</p>
                  <p className="text-white">32</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">迭代次数</p>
                  <p className="text-white">100</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">激活函数</p>
                  <p className="text-white">ReLU</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">优化器</p>
                  <p className="text-white">Adam</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">正则化</p>
                  <p className="text-white">L2 (0.01)</p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}