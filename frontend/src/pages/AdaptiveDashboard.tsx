import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BarChart3, Cpu, Database, LineChart, Settings, Shield, TrendingUp } from 'lucide-react';
import { Industry } from '@/config/industries';
import { getAdaptiveModels, getModelDeployments, ModelGenerationTask, ModelDeployment, generateAdaptiveModel, getModelGenerationStatus } from '@/services/modelService';
import { useIndustry } from '@/contexts/IndustryContext';

const AdaptiveDashboard: React.FC = () => {
  // 使用行业上下文
  const { currentIndustry, setCurrentIndustry, industryList } = useIndustry();
  
  // 状态管理
  const [adaptiveModels, setAdaptiveModels] = useState<any[]>([]);
  const [modelDeployments, setModelDeployments] = useState<ModelDeployment[]>([]);
  const [generationTasks, setGenerationTasks] = useState<ModelGenerationTask[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [environment, setEnvironment] = useState('local');

  // 初始化数据
  useEffect(() => {
    loadAdaptiveModels();
    loadModelDeployments();
  }, [currentIndustry]);

  // 加载自适应模型
  const loadAdaptiveModels = async () => {
    try {
      const models = await getAdaptiveModels(currentIndustry);
      setAdaptiveModels(models);
    } catch (error) {
      console.error('加载自适应模型失败:', error);
    }
  };

  // 加载模型部署
  const loadModelDeployments = async () => {
    if (!selectedModel) return;
    try {
      const deployments = await getModelDeployments(selectedModel);
      setModelDeployments(deployments);
    } catch (error) {
      console.error('加载模型部署失败:', error);
    }
  };

  // 生成自适应模型
  const handleGenerateModel = async () => {
    setIsGenerating(true);
    try {
      const task = await generateAdaptiveModel(currentIndustry, {});
      setGenerationTasks([task, ...generationTasks]);
      
      // 定期检查任务状态
      const checkStatusInterval = setInterval(async () => {
        try {
          const status = await getModelGenerationStatus(task.id);
          setGenerationTasks(prev => prev.map(t => t.id === status.id ? status : t));
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(checkStatusInterval);
            loadAdaptiveModels();
          }
        } catch (error) {
          console.error('获取模型生成状态失败:', error);
          clearInterval(checkStatusInterval);
        }
      }, 2000);
    } catch (error) {
      console.error('生成自适应模型失败:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  // 部署模型 - 暂时注释，因为deployModel函数未定义
  const handleDeployModel = async (modelId: string) => {
    try {
      // const deployment = await deployModel(modelId, environment);
      // setGenerationTasks([deployment, ...generationTasks]);
      alert('部署功能暂未实现');
    } catch (error) {
      console.error('部署模型失败:', error);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* 页面标题 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">自适应AI平台</h1>
          <p className="text-gray-300">为各行各业自动生成和部署预测模型</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Select value={currentIndustry} onValueChange={setCurrentIndustry}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="选择行业" />
            </SelectTrigger>
            <SelectContent>
              {industryList.map(industry => (
                <SelectItem key={industry.value} value={industry.value}>
                  <div className="flex items-center gap-2">
                    <industry.icon className="w-4 h-4" />
                    <span>{industry.label}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button onClick={handleGenerateModel} disabled={isGenerating} className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            {isGenerating ? '生成中...' : '生成模型'}
          </Button>
        </div>
      </div>

      {/* 行业概览卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="glass-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">活跃模型</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{adaptiveModels.length}</div>
              <p className="text-sm text-gray-400">{currentIndustry}行业</p>
            </CardContent>
          </Card>

        <Card className="glass-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">部署模型</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{modelDeployments.length}</div>
            <p className="text-sm text-gray-400">已部署的模型</p>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">生成任务</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{generationTasks.length}</div>
            <p className="text-sm text-gray-400">进行中的生成任务</p>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">平均准确率</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">92%</div>
            <p className="text-sm text-gray-400">所有模型的平均准确率</p>
          </CardContent>
        </Card>
      </div>

      {/* 主内容区 */}
      <Tabs defaultValue="models" className="w-full">
        <TabsList className="mb-6">
          <TabsTrigger value="models">模型管理</TabsTrigger>
          <TabsTrigger value="deployments">部署管理</TabsTrigger>
          <TabsTrigger value="generation">生成任务</TabsTrigger>
        </TabsList>

        {/* 模型管理 */}
        <TabsContent value="models">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 模型列表 */}
            <div className="lg:col-span-2 space-y-4">
              {adaptiveModels.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {adaptiveModels.map(model => (
                    <motion.div
                      key={model.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <Card className="h-full glass-card hover:border-primary/50 transition-all duration-300">
                        <CardHeader>
                          <div className="flex justify-between items-start">
                            <div>
                              <CardTitle>{model.name}</CardTitle>
                              <CardDescription>类型: {model.type}</CardDescription>
                            </div>
                            <div className="text-sm px-2 py-0.5 rounded-full bg-primary/20 text-primary">
                              {model.status}
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            <div>
                              <div className="flex justify-between text-sm mb-1">
                                <span>准确率</span>
                                <span className="font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                              </div>
                              <Progress value={model.accuracy * 100} className="h-2" />
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                              <div className="flex items-center gap-2">
                                <Database className="w-4 h-4 text-gray-400" />
                                <span>数据量: {model.data_size || 'N/A'}</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <Cpu className="w-4 h-4 text-gray-400" />
                                <span>类型: {model.model_type || 'N/A'}</span>
                              </div>
                            </div>
                            <div className="flex justify-between">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setSelectedModel(model.id)}
                                className="flex items-center gap-2"
                              >
                                <Settings className="w-4 h-4" />
                                配置
                              </Button>
                              <Button
                                size="sm"
                                onClick={() => handleDeployModel(model.id)}
                                className="flex items-center gap-2"
                              >
                                <BarChart3 className="w-4 h-4" />
                                部署
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <Card className="h-full glass-card">
                  <CardContent className="h-full flex flex-col items-center justify-center py-10">
                    <Database className="w-12 h-12 text-gray-500 mb-4" />
                    <h3 className="text-lg font-medium mb-2">暂无模型</h3>
                    <p className="text-gray-400 text-center mb-4">
                    点击右上角"生成模型"按钮，为{industryList.find(i => i.value === currentIndustry)?.label}行业创建第一个模型
                  </p>
                    <Button onClick={handleGenerateModel} className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  生成模型
                </Button>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* 右侧详情面板 */}
            <div>
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>部署配置</CardTitle>
                  <CardDescription>配置模型部署环境</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">选择模型</label>
                    <Select value={selectedModel || ''} onValueChange={setSelectedModel}>
                      <SelectTrigger>
                        <SelectValue placeholder="选择要部署的模型" />
                      </SelectTrigger>
                      <SelectContent>
                        {adaptiveModels.map(model => (
                          <SelectItem key={model.id} value={model.id}>
                            {model.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">部署环境</label>
                    <Select value={environment} onValueChange={setEnvironment}>
                      <SelectTrigger>
                        <SelectValue placeholder="选择部署环境" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="local">本地</SelectItem>
                        <SelectItem value="edge">边缘设备</SelectItem>
                        <SelectItem value="cloud">云端</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">设备名称</label>
                    <Input placeholder="输入设备名称" />
                  </div>
                  <Button className="w-full">部署模型</Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* 部署管理 */}
        <TabsContent value="deployments">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>模型部署管理</CardTitle>
              <CardDescription>管理所有模型部署</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">模型名称</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">部署环境</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">状态</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">部署时间</th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelDeployments.length > 0 ? (
                      modelDeployments.map(deployment => (
                        <tr key={deployment.id} className="border-b border-gray-200 hover:bg-gray-800/50 transition-colors">
                          <td className="py-3 px-4">
                            <div className="font-medium">{adaptiveModels.find(m => m.id === deployment.modelId)?.name || deployment.modelId}</div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="text-sm">{deployment.environment}</div>
                          </td>
                          <td className="py-3 px-4">
                            <div className={`inline-flex items-center text-sm px-2 py-0.5 rounded-full ${deployment.status === 'deployed' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                              {deployment.status}
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="text-sm">{new Date(deployment.deploymentTime).toLocaleString()}</div>
                          </td>
                          <td className="py-3 px-4 text-right">
                            <Button variant="outline" size="sm">
                              查看
                            </Button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={5} className="py-10 text-center">
                          <Database className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                          <h3 className="text-lg font-medium mb-2">暂无部署</h3>
                          <p className="text-gray-400">请先生成模型，然后部署到目标环境</p>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 生成任务 */}
        <TabsContent value="generation">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>生成任务管理</CardTitle>
                  <CardDescription>管理模型生成任务</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {generationTasks.length > 0 ? (
                      generationTasks.map(task => (
                        <div key={task.id} className="border border-gray-200 rounded-lg p-4 hover:border-primary/50 transition-all duration-300">
                          <div className="flex justify-between items-start mb-2">
                            <div>
                              <div className="font-medium">任务 ID: {task.id}</div>
                              <div className="text-sm text-gray-400">行业: {task.industry}</div>
                            </div>
                            <div className={`text-sm px-2 py-0.5 rounded-full ${task.status === 'completed' ? 'bg-green-200 text-green-800' : task.status === 'failed' ? 'bg-red-200 text-red-800' : 'bg-yellow-200 text-yellow-800'}`}>
                              {task.status}
                            </div>
                          </div>
                          <div className="space-y-2">
                            <div>
                              <div className="flex justify-between text-sm mb-1">
                                <span>进度</span>
                                <span className="font-medium">{task.progress}%</span>
                              </div>
                              <Progress value={task.progress} className="h-2" />
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                              <div className="flex items-center gap-2">
                                <LineChart className="w-4 h-4 text-gray-400" />
                                <span>开始时间: {new Date(task.startTime).toLocaleString()}</span>
                              </div>
                              {task.endTime && (
                                <div className="flex items-center gap-2">
                                  <LineChart className="w-4 h-4 text-gray-400" />
                                  <span>结束时间: {new Date(task.endTime).toLocaleString()}</span>
                                </div>
                              )}
                            </div>
                            {task.error && (
                              <div className="text-sm text-red-500">
                                错误: {task.error}
                              </div>
                            )}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-center py-10">
                        <TrendingUp className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                        <h3 className="text-lg font-medium mb-2">暂无生成任务</h3>
                        <p className="text-gray-400">点击右上角"生成模型"按钮，开始第一个生成任务</p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            <div>
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>生成模型</CardTitle>
                  <CardDescription>配置并生成新模型</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">模型名称</label>
                    <Input placeholder="输入模型名称" />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">模型类型</label>
                    <Select defaultValue="regression">
                      <SelectTrigger>
                        <SelectValue placeholder="选择模型类型" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="regression">回归</SelectItem>
                        <SelectItem value="classification">分类</SelectItem>
                        <SelectItem value="clustering">聚类</SelectItem>
                        <SelectItem value="forecasting">预测</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">优化目标</label>
                    <Select defaultValue="accuracy">
                      <SelectTrigger>
                        <SelectValue placeholder="选择优化目标" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="accuracy">准确率</SelectItem>
                        <SelectItem value="speed">速度</SelectItem>
                        <SelectItem value="memory">内存</SelectItem>
                        <SelectItem value="custom">自定义</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button className="w-full" onClick={handleGenerateModel} disabled={isGenerating}>
                    {isGenerating ? '生成中...' : '生成模型'}
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AdaptiveDashboard;
