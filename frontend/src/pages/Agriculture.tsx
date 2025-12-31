import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { apiClient } from '../services/api';


interface CropConfig {
  crop_name: string;
  growth_stages: Array<{
    stage_name: string;
    duration_days: number;
    optimal_temperature: [number, number];
    optimal_humidity: [number, number];
    light_hours: number;
  }>;
  target_yield: string;
  quality_metrics: Record<string, number>;
}

interface LightRecipe {
  uv_380nm: number;
  far_red_720nm: number;
  white_light: number;
  red_660nm: number;
  white_red_ratio: number;
}

interface GrowthPrediction {
  growth_rate: number;
  health_score: number;
  yield_potential: number;
}

const AgriculturePage: React.FC = () => {
  const [cropConfigs, setCropConfigs] = useState<Record<string, CropConfig>>({});
  const [selectedCrop, setSelectedCrop] = useState('番茄');
  const [currentDay, setCurrentDay] = useState(1);
  const [targetObjective, setTargetObjective] = useState('最大化产量');
  const [environment, setEnvironment] = useState({
    temperature: 25,
    humidity: 60,
    co2: 400
  });
  const [lightRecipe, setLightRecipe] = useState<LightRecipe | null>(null);
  const [growthPrediction, setGrowthPrediction] = useState<GrowthPrediction | null>(null);
  const [plantingPlan, setPlantingPlan] = useState<any>(null);

  useEffect(() => {
    loadCropConfigs();
  }, []);

  const [loadingCrops, setLoadingCrops] = useState(false);
  const [cropError, setCropError] = useState<string | null>(null);

  const loadCropConfigs = async () => {
    setLoadingCrops(true);
    setCropError(null);
    try {
      const response = await apiClient.getCropConfigs();
      if (response.success && Object.keys(response.data).length > 0) {
        setCropConfigs(response.data);
      } else {
        // 使用默认作物配置
        const defaultCrops: Record<string, CropConfig> = {
          '番茄': {
            crop_name: '番茄',
            growth_stages: [
              { stage_name: '发芽期', duration_days: 7, optimal_temperature: [20, 25], optimal_humidity: [70, 80], light_hours: 12 },
              { stage_name: '幼苗期', duration_days: 21, optimal_temperature: [18, 22], optimal_humidity: [60, 70], light_hours: 14 },
              { stage_name: '生长期', duration_days: 28, optimal_temperature: [20, 25], optimal_humidity: [65, 75], light_hours: 16 },
              { stage_name: '结果期', duration_days: 35, optimal_temperature: [22, 28], optimal_humidity: [70, 80], light_hours: 18 }
            ],
            target_yield: '5kg/m²',
            quality_metrics: { '甜度': 8.5, '硬度': 7.2, '色泽': 9.0 }
          },
          '黄瓜': {
            crop_name: '黄瓜',
            growth_stages: [
              { stage_name: '发芽期', duration_days: 5, optimal_temperature: [25, 30], optimal_humidity: [75, 85], light_hours: 12 },
              { stage_name: '幼苗期', duration_days: 15, optimal_temperature: [20, 25], optimal_humidity: [65, 75], light_hours: 14 },
              { stage_name: '生长期', duration_days: 20, optimal_temperature: [25, 30], optimal_humidity: [70, 80], light_hours: 16 },
              { stage_name: '结果期', duration_days: 30, optimal_temperature: [25, 32], optimal_humidity: [75, 85], light_hours: 18 }
            ],
            target_yield: '8kg/m²',
            quality_metrics: { '长度': 15, '直径': 3, '色泽': 8.5 }
          }
        };
        setCropConfigs(defaultCrops);
        setCropError('使用默认作物数据');
      }
    } catch (error) {
      console.error('加载作物配置失败:', error);
      // 使用默认作物配置
      const defaultCrops: Record<string, CropConfig> = {
        '番茄': {
          crop_name: '番茄',
          growth_stages: [
            { stage_name: '发芽期', duration_days: 7, optimal_temperature: [20, 25], optimal_humidity: [70, 80], light_hours: 12 },
            { stage_name: '幼苗期', duration_days: 21, optimal_temperature: [18, 22], optimal_humidity: [60, 70], light_hours: 14 },
            { stage_name: '生长期', duration_days: 28, optimal_temperature: [20, 25], optimal_humidity: [65, 75], light_hours: 16 },
            { stage_name: '结果期', duration_days: 35, optimal_temperature: [22, 28], optimal_humidity: [70, 80], light_hours: 18 }
          ],
          target_yield: '5kg/m²',
          quality_metrics: { '甜度': 8.5, '硬度': 7.2, '色泽': 9.0 }
        },
        '黄瓜': {
          crop_name: '黄瓜',
          growth_stages: [
            { stage_name: '发芽期', duration_days: 5, optimal_temperature: [25, 30], optimal_humidity: [75, 85], light_hours: 12 },
            { stage_name: '幼苗期', duration_days: 15, optimal_temperature: [20, 25], optimal_humidity: [65, 75], light_hours: 14 },
            { stage_name: '生长期', duration_days: 20, optimal_temperature: [25, 30], optimal_humidity: [70, 80], light_hours: 16 },
            { stage_name: '结果期', duration_days: 30, optimal_temperature: [25, 32], optimal_humidity: [75, 85], light_hours: 18 }
          ],
          target_yield: '8kg/m²',
          quality_metrics: { '长度': 15, '直径': 3, '色泽': 8.5 }
        }
      };
      setCropConfigs(defaultCrops);
      setCropError('加载失败，使用默认作物数据');
    } finally {
      setLoadingCrops(false);
    }
  };

  const generateLightRecipe = async () => {
    try {
      const response = await apiClient.generateLightRecipe({
        crop_type: selectedCrop,
        current_day: currentDay,
        target_objective: targetObjective,
        environment: environment
      });
      
      if (response.success) {
        setLightRecipe(response.data.recipe);
      }
    } catch (error) {
      console.error('生成光配方失败:', error);
    }
  };

  const predictGrowth = async () => {
    try {
      // 模拟光谱数据
      const spectrumData = Array.from({length: 800}, (_, i) => 
        Math.random() * 0.1 + Math.exp(-Math.pow(i - 380, 2) / 1000) * 0.5
      );

      const response = await apiClient.predictGrowth({
        crop_type: selectedCrop,
        current_day: currentDay,
        environmental_data: environment,
        spectrum_data: spectrumData
      });
      
      if (response.success) {
        setGrowthPrediction(response.data);
      }
    } catch (error) {
      console.error('预测生长状态失败:', error);
    }
  };

  const createPlantingPlan = async () => {
    try {
      const response = await apiClient.createPlantingPlan({
        crop_type: selectedCrop,
        target_yield: targetObjective,
        start_date: new Date().toISOString().split('T')[0],
        expected_harvest_date: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
      });
      
      if (response.success) {
        setPlantingPlan(response.data);
      }
    } catch (error) {
      console.error('创建种植计划失败:', error);
    }
  };

  const contributeData = async () => {
    try {
      const response = await apiClient.contributeAgricultureData({
        user_id: 'user_001',
        crop_type: selectedCrop,
        growth_data: {
          day: currentDay,
          environment: environment,
          recipe: lightRecipe
        }
      });
      
      if (response.success) {
        alert(`数据贡献成功！获得 ${response.data.photon_points_earned} 光子积分`);
      }
    } catch (error) {
      console.error('数据贡献失败:', error);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：作物选择和种植规划 */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>作物选择与规划</CardTitle>
              <CardDescription>选择作物类型并制定种植计划</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-card-foreground">选择作物</label>
                <select 
                  className="w-full p-2 border rounded bg-black text-white"
                  value={selectedCrop}
                  onChange={(e) => setSelectedCrop(e.target.value)}
                >
                  {Object.keys(cropConfigs).map(crop => (
                    <option key={crop} value={crop}>{crop}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2 text-card-foreground">生长天数</label>
                <Input
                  type="number"
                  value={currentDay}
                  onChange={(e) => setCurrentDay(parseInt(e.target.value))}
                  min="1"
                  max="365"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2 text-card-foreground">种植目标</label>
                <select 
                  className="w-full p-2 border rounded bg-black text-white"
                  value={targetObjective}
                  onChange={(e) => setTargetObjective(e.target.value)}
                >
                  <option value="最大化产量">最大化产量</option>
                  <option value="提升甜度">提升甜度</option>
                  <option value="提升抗性">提升抗性</option>
                </select>
              </div>
              
              <Button onClick={createPlantingPlan} className="w-full">
                制定种植计划
              </Button>
            </CardContent>
          </Card>

          {/* 环境参数设置 */}
          <Card>
            <CardHeader>
              <CardTitle>环境参数</CardTitle>
              <CardDescription>设置当前环境条件</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-card-foreground">温度 (°C)</label>
                <Input
                  type="number"
                  value={environment.temperature}
                  onChange={(e) => setEnvironment({...environment, temperature: parseFloat(e.target.value)})}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2 text-card-foreground">湿度 (%)</label>
                <Input
                  type="number"
                  value={environment.humidity}
                  onChange={(e) => setEnvironment({...environment, humidity: parseFloat(e.target.value)})}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2 text-card-foreground">CO2浓度 (ppm)</label>
                <Input
                  type="number"
                  value={environment.co2}
                  onChange={(e) => setEnvironment({...environment, co2: parseFloat(e.target.value)})}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 中间：光配方和生长预测 */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>光配方系统</CardTitle>
              <CardDescription>AI生成最优光配方</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={generateLightRecipe} className="w-full">
                生成光配方
              </Button>
              
              {lightRecipe && (
                <div className="space-y-2">
                  <h4 className="font-semibold">光配方详情：</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>380nm紫外线: {(lightRecipe.uv_380nm * 100).toFixed(1)}%</div>
                    <div>720nm远红外: {(lightRecipe.far_red_720nm * 100).toFixed(1)}%</div>
                    <div>白光: {(lightRecipe.white_light * 100).toFixed(1)}%</div>
                    <div>660nm红光: {(lightRecipe.red_660nm * 100).toFixed(1)}%</div>
                    <div className="col-span-2">白红配比: {lightRecipe.white_red_ratio.toFixed(1)}:1</div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>生长预测</CardTitle>
              <CardDescription>AI预测植物生长状态</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={predictGrowth} className="w-full">
                预测生长状态
              </Button>
              
              {growthPrediction && (
                <div className="space-y-2">
                  <h4 className="font-semibold">生长状态预测：</h4>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {(growthPrediction.growth_rate * 100).toFixed(0)}%
                      </div>
                      <div>生长率</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {(growthPrediction.health_score * 100).toFixed(0)}%
                      </div>
                      <div>健康度</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">
                        {(growthPrediction.yield_potential * 100).toFixed(0)}%
                      </div>
                      <div>产量潜力</div>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>数据贡献</CardTitle>
              <CardDescription>贡献数据获得光子积分</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={contributeData} className="w-full" variant="outline">
                贡献当前数据
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* 右侧：种植计划和生长阶段 */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>种植计划</CardTitle>
              <CardDescription>详细的种植时间表</CardDescription>
            </CardHeader>
            <CardContent>
              {plantingPlan ? (
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span>作物类型:</span>
                    <span className="font-semibold">{plantingPlan.crop_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>总天数:</span>
                    <span className="font-semibold">{plantingPlan.total_days}天</span>
                  </div>
                  <div className="flex justify-between">
                    <span>目标产量:</span>
                    <span className="font-semibold">{plantingPlan.target_yield}</span>
                  </div>
                  
                  <div className="mt-4">
                    <h4 className="font-semibold mb-2">生长阶段计划：</h4>
                    <div className="space-y-2">
                      {(plantingPlan.planting_plan || []).map((stage: any, index: number) => (
                        <div key={index} className="p-2 border rounded">
                          <div className="font-medium">{stage.stage}</div>
                          <div className="text-sm text-gray-600">
                            第{stage.start_day}-{stage.end_day}天 | 
                            光照: {stage.light_hours}小时 | 
                            温度: {stage.temperature_range[0]}-{stage.temperature_range[1]}°C
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500 text-center">请先制定种植计划</p>
              )}
            </CardContent>
          </Card>

          {/* 当前生长阶段信息 */}
          <Card>
            <CardHeader>
              <CardTitle>当前生长阶段</CardTitle>
              <CardDescription>基于选择的作物和生长天数</CardDescription>
            </CardHeader>
            <CardContent>
              {selectedCrop && cropConfigs[selectedCrop] && (
                <div className="space-y-2">
                  {(cropConfigs[selectedCrop].growth_stages || []).map((stage, index) => {
                    const accumulatedDays = cropConfigs[selectedCrop].growth_stages
                      .slice(0, index)
                      .reduce((sum, s) => sum + s.duration_days, 0);
                    
                    const isCurrent = currentDay > accumulatedDays && 
                      currentDay <= accumulatedDays + stage.duration_days;
                    
                    return (
                      <div key={index} className={`p-2 border rounded ${isCurrent ? 'bg-black border-gray-700' : ''}`}>
                        <div className="flex justify-between">
                          <span className={`font-medium ${isCurrent ? 'text-white' : ''}`}>
                            {stage.stage_name}
                            {isCurrent && <span className="ml-2 text-xs bg-gray-700 text-white px-2 py-1 rounded">当前</span>}
                          </span>
                          <span className="text-sm text-gray-300">
                            第{accumulatedDays + 1}-{accumulatedDays + stage.duration_days}天
                          </span>
                        </div>
                        {isCurrent && (
                          <div className="mt-1 text-sm text-gray-300">
                            光照: {stage.light_hours}小时 | 
                            温度: {stage.optimal_temperature[0]}-{stage.optimal_temperature[1]}°C | 
                            湿度: {stage.optimal_humidity[0]}%-{stage.optimal_humidity[1]}%
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default AgriculturePage;