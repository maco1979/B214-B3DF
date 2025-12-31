import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Slider } from './ui/slider';

interface SpectrumConfig {
  uv_380nm: number;      // 380nm紫外线强度 (0-1)
  far_red_720nm: number; // 720nm远红外线强度 (0-1)
  white_light: number;   // 白光强度 (0-1)
  red_660nm: number;     // 660nm红光强度 (0-1)
}

interface SpectrumControlProps {
  initialConfig?: SpectrumConfig;
  onConfigChange?: (config: SpectrumConfig) => void;
  onApplyRecipe?: (config: SpectrumConfig) => void;
}

const SpectrumControl: React.FC<SpectrumControlProps> = ({
  initialConfig = {
    uv_380nm: 0.05,
    far_red_720nm: 0.1,
    white_light: 0.7,
    red_660nm: 0.15
  },
  onConfigChange,
  onApplyRecipe
}) => {
  const [config, setConfig] = useState<SpectrumConfig>(initialConfig);
  const [whiteRedRatio, setWhiteRedRatio] = useState<number>(0);

  useEffect(() => {
    // 计算白红配比
    const ratio = config.red_660nm > 0 ? config.white_light / config.red_660nm : 0;
    setWhiteRedRatio(ratio);
    
    onConfigChange?.(config);
  }, [config, onConfigChange]);

  const handleSliderChange = (key: keyof SpectrumConfig, value: number[]) => {
    setConfig(prev => ({
      ...prev,
      [key]: value[0] / 100 // 转换为0-1范围
    }));
  };

  const applyRatioConstraint = () => {
    // 强制应用31:1白红配比
    const targetRatio = 31;
    const newRed = config.white_light / targetRatio;
    
    // 调整其他波段以保持总和为1
    const totalOthers = config.uv_380nm + config.far_red_720nm;
    const availableForWhiteRed = 1 - totalOthers;
    
    if (newRed + config.white_light > availableForWhiteRed) {
      // 需要调整白光和红光的比例
      const adjustedWhite = availableForWhiteRed * (targetRatio / (targetRatio + 1));
      const adjustedRed = availableForWhiteRed * (1 / (targetRatio + 1));
      
      setConfig({
        uv_380nm: config.uv_380nm,
        far_red_720nm: config.far_red_720nm,
        white_light: adjustedWhite,
        red_660nm: adjustedRed
      });
    } else {
      setConfig({
        ...config,
        red_660nm: newRed
      });
    }
  };

  const getBandColor = (band: keyof SpectrumConfig) => {
    const colors = {
      uv_380nm: 'bg-purple-500',
      far_red_720nm: 'bg-red-800',
      white_light: 'bg-white border',
      red_660nm: 'bg-red-500'
    };
    return colors[band];
  };

  const getBandName = (band: keyof SpectrumConfig) => {
    const names = {
      uv_380nm: '380nm紫外线',
      far_red_720nm: '720nm远红外',
      white_light: '白光(420-450nm)',
      red_660nm: '660nm红光'
    };
    return names[band];
  };

  const getBandDescription = (band: keyof SpectrumConfig) => {
    const descriptions = {
      uv_380nm: '刺激次生代谢物，提升品质与抗性',
      far_red_720nm: '调控光形态建成，影响开花时间',
      white_light: '基础光合作用能量，高色温蓝光成分',
      red_660nm: '光合作用效率峰值，驱动碳水化合物合成'
    };
    return descriptions[band];
  };

  const presetRecipes = {
    seedling: { uv_380nm: 0.02, far_red_720nm: 0.08, white_light: 0.75, red_660nm: 0.15 },
    flowering: { uv_380nm: 0.08, far_red_720nm: 0.12, white_light: 0.65, red_660nm: 0.15 },
    fruiting: { uv_380nm: 0.06, far_red_720nm: 0.1, white_light: 0.7, red_660nm: 0.14 },
    maxYield: { uv_380nm: 0.05, far_red_720nm: 0.09, white_light: 0.68, red_660nm: 0.18 },
    highQuality: { uv_380nm: 0.1, far_red_720nm: 0.11, white_light: 0.62, red_660nm: 0.17 }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>光谱控制系统</CardTitle>
        <CardDescription>
          精准控制LED光谱各波段强度，优化植物生长环境
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        
        {/* 光谱波段控制 */}
        <div className="space-y-4">
          {(Object.keys(config) as Array<keyof SpectrumConfig>).map((band) => (
            <div key={band} className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-2">
                  <div className={`w-4 h-4 rounded ${getBandColor(band)}`}></div>
                  <span className="font-medium">{getBandName(band)}</span>
                </div>
                <span className="text-sm font-semibold">
                  {(config[band] * 100).toFixed(0)}%
                </span>
              </div>
              
              <Slider
                value={[config[band] * 100]}
                onValueChange={(value) => handleSliderChange(band, value)}
                max={100}
                step={1}
                className="w-full"
              />
              
              <p className="text-xs text-gray-500">
                {getBandDescription(band)}
              </p>
            </div>
          ))}
        </div>

        {/* 配比信息和控制 */}
        <div className="border-t pt-4">
          <div className="flex justify-between items-center mb-4">
            <div>
              <span className="font-medium">白红配比: </span>
              <span className={whiteRedRatio >= 30 && whiteRedRatio <= 32 ? 'text-green-600' : 'text-red-600'}>
                {whiteRedRatio.toFixed(1)}:1
              </span>
            </div>
            <Button 
              size="sm" 
              variant="outline"
              onClick={applyRatioConstraint}
              disabled={whiteRedRatio >= 30 && whiteRedRatio <= 32}
            >
              应用31:1配比
            </Button>
          </div>
          
          {whiteRedRatio < 30 && (
            <p className="text-sm text-yellow-600">
              ⚠️ 白红配比偏低，建议增加白光比例
            </p>
          )}
          
          {whiteRedRatio > 32 && (
            <p className="text-sm text-yellow-600">
              ⚠️ 白红配比偏高，建议增加红光比例
            </p>
          )}
          
          {whiteRedRatio >= 30 && whiteRedRatio <= 32 && (
            <p className="text-sm text-green-600">
              ✅ 白红配比符合31:1标准
            </p>
          )}
        </div>

        {/* 预设配方 */}
        <div className="border-t pt-4">
          <h4 className="font-medium mb-3">预设光配方</h4>
          <div className="grid grid-cols-2 gap-2">
            <Button 
              size="sm" 
              variant="outline"
              onClick={() => setConfig(presetRecipes.seedling)}
            >
              苗期配方
            </Button>
            <Button 
              size="sm" 
              variant="outline"
              onClick={() => setConfig(presetRecipes.flowering)}
            >
              开花期配方
            </Button>
            <Button 
              size="sm" 
              variant="outline"
              onClick={() => setConfig(presetRecipes.fruiting)}
            >
              结果期配方
            </Button>
            <Button 
              size="sm" 
              variant="outline"
              onClick={() => setConfig(presetRecipes.maxYield)}
            >
              高产配方
            </Button>
            <Button 
              size="sm" 
              variant="outline"
              onClick={() => setConfig(presetRecipes.highQuality)}
              className="col-span-2"
            >
              高品质配方
            </Button>
          </div>
        </div>

        {/* 光谱可视化 */}
        <div className="border-t pt-4">
          <h4 className="font-medium mb-3">光谱分布图</h4>
          <div className="h-20 bg-gradient-to-r from-purple-400 via-white to-red-600 rounded relative">
            {/* 标记关键波段位置 */}
            <div className="absolute top-0 left-[5%] w-0.5 h-full bg-purple-800">
              <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs">380nm</div>
            </div>
            <div className="absolute top-0 left-[45%] w-0.5 h-full bg-gray-400">
              <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs">白光</div>
            </div>
            <div className="absolute top-0 left-[82.5%] w-0.5 h-full bg-red-800">
              <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs">660nm</div>
            </div>
            <div className="absolute top-0 left-[90%] w-0.5 h-full bg-red-900">
              <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs">720nm</div>
            </div>
            
            {/* 强度指示 */}
            <div 
              className="absolute bottom-0 left-[5%] w-2 bg-purple-800 opacity-70"
              style={{ height: `${config.uv_380nm * 80}%` }}
            ></div>
            <div 
              className="absolute bottom-0 left-[90%] w-2 bg-red-900 opacity-70"
              style={{ height: `${config.far_red_720nm * 80}%` }}
            ></div>
            <div 
              className="absolute bottom-0 left-[45%] w-10 bg-white opacity-70"
              style={{ height: `${config.white_light * 80}%` }}
            ></div>
            <div 
              className="absolute bottom-0 left-[82.5%] w-2 bg-red-800 opacity-70"
              style={{ height: `${config.red_660nm * 80}%` }}
            ></div>
          </div>
        </div>

        {/* 应用按钮 */}
        <Button 
          className="w-full"
          onClick={() => onApplyRecipe?.(config)}
        >
          应用光配方到设备
        </Button>
      </CardContent>
    </Card>
  );
};



export default SpectrumControl;