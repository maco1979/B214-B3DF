"""
农业AI API路由 - 处理光谱分析、植物生长、光配方等农业功能
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import jax.numpy as jnp

from src.core.models.agriculture_model import AgricultureAIService, SpectrumConfig, CropConfig

router = APIRouter(prefix="/agriculture", tags=["agriculture"])

# 全局农业AI服务实例
agriculture_service = AgricultureAIService()


class LightRecipeRequest(BaseModel):
    """光配方请求"""
    crop_type: str
    current_day: int
    target_objective: str  # "最大化产量", "提升甜度", "提升抗性"
    environment: Dict[str, float]  # 温度、湿度等环境数据


class PlantGrowthRequest(BaseModel):
    """植物生长预测请求"""
    crop_type: str
    current_day: int
    environmental_data: Dict[str, float]
    spectrum_data: List[float]  # 光谱数据


class SpectrumAnalysisRequest(BaseModel):
    """光谱分析请求"""
    spectrum_data: List[float]  # 光谱数据数组


class CropPlanningRequest(BaseModel):
    """种植规划请求"""
    crop_type: str
    target_yield: str
    start_date: str
    expected_harvest_date: str


class DataContributionRequest(BaseModel):
    """数据贡献请求"""
    user_id: str
    crop_type: str
    growth_data: Dict[str, Any]
    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None


@router.post("/light-recipe")
async def generate_light_recipe(request: LightRecipeRequest):
    """生成光配方"""
    try:
        result = agriculture_service.generate_light_recipe(
            crop_type=request.crop_type,
            current_day=request.current_day,
            target_objective=request.target_objective,
            environment=request.environment
        )
        
        return {
            "success": True,
            "data": {
                "recipe": {
                    "uv_380nm": result["recipe"].uv_380nm,
                    "far_red_720nm": result["recipe"].far_red_720nm,
                    "white_light": result["recipe"].white_light,
                    "red_660nm": result["recipe"].red_660nm,
                    "white_red_ratio": result["recipe"].white_red_ratio
                },
                "current_stage": result["current_stage"],
                "light_hours": result["light_hours"],
                "recommendations": result["recommendations"]
            }
        }
    except Exception as e:
        import traceback
        print(f"Error in generate_light_recipe: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/growth-prediction")
async def predict_plant_growth(request: PlantGrowthRequest):
    """预测植物生长状态"""
    try:
        # 转换为JAX数组
        env_array = jnp.array([
            request.environmental_data.get('temperature', 25),
            request.environmental_data.get('humidity', 60),
            request.environmental_data.get('co2', 400)
        ])
        
        spec_array = jnp.array(request.spectrum_data)
        
        # 调用生长模型
        prediction = agriculture_service.growth_model(
            environmental_data=env_array,
            spectrum_data=spec_array,
            growth_days=request.current_day
        )
        
        return {
            "success": True,
            "data": {
                "growth_rate": float(prediction['growth_rate']),
                "health_score": float(prediction['health_score']),
                "yield_potential": float(prediction['yield_potential'])
            }
        }
    except Exception as e:
        import traceback
        print(f"Error in predict_plant_growth: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/spectrum-analysis")
async def analyze_spectrum(request: SpectrumAnalysisRequest):
    """分析光谱数据"""
    try:
        spectrum_array = jnp.array(request.spectrum_data)
        analysis = agriculture_service.spectrum_analyzer(spectrum_array)
        
        return {
            "success": True,
            "data": {
                "uv_380nm": float(analysis['uv_380nm']),
                "far_red_720nm": float(analysis['far_red_720nm']),
                "white_light": float(analysis['white_light']),
                "red_660nm": float(analysis['red_660nm']),
                "white_red_ratio": float(analysis['white_red_ratio'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/crop-configs")
async def get_available_crops():
    """获取可用的作物配置"""
    crop_configs = {}
    for crop_name, config in agriculture_service.crop_configs.items():
        crop_configs[crop_name] = {
            "growth_stages": [
                {
                    "stage_name": stage.stage_name,
                    "duration_days": stage.duration_days,
                    "optimal_temperature": stage.optimal_temperature,
                    "optimal_humidity": stage.optimal_humidity,
                    "light_hours": stage.light_hours
                }
                for stage in config.growth_stages
            ],
            "target_yield": config.target_yield,
            "quality_metrics": config.quality_metrics
        }
    
    return {
        "success": True,
        "data": crop_configs
    }


@router.post("/crop-planning")
async def plan_crop_growth(request: CropPlanningRequest):
    """制定种植计划"""
    try:
        if request.crop_type not in agriculture_service.crop_configs:
            raise HTTPException(status_code=400, detail=f"不支持的作物类型: {request.crop_type}")
        
        crop_config = agriculture_service.crop_configs[request.crop_type]
        
        # 计算总生长天数
        total_days = sum(stage.duration_days for stage in crop_config.growth_stages)
        
        # 生成详细的种植计划
        planting_plan = []
        current_day = 0
        
        for stage in crop_config.growth_stages:
            planting_plan.append({
                "stage": stage.stage_name,
                "start_day": current_day,
                "end_day": current_day + stage.duration_days,
                "light_hours": stage.light_hours,
                "temperature_range": stage.optimal_temperature,
                "humidity_range": stage.optimal_humidity,
                "key_activities": _get_stage_activities(stage.stage_name)
            })
            current_day += stage.duration_days
        
        return {
            "success": True,
            "data": {
                "crop_type": request.crop_type,
                "total_days": total_days,
                "target_yield": request.target_yield,
                "planting_plan": planting_plan
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/data-contribution")
async def contribute_growth_data(request: DataContributionRequest):
    """贡献生长数据"""
    try:
        # 计算光子积分
        photon_points = _calculate_photon_points(request)
        
        # 记录数据贡献
        contribution_record = {
            "user_id": request.user_id,
            "crop_type": request.crop_type,
            "contribution_time": "2024-01-01T00:00:00Z",  # 实际应该使用当前时间
            "photon_points": photon_points,
            "growth_data": request.growth_data,
            "media_count": len(request.images or []) + len(request.videos or [])
        }
        
        # 这里应该将数据保存到数据库或区块链
        
        return {
            "success": True,
            "data": {
                "photon_points_earned": photon_points,
                "contribution_id": "contribution_001",  # 生成唯一ID
                "message": "数据贡献成功，感谢您为农业AI发展做出的贡献！"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _get_stage_activities(stage_name: str) -> List[str]:
    """获取各阶段关键农事活动"""
    activities = {
        "苗期": ["播种", "育苗管理", "温度控制", "光照调节"],
        "开花期": ["授粉管理", "营养调控", "病虫害防治", "环境优化"],
        "结果期": ["果实管理", "营养补充", "采收准备", "品质控制"],
        "生长期": ["日常管理", "生长监测", "环境调控", "营养管理"]
    }
    return activities.get(stage_name, ["日常管理", "生长监测"])


def _calculate_photon_points(request: DataContributionRequest) -> int:
    """计算光子积分"""
    base_points = 10
    
    # 根据数据质量增加积分
    data_quality_bonus = 0
    if request.growth_data:
        data_quality_bonus += 5
    if request.images:
        data_quality_bonus += len(request.images) * 2
    if request.videos:
        data_quality_bonus += len(request.videos) * 5
    
    return base_points + data_quality_bonus


@router.get("/recommendations/{crop_type}")
async def get_growth_recommendations(crop_type: str, current_day: int):
    """获取生长建议"""
    try:
        if crop_type not in agriculture_service.crop_configs:
            raise HTTPException(status_code=400, detail=f"不支持的作物类型: {crop_type}")
        
        crop_config = agriculture_service.crop_configs[crop_type]
        current_stage = agriculture_service._get_current_stage(crop_config, current_day)
        
        recommendations = agriculture_service._get_growth_recommendations(crop_config, current_stage)
        
        return {
            "success": True,
            "data": {
                "crop_type": crop_type,
                "current_stage": current_stage.stage_name,
                "current_day": current_day,
                "recommendations": recommendations,
                "next_stage": _get_next_stage_info(crop_config, current_day)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _get_next_stage_info(crop_config: CropConfig, current_day: int) -> Dict[str, any]:
    """获取下一阶段信息"""
    accumulated_days = 0
    for i, stage in enumerate(crop_config.growth_stages):
        accumulated_days += stage.duration_days
        if current_day <= accumulated_days:
            if i + 1 < len(crop_config.growth_stages):
                next_stage = crop_config.growth_stages[i + 1]
                return {
                    "stage_name": next_stage.stage_name,
                    "days_until": accumulated_days - current_day,
                    "preparation_tips": _get_preparation_tips(next_stage.stage_name)
                }
    return {"stage_name": "已完成", "days_until": 0, "preparation_tips": []}


def _get_preparation_tips(self, next_stage: str) -> List[str]:
    """获取阶段转换准备建议"""
    tips = {
        "开花期": ["调整营养配方", "准备授粉工具", "优化环境条件"],
        "结果期": ["增加营养供应", "准备支撑设施", "监测病虫害"],
        "生长期": ["常规管理", "监测生长", "调整环境"]
    }
    return tips.get(next_stage, ["继续当前管理措施"])