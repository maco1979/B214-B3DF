"""自适应模型生成API路由
提供跨行业自适应模型生成、管理和部署功能

安全特性:
- 输入参数验证（防止SQL注入/XSS攻击）
- 行业类型验证
- 模型生成任务管理
"""

import re
import html
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Body, Query, Path
from pydantic import BaseModel, validator, Field

# 使用绝对导入
from src.core.services import model_manager

router = APIRouter(prefix="/adaptive", tags=["adaptive"])


# ===== 安全验证工具函数 =====
def validate_industry_format(industry: str) -> str:
    """验证行业类型格式，防止注入攻击
    
    允许的行业类型:
    - agriculture: 农业
    - industry: 工业
    - home: 家庭
    - healthcare: 医疗
    - commercial: 商业
    - automotive: 汽车
    - logistics: 物流
    - energy: 能源
    """
    if not industry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="行业类型不能为空"
        )
    
    # 允许的行业类型白名单
    allowed_industries = [
        "agriculture", "industry", "home", "healthcare",
        "commercial", "automotive", "logistics", "energy"
    ]
    
    if industry not in allowed_industries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的行业类型: {industry}"
        )
    
    return industry


def validate_task_id_format(task_id: str) -> str:
    """验证任务ID格式，防止注入攻击"""
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="任务ID不能为空"
        )
    
    # 长度限制
    if len(task_id) > 256:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="任务ID长度不能超过256个字符"
        )
    
    # 检查危险字符（SQL注入特征）
    dangerous_patterns = [
        r"[\"'\-#/*\\]",  # SQL注释和引号
        r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|OR|AND)\b",  # SQL关键字
        r"--",  # SQL注释
        r"<\s*script",  # XSS
        r"javascript:",  # XSS
        r"on\w+\s*=",  # XSS事件处理器
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, task_id, re.IGNORECASE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="任务ID包含非法字符"
            )
    
    return task_id


def sanitize_string(value: str) -> str:
    """清理字符串，防止XSS"""
    if not value:
        return value
    return html.escape(str(value))


# ===== 请求和响应模型 =====
class GenerateAdaptiveModelRequest(BaseModel):
    """生成自适应模型请求"""
    industry: str = Field(..., description="目标行业类型")
    data: Optional[Dict[str, Any]] = Field({}, description="模型生成配置数据")
    model_name: Optional[str] = Field(None, max_length=256, description="自定义模型名称")
    model_type: Optional[str] = Field("adaptive", max_length=50, description="模型类型")
    
    @validator('industry')
    def validate_industry(cls, v):
        allowed_industries = [
            "agriculture", "industry", "home", "healthcare",
            "commercial", "automotive", "logistics", "energy"
        ]
        if v not in allowed_industries:
            raise ValueError(f"不支持的行业类型: {v}")
        return v
    
    @validator('model_name', pre=True, always=True)
    def sanitize_model_name(cls, v):
        if v is None:
            return v
        return html.escape(str(v)) if v else v


class ModelGenerationTaskResponse(BaseModel):
    """模型生成任务响应"""
    id: str
    industry: str
    status: str
    progress: int
    startTime: str
    endTime: Optional[str]
    error: Optional[str]


class AdaptiveModelResponse(BaseModel):
    """自适应模型响应"""
    id: str
    name: str
    industry: str
    type: str
    status: str
    accuracy: Optional[float]
    createdAt: str
    updatedAt: str
    description: Optional[str]


class ModelDeploymentResponse(BaseModel):
    """模型部署响应"""
    id: str
    modelId: str
    environment: str
    status: str
    deploymentTime: str
    error: Optional[str]


# ===== 模型生成任务管理 =====
# 存储生成任务的内存数据库
model_generation_tasks: Dict[str, Dict] = {}


# ===== API端点 =====
@router.post("/generate", response_model=ModelGenerationTaskResponse, status_code=status.HTTP_201_CREATED)
async def generate_adaptive_model(request: GenerateAdaptiveModelRequest):
    """生成自适应模型
    
    根据指定行业自动生成适配模型，支持跨行业应用。
    """
    try:
        # 验证行业类型
        industry = validate_industry_format(request.industry)
        
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 创建生成任务
        task = {
            "id": task_id,
            "industry": industry,
            "status": "pending",
            "progress": 0,
            "startTime": datetime.now().isoformat(),
            "endTime": None,
            "error": None
        }
        
        # 存储任务
        model_generation_tasks[task_id] = task
        
        # 异步模拟模型生成过程
        import asyncio
        asyncio.create_task(_simulate_model_generation(task_id, industry))
        
        # 返回任务信息
        return ModelGenerationTaskResponse(
            id=task["id"],
            industry=task["industry"],
            status=task["status"],
            progress=task["progress"],
            startTime=task["startTime"],
            endTime=task["endTime"],
            error=task["error"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成自适应模型失败: {str(e)}"
        )


async def _simulate_model_generation(task_id: str, industry: str):
    """模拟模型生成过程
    
    实际实现中，这里应该调用真实的模型生成服务。
    """
    try:
        # 更新任务状态为生成中
        model_generation_tasks[task_id]["status"] = "generating"
        
        # 模拟生成进度
        for progress in range(1, 101, 10):
            await asyncio.sleep(0.5)  # 模拟生成耗时
            model_generation_tasks[task_id]["progress"] = progress
        
        # 生成完成
        model_generation_tasks[task_id]["status"] = "completed"
        model_generation_tasks[task_id]["progress"] = 100
        model_generation_tasks[task_id]["endTime"] = datetime.now().isoformat()
        
        # 实际实现中，这里应该将生成的模型保存到模型仓库
        # 并调用model_manager.register_model()注册模型
        
    except Exception as e:
        # 更新任务状态为失败
        model_generation_tasks[task_id]["status"] = "failed"
        model_generation_tasks[task_id]["error"] = str(e)
        model_generation_tasks[task_id]["endTime"] = datetime.now().isoformat()


@router.get("/generate/{task_id}", response_model=ModelGenerationTaskResponse)
async def get_model_generation_status(task_id: str = Path(..., description="模型生成任务ID")):
    """获取模型生成任务状态"""
    try:
        # 验证任务ID
        task_id = validate_task_id_format(task_id)
        
        # 查找任务
        if task_id not in model_generation_tasks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="任务不存在"
            )
        
        task = model_generation_tasks[task_id]
        
        # 返回任务状态
        return ModelGenerationTaskResponse(
            id=task["id"],
            industry=task["industry"],
            status=task["status"],
            progress=task["progress"],
            startTime=task["startTime"],
            endTime=task["endTime"],
            error=task["error"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务状态失败: {str(e)}"
        )


@router.get("/models", response_model=List[AdaptiveModelResponse])
async def get_adaptive_models(industry: Optional[str] = Query(None, description="按行业筛选模型")):
    """获取自适应模型列表
    
    可选参数:
    - industry: 按行业筛选模型
    """
    try:
        # 验证行业类型（如果提供）
        if industry:
            industry = validate_industry_format(industry)
        
        # 获取所有模型
        all_models_result = await model_manager.list_models()
        
        if not all_models_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="获取模型列表失败"
            )
        
        # 筛选自适应模型
        adaptive_models = []
        for model in all_models_result["models"]:
            # 检查是否为自适应模型
            if model.get("type") == "adaptive" or "adaptive" in model.get("name", "").lower():
                # 如果指定了行业，只返回该行业的模型
                if industry and model.get("industry") != industry:
                    continue
                
                adaptive_model = {
                    "id": model["model_id"],
                    "name": model["name"],
                    "industry": model.get("industry", "agriculture"),
                    "type": model["type"],
                    "status": model["status"],
                    "accuracy": model.get("accuracy"),
                    "createdAt": model["created_at"],
                    "updatedAt": model["updated_at"],
                    "description": model.get("description")
                }
                adaptive_models.append(AdaptiveModelResponse(**adaptive_model))
        
        return adaptive_models
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取自适应模型失败: {str(e)}"
        )


@router.get("/models/{model_id}", response_model=AdaptiveModelResponse)
async def get_adaptive_model(model_id: str = Path(..., description="模型ID")):
    """获取自适应模型详情"""
    try:
        # 验证模型ID
        model_id = validate_task_id_format(model_id)  # 复用任务ID验证函数
        
        # 获取模型信息
        result = await model_manager.get_model_info(model_id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="模型不存在"
            )
        
        model = result["model"]
        
        # 检查是否为自适应模型
        if model.get("type") != "adaptive" and "adaptive" not in model.get("name", "").lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不是自适应模型"
            )
        
        # 构造响应
        adaptive_model = AdaptiveModelResponse(
            id=model["model_id"],
            name=model["name"],
            industry=model.get("industry", "agriculture"),
            type=model["type"],
            status=model["status"],
            accuracy=model.get("accuracy"),
            createdAt=model["created_at"],
            updatedAt=model["updated_at"],
            description=model.get("description")
        )
        
        return adaptive_model
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型详情失败: {str(e)}"
        )


@router.post("/deploy", response_model=ModelDeploymentResponse, status_code=status.HTTP_201_CREATED)
async def deploy_model(
    model_id: str = Body(..., description="模型ID"),
    environment: str = Body(..., description="部署环境")
):
    """部署自适应模型
    
    支持本地、边缘和云端部署。
    """
    try:
        # 验证模型ID
        model_id = validate_task_id_format(model_id)
        
        # 验证部署环境
        allowed_environments = ["local", "edge", "cloud"]
        if environment not in allowed_environments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的部署环境: {environment}"
            )
        
        # 生成部署ID
        deployment_id = str(uuid.uuid4())
        
        # 模拟部署过程
        # 实际实现中，这里应该调用真实的部署服务
        
        deployment = {
            "id": deployment_id,
            "modelId": model_id,
            "environment": environment,
            "status": "deployed",
            "deploymentTime": datetime.now().isoformat(),
            "error": None
        }
        
        # 返回部署结果
        return ModelDeploymentResponse(**deployment)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"部署模型失败: {str(e)}"
        )


@router.get("/deployments/{model_id}", response_model=List[ModelDeploymentResponse])
async def get_model_deployments(model_id: str = Path(..., description="模型ID")):
    """获取模型部署列表"""
    try:
        # 验证模型ID
        model_id = validate_task_id_format(model_id)
        
        # 模拟返回部署列表
        # 实际实现中，这里应该从部署服务获取真实的部署记录
        
        deployments = [
            {
                "id": str(uuid.uuid4()),
                "modelId": model_id,
                "environment": "local",
                "status": "deployed",
                "deploymentTime": datetime.now().isoformat(),
                "error": None
            }
        ]
        
        return [ModelDeploymentResponse(**deployment) for deployment in deployments]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取部署列表失败: {str(e)}"
        )


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_adaptive_model(model_id: str = Path(..., description="模型ID")):
    """删除自适应模型"""
    try:
        # 验证模型ID
        model_id = validate_task_id_format(model_id)
        
        # 调用模型管理器删除模型
        result = await model_manager.delete_model(model_id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="模型不存在"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除模型失败: {str(e)}"
        )
