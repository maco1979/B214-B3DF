"""
模型管理API路由
提供模型的创建、查询、更新和删除功能

安全特性:
- 输入参数验证（防止SQL注入/XSS攻击）
- 模型ID格式验证
"""

import re
import html
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Path, Body, Query
from pydantic import BaseModel, validator, Field

# 使用绝对导入
from src.core.services import model_manager

router = APIRouter(prefix="/models", tags=["models"])


# ===== 安全验证工具函数 =====
def validate_model_id_format(model_id: str) -> str:
    """验证模型ID格式，防止注入攻击
    
    允许的字符: 字母、数字、下划线、短横线、点、中文字符
    """
    if not model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="模型ID不能为空"
        )
    
    # 长度限制
    if len(model_id) > 256:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="模型ID长度不能超过256个字符"
        )
    
    # 检查危险字符（SQL注入特征）
    dangerous_patterns = [
        r"['\";–#/*\\]",  # SQL注释和引号
        r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|OR|AND)\b",  # SQL关键字
        r"--",  # SQL注释
        r"<\s*script",  # XSS
        r"javascript:",  # XSS
        r"on\w+\s*=",  # XSS事件处理器
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, model_id, re.IGNORECASE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="模型ID包含非法字符"
            )
    
    return model_id


def validate_search_query_format(query: str) -> str:
    """验证搜索查询格式，防止注入攻击"""
    if not query:
        return query
    
    # 长度限制
    if len(query) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="搜索查询长度不能超过500个字符"
        )
    
    # 检查SQL注入特征
    sql_patterns = [
        r"'\s*;\s*--",
        r"\b(DROP|DELETE|TRUNCATE|INSERT|UPDATE)\b.*\b(TABLE|FROM|INTO)\b",
        r"UNION\s+SELECT",
        r"1\s*=\s*1",
        r"OR\s+1\s*=\s*1",
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="搜索查询包含非法字符"
            )
    
    # 转义特殊字符
    return html.escape(query)


def sanitize_string(value: str) -> str:
    """清理字符串，防止XSS"""
    if not value:
        return value
    return html.escape(str(value))


class CreateModelRequest(BaseModel):
    """创建模型请求"""
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = Field("", max_length=2000)
    status: Optional[str] = Field("ready", max_length=50)
    version: Optional[str] = Field("v1.0.0", max_length=50)
    model_type: Optional[str] = Field("ai", max_length=50)
    hyperparameters: Optional[Dict[str, Any]] = {}
    model_id: Optional[str] = Field(None, max_length=256)
    
    @validator('name', 'model_id', 'description', pre=True, always=True)
    def sanitize_strings(cls, v):
        if v is None:
            return v
        # XSS防护
        return html.escape(str(v)) if v else v
    
    @validator('model_id', pre=True, always=True)
    def validate_model_id_field(cls, v):
        if v is None:
            return v
        # 检查危险模式
        dangerous = ["'", '"', ';', '--', '/*', '*/', 'DROP', 'DELETE', 'UNION', 'SELECT']
        v_upper = str(v).upper()
        for d in dangerous:
            if d.upper() in v_upper:
                raise ValueError('模型ID包含非法字符')
        return v


class LoadPretrainedModelRequest(BaseModel):
    """加载预训练模型请求"""
    model_name_or_path: str
    model_format: Optional[str] = "huggingface"
    model_type: Optional[str] = "transformer"
    name: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}


class ModelResponse(BaseModel):
    """模型响应"""
    id: str
    model_id: str
    model_type: str
    name: str
    version: str
    created_at: str
    updated_at: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    description: str
    status: str
    accuracy: Optional[float] = None
    size: Optional[int] = None
    framework: Optional[str] = None
    is_pretrained: Optional[bool] = False
    pretrained_source: Optional[str] = None
    model_format: Optional[str] = None
    quantization: Optional[Dict[str, Any]] = None


class ListModelsResponse(BaseModel):
    """模型列表响应"""
    models: List[ModelResponse]
    total_count: int


@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(request: CreateModelRequest):
    """创建新模型"""
    try:
        # 如果没有提供model_id，则使用name生成
        model_id = request.model_id or request.name.lower().replace(" ", "-")
        
        # 检查是否需要添加版本号
        if "_v" not in model_id:
            model_id = f"{model_id}_v1"
        
        # 映射前端模型类型到后端支持的模型类型
        model_type_mapping = {
            "ai": "transformer",
            "ml": "vision",
            "transformer": "transformer",
            "vision": "vision",
            "diffusion": "diffusion"
        }
        
        # 获取实际的模型类型，如果没有映射则默认使用"transformer"
        actual_model_type = model_type_mapping.get(request.model_type, "transformer")
        
        # 根据模型类型设置默认超参数
        default_hyperparameters = {}
        if actual_model_type == "transformer":
            default_hyperparameters = {
                "vocab_size": 30000,
                "max_seq_len": 2048,
                "d_model": 2048,
                "num_heads": 16,
                "num_layers": 24,
                "d_ff": 8192
            }
        elif actual_model_type == "vision":
            default_hyperparameters = {
                "image_size": 224,
                "image_channels": 3,
                "num_classes": 10,
                "base_channels": 64
            }
        elif actual_model_type == "diffusion":
            default_hyperparameters = {
                "image_size": 64,
                "image_channels": 3,
                "base_channels": 64
            }
        
        # 合并默认超参数和请求中的超参数
        hyperparameters = {
            **default_hyperparameters,
            **(request.hyperparameters or {})
        }
        
        # 准备模型数据
        model_data = {
            "name": request.name,
            "type": actual_model_type,
            "version": request.version,
            "status": request.status,
            "metadata": hyperparameters,
            "description": request.description or ""
        }
        
        # 注册模型
        result = await model_manager.register_model(model_id, model_data)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        # 构造响应
        model_dict = {
            "id": result["model_id"],
            "model_id": result["model_id"],
            "model_type": request.model_type,
            "name": request.name,
            "version": request.version,
            "created_at": model_manager.model_metadata[result["model_id"]]["created_at"],
            "updated_at": model_manager.model_metadata[result["model_id"]]["updated_at"],
            "metrics": {},
            "hyperparameters": hyperparameters,
            "description": request.description or "",
            "status": request.status
        }
        
        return ModelResponse(**model_dict)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建模型失败: {str(e)}"
        )


@router.get("/", response_model=List[ModelResponse])
async def list_models(
    search: Optional[str] = Query(None, max_length=500, description="搜索关键词")
):
    """获取模型列表
    
    可选参数:
    - search: 搜索关键词（已进行安全验证）
    """
    try:
        # 验证搜索参数
        if search:
            search = validate_search_query_format(search)
        
        result = await model_manager.list_models()
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        model_responses = []
        for model in result["models"]:
            model_dict = {
                "id": model["model_id"],
                "model_id": model["model_id"],
                "model_type": model["type"],
                "name": model["name"],
                "version": model["version"],
                "created_at": model["created_at"],
                "updated_at": model["updated_at"],
                "metrics": model.get("metrics", {}),
                "hyperparameters": model.get("metadata", {}),
                "description": model.get("description", ""),
                "status": model["status"],
                "accuracy": model.get("accuracy"),
                "framework": model.get("framework"),
                "is_pretrained": model.get("is_pretrained", False),
                "pretrained_source": model.get("pretrained_source"),
                "model_format": model.get("model_format"),
                "quantization": model.get("quantization")
            }
            model_responses.append(ModelResponse(**model_dict))
        
        return model_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型列表失败: {str(e)}"
        )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str = Path(..., min_length=1, max_length=256)):
    """获取模型详情
    
    路径参数:
    - model_id: 模型ID（已进行安全验证）
    """
    try:
        # 安全验证模型ID
        model_id = validate_model_id_format(model_id)
        
        result = await model_manager.get_model_info(model_id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        model = result["model"]
        model_dict = {
            "id": model["model_id"],
            "model_id": model["model_id"],
            "model_type": model["type"],
            "name": model["name"],
            "version": model["version"],
            "created_at": model["created_at"],
            "updated_at": model["updated_at"],
            "metrics": model.get("metrics", {}),
            "hyperparameters": model.get("metadata", {}),
            "description": model.get("description", ""),
            "status": model["status"],
            "accuracy": model.get("accuracy"),
            "framework": model.get("framework"),
            "is_pretrained": model.get("is_pretrained", False),
            "pretrained_source": model.get("pretrained_source"),
            "model_format": model.get("model_format"),
            "quantization": model.get("quantization")
        }
        return ModelResponse(**model_dict)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型详情失败: {str(e)}"
        )


@router.put("/{model_id}/metrics")
async def update_model_metrics(
    model_id: str = Path(..., min_length=1, max_length=256),
    metrics: Dict[str, float] = Body(...)
):
    """更新模型指标"""
    try:
        # 安全验证模型ID
        model_id = validate_model_id_format(model_id)
        
        # 直接更新模型元数据中的metrics字段
        if model_id not in model_manager.model_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="模型不存在"
            )
        
        # 更新指标
        if "metrics" not in model_manager.model_metadata[model_id]:
            model_manager.model_metadata[model_id]["metrics"] = {}
        
        model_manager.model_metadata[model_id]["metrics"].update(metrics)
        model_manager.model_metadata[model_id]["updated_at"] = datetime.now().isoformat()
        
        # 保存更新后的元数据
        await model_manager._save_model_metadata()
        
        return {"message": "模型指标更新成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新模型指标失败: {str(e)}"
        )


@router.delete("/{model_id}")
async def delete_model(model_id: str = Path(..., min_length=1, max_length=256)):
    """删除模型"""
    try:
        # 安全验证模型ID
        model_id = validate_model_id_format(model_id)
        
        result = await model_manager.delete_model(model_id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        return {"message": "模型删除成功"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除模型失败: {str(e)}"
        )


@router.post("/pretrained", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def load_pretrained_model(request: LoadPretrainedModelRequest):
    """加载预训练模型"""
    try:
        result = await model_manager.load_pretrained_model(request.dict())
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"加载预训练模型失败: {str(e)}"
        )


@router.post("/import", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def import_model(
    name: str = Form(...),
    file: UploadFile = File(...),
    description: str = Form(""),
    version: str = Form("v1.0.0")
):
    """导入模型文件"""
    try:
        # 保存上传的文件
        import tempfile
        import os
        
        # 使用系统临时目录
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        
        try:
            # 保存上传的文件
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # 生成model_id
            model_id = name.lower().replace(" ", "-")
            
            # 使用模型管理器创建模型
            # 默认使用transformer模型类型
            metadata = model_manager.create_model(
                model_type="transformer",
                model_id=model_id,
                hyperparameters={},
                description=description or "",
                file_path=file_path,
                name=name,
                status="ready",
                version=version
            )
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # 构造响应
        model_dict = metadata.to_dict()
        model_dict['id'] = model_dict.get('model_id', '')
        model_dict['name'] = name
        model_dict['status'] = model_dict.get('status', 'ready')
        model_dict['updated_at'] = model_dict.get('updated_at', model_dict.get('created_at', ''))
        
        return ModelResponse(**model_dict)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导入模型失败: {str(e)}"
        )


@router.get("/{model_id}/versions")
async def get_model_versions(model_id: str):
    """获取模型版本历史"""
    try:
        # 调用模型管理器的方法获取版本历史
        result = await model_manager.get_model_versions(model_id)
        
        if result["success"]:
            return {
                "success": True,
                "versions": result["versions"]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型版本历史失败: {str(e)}"
        )


class CreateModelVersionRequest(BaseModel):
    name: Optional[str] = ""
    type: Optional[str] = "transformer"
    version: Optional[str] = None
    status: Optional[str] = "ready"
    metadata: Optional[Dict[str, Any]] = {}
    description: Optional[str] = ""
    framework: Optional[str] = "pytorch"


@router.post("/{model_id}/versions")
async def create_model_version(model_id: str, request_data: CreateModelVersionRequest = Body(...)): 
    """创建模型新版本"""
    try:
        # 首先检查原始模型是否存在
        original_model_info = await model_manager.get_model_info(model_id)
        if not original_model_info["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="原始模型不存在"
            )
        
        # 从原始模型信息中获取基础信息
        original_model = original_model_info["model"]
        
        # 从请求中提取必要的模型数据，避免传递不兼容的字段
        model_data = {
            "name": request_data.name or original_model.get("name", f"{model_id}_new_version"),
            "type": request_data.type or original_model.get("type", "transformer"),
            "version": request_data.version,
            "status": request_data.status or "ready",
            "metadata": request_data.metadata or original_model.get("metadata", {}),
            "description": request_data.description or original_model.get("description", ""),
            "framework": request_data.framework or original_model.get("framework", "pytorch"),
        }
        
        # 调用模型管理器的方法创建新版本，让模型管理器内部处理版本号生成
        result = await model_manager.register_model(
            model_id=model_id,  # 使用原始模型ID作为基础
            model_data=model_data,
            is_new_version=True
        )
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
            
    except Exception as e:
        import traceback
        print(f"创建模型新版本失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建模型新版本失败: {str(e)}"
        )


# 训练相关的Pydantic模型
class StartTrainingRequest(BaseModel):
    """开始训练请求"""
    training_data: Dict[str, Any]


class TrainingStatusResponse(BaseModel):
    """训练状态响应"""
    task_id: str
    model_id: str
    status: str
    progress: int
    stage: str
    current_step: int
    total_steps: int
    started_at: str
    completed_at: Optional[str]
    metrics: Optional[Dict[str, float]] = None
    model_status: Optional[str] = None


@router.post("/{model_id}/train", status_code=status.HTTP_200_OK)
async def start_training(model_id: str, request: StartTrainingRequest):
    """开始模型训练"""
    try:
        result = await model_manager.start_training(model_id, request.training_data)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"开始训练失败: {str(e)}"
        )


@router.get("/training/{task_id}", response_model=TrainingStatusResponse)
async def get_training_status(task_id: str):
    """获取训练任务状态"""
    try:
        result = await model_manager.get_training_status(task_id)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练状态失败: {str(e)}"
        )


class QuantizeModelRequest(BaseModel):
    """量化模型请求"""
    quantization_type: Optional[str] = "int8"


@router.post("/{model_id}/quantize", status_code=status.HTTP_200_OK)
async def quantize_model(model_id: str, request: QuantizeModelRequest):
    """量化模型"""
    try:
        result = await model_manager.quantize_model(
            model_id=model_id,
            quantization_type=request.quantization_type
        )
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"量化模型失败: {str(e)}"
        )


class QuantizedInferenceRequest(BaseModel):
    """量化推理请求"""
    input_data: Dict[str, Any]
    quantization_type: Optional[str] = "int8"


# 模型控制相关API端点
@router.post("/{model_id}/start", status_code=status.HTTP_200_OK)
async def start_model(model_id: str = Path(..., title="Model ID")):
    """启动模型 - 将模型加载到内存中"""
    try:
        # 通过加载模型来启动它
        result = await model_manager.load_model(model_id)
        
        if result["success"]:
            # 更新模型状态为运行中
            if model_id in model_manager.model_metadata:
                model_manager.model_metadata[model_id]["status"] = "running"
                model_manager.model_metadata[model_id]["updated_at"] = datetime.now().isoformat()
                await model_manager._save_model_metadata()
            
            return {
                "success": True,
                "message": "模型启动成功",
                "model_id": model_id,
                "status": "running"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动模型失败: {str(e)}"
        )


@router.post("/{model_id}/pause", status_code=status.HTTP_200_OK)
async def pause_model(model_id: str = Path(..., title="Model ID")):
    """暂停模型 - 从内存中卸载模型"""
    try:
        # 通过卸载模型来暂停它
        result = await model_manager.unload_model(model_id)
        
        if result["success"]:
            # 更新模型状态为已停止
            if model_id in model_manager.model_metadata:
                model_manager.model_metadata[model_id]["status"] = "stopped"
                model_manager.model_metadata[model_id]["updated_at"] = datetime.now().isoformat()
                await model_manager._save_model_metadata()
            
            return {
                "success": True,
                "message": "模型暂停成功",
                "model_id": model_id,
                "status": "stopped"
            }
        else:
            # 如果模型未加载，也认为是成功的暂停
            if "not loaded" in str(result.get("error", "")):
                if model_id in model_manager.model_metadata:
                    model_manager.model_metadata[model_id]["status"] = "stopped"
                    model_manager.model_metadata[model_id]["updated_at"] = datetime.now().isoformat()
                    await model_manager._save_model_metadata()
                
                return {
                    "success": True,
                    "message": "模型已经处于停止状态",
                    "model_id": model_id,
                    "status": "stopped"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停模型失败: {str(e)}"
        )


@router.post("/{model_id}/inference/quantized", status_code=status.HTTP_200_OK)
async def predict_with_quantization(model_id: str, request: QuantizedInferenceRequest):
    """使用量化模型进行推理"""
    try:
        result = await model_manager.predict_with_quantization(
            model_id=model_id,
            input_data=request.input_data,
            quantization_type=request.quantization_type
        )
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"量化推理失败: {str(e)}"
        )