"""
模型管理器API路由
提供模型管理的高级功能：统计、搜索、导出、导入、备份等
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Query, File, UploadFile, Form
from pydantic import BaseModel

# 导入模型管理器
from src.core.services import model_manager

router = APIRouter(prefix="/model-manager", tags=["模型管理器"])


class ModelStatisticsResponse(BaseModel):
    """模型统计响应"""
    total_models: int
    loaded_models: int
    training_tasks: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    by_framework: Dict[str, int]


class SearchModelsRequest(BaseModel):
    """搜索模型请求"""
    query: Optional[str] = ""
    filters: Optional[Dict[str, Any]] = {}


class SearchModelsResponse(BaseModel):
    """搜索模型响应"""
    results: List[Dict[str, Any]]
    total_count: int
    query: str
    filters: Dict[str, Any]


class ExportModelRequest(BaseModel):
    """导出模型请求"""
    format: Optional[str] = "onnx"


class ExportModelResponse(BaseModel):
    """导出模型响应"""
    model_id: str
    format: str
    export_path: str
    file_size: int
    exported_at: str
    compatibility: Dict[str, Any]


class ImportModelRequest(BaseModel):
    """导入模型请求"""
    model_id: Optional[str] = None
    name: str
    model_type: str
    framework: Optional[str] = "pytorch"
    version: Optional[str] = "1.0.0"
    metadata: Optional[Dict[str, Any]] = {}


class BackupResponse(BaseModel):
    """备份响应"""
    backup_path: str
    backup_size: int
    total_models: int
    backup_time: str


class RestoreRequest(BaseModel):
    """恢复请求"""
    backup_path: str


@router.get("/statistics", response_model=ModelStatisticsResponse)
async def get_model_statistics():
    """获取模型统计信息"""
    try:
        result = await model_manager.get_model_statistics()
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return result["statistics"]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息失败: {str(e)}"
        )


@router.post("/search", response_model=SearchModelsResponse)
async def search_models(request: SearchModelsRequest):
    """搜索模型"""
    try:
        result = await model_manager.search_models(
            query=request.query,
            filters=request.filters
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return SearchModelsResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索模型失败: {str(e)}"
        )


@router.post("/{model_id}/export", response_model=ExportModelResponse)
async def export_model(model_id: str, request: ExportModelRequest):
    """导出模型到指定格式"""
    try:
        result = await model_manager.export_model(
            model_id=model_id,
            format=request.format
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return ExportModelResponse(**result["export_info"])
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出模型失败: {str(e)}"
        )


@router.post("/import-from-file", status_code=status.HTTP_201_CREATED)
async def import_model_from_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    model_type: str = Form(...),
    framework: str = Form("pytorch"),
    version: str = Form("1.0.0"),
    model_id: str = Form(None),
    metadata: str = Form("{}")
):
    """从文件导入模型"""
    try:
        import tempfile
        import os
        import json
        
        # 解析元数据
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # 保存上传的文件到临时目录
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # 准备模型信息
            model_info = {
                "model_id": model_id,
                "name": name,
                "type": model_type,
                "framework": framework,
                "version": version,
                "metadata": metadata_dict
            }
            
            # 导入模型
            result = await model_manager.import_model_from_file(
                file_path=file_path,
                model_info=model_info
            )
            
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
            
            return result
            
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导入模型失败: {str(e)}"
        )


@router.post("/backup", response_model=BackupResponse)
async def backup_model_metadata(backup_path: str = None):
    """备份模型元数据"""
    try:
        result = await model_manager.backup_model_metadata(backup_path)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return BackupResponse(**result["backup_info"])
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"备份模型元数据失败: {str(e)}"
        )


@router.post("/restore")
async def restore_model_metadata(request: RestoreRequest):
    """从备份恢复模型元数据"""
    try:
        result = await model_manager.restore_model_metadata(
            backup_path=request.backup_path
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return {
            "success": True,
            "message": "模型元数据恢复成功",
            "restore_info": result["restore_info"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复模型元数据失败: {str(e)}"
        )


@router.get("/status")
async def get_model_manager_status():
    """获取模型管理器状态"""
    try:
        # 获取统计信息
        stats_result = await model_manager.get_model_statistics()
        
        if not stats_result["success"]:
            stats = {"total_models": 0, "loaded_models": 0, "training_tasks": 0}
        else:
            stats = stats_result["statistics"]
        
        # 检查模型存储目录
        import os
        storage_path = model_manager.model_storage_path
        storage_exists = os.path.exists(storage_path)
        metadata_file_exists = os.path.exists(storage_path / "metadata.json")
        
        # 检查缓存状态
        cache_size = len(model_manager.model_cache)
        training_tasks = len(model_manager.training_tasks)
        
        return {
            "status": "running",
            "storage": {
                "path": str(storage_path),
                "exists": storage_exists,
                "metadata_file_exists": metadata_file_exists
            },
            "cache": {
                "size": cache_size,
                "models": list(model_manager.model_cache.keys())
            },
            "training": {
                "active_tasks": training_tasks,
                "task_ids": list(model_manager.training_tasks.keys())
            },
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型管理器状态失败: {str(e)}"
        )


@router.post("/{model_id}/reload")
async def reload_model(model_id: str):
    """重新加载模型"""
    try:
        # 先卸载模型
        unload_result = await model_manager.unload_model(model_id)
        
        if not unload_result["success"]:
            # 如果卸载失败，可能是模型未加载，继续尝试加载
            pass
        
        # 重新加载模型
        load_result = await model_manager.load_model(model_id)
        
        if not load_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=load_result["error"]
            )
        
        return {
            "success": True,
            "model_id": model_id,
            "message": "模型重新加载成功",
            "from_cache": load_result["from_cache"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重新加载模型失败: {str(e)}"
        )


@router.post("/cache/clear")
async def clear_model_cache():
    """清空模型缓存"""
    try:
        cache_size = len(model_manager.model_cache)
        cache_keys = list(model_manager.model_cache.keys())
        
        # 清空缓存
        model_manager.model_cache.clear()
        
        return {
            "success": True,
            "message": f"模型缓存已清空，释放了 {cache_size} 个模型",
            "cleared_models": cache_keys,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清空模型缓存失败: {str(e)}"
        )


@router.get("/versions/{base_model_id}")
async def get_model_version_history(base_model_id: str):
    """获取模型版本历史"""
    try:
        result = await model_manager.get_model_versions(base_model_id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型版本历史失败: {str(e)}"
        )