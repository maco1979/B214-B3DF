"""
推理服务API路由
提供文本生成、图像分类、图像生成等推理功能
"""

from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp
from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel

# 允许的API密钥列表
ALLOWED_API_KEYS = {
    "sk-2a8aa989f4864f32a3131201bcc04ad2"
}


async def verify_api_key(api_key: Optional[str] = Header(None)):
    """验证API密钥"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少API密钥",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if api_key not in ALLOWED_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API密钥",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key

# 使用绝对导入
from src.core.services import inference_engine, model_manager
from src.core.services.inference_engine import InferenceResult

router = APIRouter(prefix="/inference", tags=["inference"])


class TextGenerationRequest(BaseModel):
    """文本生成请求"""
    model_id: str
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0
    num_return_sequences: Optional[int] = 1
    beam_search: Optional[bool] = False
    beam_width: Optional[int] = 5
    early_stopping: Optional[bool] = True
    no_repeat_ngram_size: Optional[int] = 0
    do_sample: Optional[bool] = False
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0


class ImageClassificationRequest(BaseModel):
    """图像分类请求"""
    model_id: str
    image_data: List[List[List[float]]]  # 3D数组: [height, width, channels]
    top_k: Optional[int] = 5


class ImageGenerationRequest(BaseModel):
    """图像生成请求"""
    model_id: str
    num_samples: Optional[int] = 1
    image_size: Optional[int] = 256
    guidance_scale: Optional[float] = 7.5


class BatchInferenceRequest(BaseModel):
    """批量推理请求"""
    model_id: str
    inputs: List[Union[str, List[List[List[float]]]]]
    batch_size: Optional[int] = 32


class InferenceResponse(BaseModel):
    """推理响应"""
    predictions: Any
    confidence: Optional[float] = None
    processing_time: float
    metadata: Dict[str, Any]


class BatchInferenceResponse(BaseModel):
    """批量推理响应"""
    results: List[InferenceResponse]
    total_count: int
    success_count: int
    error_count: int


@router.post("/text/generation", response_model=InferenceResponse)
async def text_generation(request: TextGenerationRequest, api_key: str = Depends(verify_api_key)):
    """文本生成推理"""
    try:
        result = await inference_engine.text_generation(
            model_id=request.model_id,
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            num_return_sequences=request.num_return_sequences,
            beam_search=request.beam_search,
            beam_width=request.beam_width,
            early_stopping=request.early_stopping,
            no_repeat_ngram_size=request.no_repeat_ngram_size,
            do_sample=request.do_sample,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        return InferenceResponse(**result.to_dict())
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文本生成失败: {str(e)}"
        )


@router.post("/image/classification", response_model=InferenceResponse)
async def image_classification(request: ImageClassificationRequest, api_key: str = Depends(verify_api_key)):
    """图像分类推理"""
    try:
        # 转换图像数据为JAX数组
        image_array = jnp.array(request.image_data, dtype=jnp.float32)
        
        # 确保正确的形状: [batch, height, width, channels]
        if len(image_array.shape) == 3:
            image_array = image_array[None, ...]  # 添加batch维度
        
        result = await inference_engine.image_classification(
            model_id=request.model_id,
            image=image_array,
            top_k=request.top_k
        )
        
        return InferenceResponse(**result.to_dict())
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像分类失败: {str(e)}"
        )


@router.post("/image/generation", response_model=InferenceResponse)
async def image_generation(request: ImageGenerationRequest, api_key: str = Depends(verify_api_key)):
    """图像生成推理"""
    try:
        result = await inference_engine.image_generation(
            model_id=request.model_id,
            num_samples=request.num_samples,
            image_size=request.image_size,
            guidance_scale=request.guidance_scale
        )
        
        # 转换图像数据为可序列化格式
        predictions = result.predictions.tolist() if hasattr(result.predictions, 'tolist') else result.predictions
        
        response_data = result.to_dict()
        response_data["predictions"] = predictions
        
        return InferenceResponse(**response_data)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像生成失败: {str(e)}"
        )


@router.post("/batch", response_model=BatchInferenceResponse)
async def batch_inference(request: BatchInferenceRequest, api_key: str = Depends(verify_api_key)):
    """批量推理"""
    try:
        # 根据输入类型确定推理类型
        metadata = model_manager.get_model_info(request.model_id)
        
        if metadata.model_type == "vision":
            # 图像批量推理
            images = [jnp.array(img_data, dtype=jnp.float32) for img_data in request.inputs]
            results = await inference_engine.batch_inference(
                model_id=request.model_id,
                inputs=images,
                batch_size=request.batch_size
            )
        elif metadata.model_type == "transformer":
            # 文本批量推理
            results = await inference_engine.batch_inference(
                model_id=request.model_id,
                inputs=request.inputs,
                batch_size=request.batch_size
            )
        else:
            raise ValueError(f"不支持的批量推理模型类型: {metadata.model_type}")
        
        # 统计成功和失败的数量
        success_count = sum(1 for r in results if r.metadata.get("error") is None)
        error_count = len(results) - success_count
        
        # 转换结果格式
        inference_results = [InferenceResponse(**r.to_dict()) for r in results]
        
        return BatchInferenceResponse(
            results=inference_results,
            total_count=len(results),
            success_count=success_count,
            error_count=error_count
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量推理失败: {str(e)}"
        )


@router.get("/stats/{model_id}")
async def get_inference_stats(model_id: str, api_key: str = Depends(verify_api_key)):
    """获取推理统计信息"""
    try:
        stats = inference_engine.get_inference_stats(model_id)
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取推理统计失败: {str(e)}"
        )


@router.delete("/cache/{model_id}")
async def clear_inference_cache(model_id: str, api_key: str = Depends(verify_api_key)):
    """清除推理缓存"""
    try:
        inference_engine.clear_cache(model_id)
        return {"message": f"模型 {model_id} 的推理缓存已清除"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清除缓存失败: {str(e)}"
        )


@router.delete("/cache")
async def clear_all_inference_cache(api_key: str = Depends(verify_api_key)):
    """清除所有推理缓存"""
    try:
        inference_engine.clear_cache()
        return {"message": "所有推理缓存已清除"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清除所有缓存失败: {str(e)}"
        )