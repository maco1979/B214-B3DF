"""
jena-DT-MPC API路由 - 提供JEPA-DT-MPC模型的激活和训练功能
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging
import numpy as np
import time
import sys
import os

# 导入JEPA-DT-MPC控制器
import importlib.util
import sys

# 动态导入jepa_dtmpc_integration
jepa_dtmpc_spec = importlib.util.spec_from_file_location(
    "jepa_dtmpc_integration",
    os.path.join(os.path.dirname(__file__), "../../../jepa_dtmpc_integration.py")
)
jepa_dtmpc_module = importlib.util.module_from_spec(jepa_dtmpc_spec)
jepa_dtmpc_spec.loader.exec_module(jepa_dtmpc_module)

JepaDtmpcController = jepa_dtmpc_module.JepaDtmpcController

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jepa-dtmpc", tags=["jepa-dtmpc"])

# JEPA-DT-MPC控制器实例
_jepa_dtmpc_controller = None

# 激活请求模型
class ActivateRequest(BaseModel):
    controller_params: Dict[str, Any]
    mv_params: Dict[str, Any]
    cv_params: Dict[str, Any]
    model_params: Dict[str, Any]
    jepa_params: Optional[Dict[str, Any]] = None

# 训练请求模型
class TrainRequest(BaseModel):
    training_data: Dict[str, Any]
    training_steps: Optional[int] = 100

# 激活响应模型
class ActivateResponse(BaseModel):
    success: bool
    message: str
    controller_info: Optional[Dict[str, Any]] = None

# 训练响应模型
class TrainResponse(BaseModel):
    success: bool
    message: str
    training_result: Optional[Dict[str, Any]] = None

@router.post("/activate", response_model=ActivateResponse)
async def activate_jepa_dtmpc(request: ActivateRequest):
    """
    激活JEPA-DT-MPC控制器
    """
    global _jepa_dtmpc_controller
    try:
        # 初始化JEPA-DT-MPC控制器
        _jepa_dtmpc_controller = JepaDtmpcController(
            controller_params=request.controller_params,
            mv_params=request.mv_params,
            cv_params=request.cv_params,
            model_params=request.model_params,
            jepa_params=request.jepa_params
        )
        
        # 构造控制器信息
        controller_info = {
            "jepa_enabled": _jepa_dtmpc_controller.jepa_enabled,
            "jepa_trained": _jepa_dtmpc_controller.jepa_trained,
            "jepa_embedding_dim": _jepa_dtmpc_controller.jepa_embedding_dim,
            "mpc_prediction_horizon": _jepa_dtmpc_controller.mpc_core.Np,
            "mpc_control_horizon": _jepa_dtmpc_controller.mpc_core.Nc
        }
        
        return ActivateResponse(
            success=True,
            message="""# JEPA stands for Joint Embedding and Predictive Analytics. In the provided code,
            # JEPA-DT-MPC (Joint Embedding Predictive Analytics with Dynamic Time Warping
            # Model Predictive Control) is a controller used for model activation and
            # training. JEPA-DT-MPC integrates embedding techniques with predictive analytics
            # to enhance control strategies in dynamic systems. The controller is activated
            # to enable JEPA features and then trained using training data to improve its
            # predictive capabilities. Additionally, the controller status and prediction
            # results can be accessed through the defined API routes.
            JEPA-DT-MPC控制器激活成功""",
            controller_info=controller_info
        )
        
    except Exception as e:
        logger.error(f"激活JEPA-DT-MPC控制器失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"激活失败: {str(e)}"
        )

@router.post("/train", response_model=TrainResponse)
async def train_jepa_dtmpc(request: TrainRequest):
    """
    训练JEPA-DT-MPC模型
    """
    global _jepa_dtmpc_controller
    
    # 检查控制器是否已激活
    if _jepa_dtmpc_controller is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JEPA-DT-MPC控制器尚未激活，请先调用/activate端点"
        )
    
    try:
        # 准备训练数据
        training_data = request.training_data
        training_steps = request.training_steps or 100
        
        # 执行训练
        training_result = {}
        losses = []
        
        for step in range(training_steps):
            # 获取当前数据批次
            batch_data = training_data.get(f"batch_{step}")
            if not batch_data:
                # 如果没有按批次提供数据，使用整个数据集
                batch_data = training_data
                
            # 执行单步训练
            loss = _jepa_dtmpc_controller._train_jepa_step()
            if loss is not None:
                losses.append(loss)
        
        # 计算平均损失
        if losses:
            average_loss = np.mean(losses)
            training_result["average_loss"] = float(average_loss)
            training_result["min_loss"] = float(np.min(losses))
            training_result["max_loss"] = float(np.max(losses))
            training_result["losses"] = [float(l) for l in losses]
        
        # 更新训练状态
        _jepa_dtmpc_controller.jepa_trained = True
        
        return TrainResponse(
            success=True,
            message=f"JEPA模型训练完成，共执行{training_steps}步",
            training_result=training_result
        )
        
    except Exception as e:
        logger.error(f"训练JEPA模型失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"训练失败: {str(e)}"
        )

@router.get("/status")
async def get_jepa_dtmpc_status():
    """
    获取JEPA-DT-MPC控制器状态
    """
    global _jepa_dtmpc_controller
    
    if _jepa_dtmpc_controller is None:
        return {
            "success": False,
            "message": "JEPA-DT-MPC控制器尚未激活",
            "controller_status": "inactive"
        }
    
    return {
        "success": True,
        "message": "JEPA-DT-MPC控制器状态",
        "controller_status": {
            "jepa_enabled": _jepa_dtmpc_controller.jepa_enabled,
            "jepa_trained": _jepa_dtmpc_controller.jepa_trained,
            "jepa_current_training_step": _jepa_dtmpc_controller.jepa_current_training_step,
            "jepa_training_steps": _jepa_dtmpc_controller.jepa_training_steps,
            "mpc_state": {
                "prediction_horizon": _jepa_dtmpc_controller.mpc_core.Np,
                "control_horizon": _jepa_dtmpc_controller.mpc_core.Nc,
                "current_state": _jepa_dtmpc_controller.current_state.tolist()
            }
        }
    }

@router.get("/prediction")
async def get_jepa_prediction():
    """
    获取JEPA-DT-MPC预测结果
    """
    global _jepa_dtmpc_controller
    
    # 检查控制器是否已激活
    if _jepa_dtmpc_controller is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JEPA-DT-MPC控制器尚未激活，请先调用/activate端点"
        )
    
    try:
        # 执行预测 - 使用step()方法
        prediction_result = _jepa_dtmpc_controller.step()
        
        # 检查prediction_result是否包含必要的字段
        if not isinstance(prediction_result, dict):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"预测结果格式错误: {type(prediction_result)}"
            )
        
        if "cv_prediction" not in prediction_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"预测结果缺少cv_prediction字段: {prediction_result.keys()}"
            )
        
        # 构造响应数据
        response_data = {
            "cv_prediction": prediction_result["cv_prediction"],  # step()中已经是tolist()
            "jepa_prediction": prediction_result.get("jepa_prediction", []),  # 可能不存在
            "fused_prediction": prediction_result["cv_prediction"],  # 使用融合后的预测结果
            "energy": float(prediction_result.get("jepa_energy", 0.0)),  # 对应jepa_energy
            "weight": float(prediction_result.get("jepa_weight", 0.5)),  # 对应jepa_weight
            "timestamp": time.time()  # 生成当前时间戳
        }
        
        return {
            "success": True,
            "data": response_data,
            "message": "获取JEPA预测结果成功"
        }
    
    except Exception as e:
        logger.error(f"获取jap预测结果失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取预测结果失败: {str(e)}"
        )
