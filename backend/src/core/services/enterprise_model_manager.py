"""
企业级AI模型管理器
封装核心AI功能，提供统一的企业级AI服务接口
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from .model_manager import ModelManager
from .inference_engine import InferenceEngine, InferenceResult
from .training_service import TrainingService
from src.core.models import EnterpriseAIConfig, AITask, TaskStatus


class EnterpriseModelManager:
    """企业级AI模型管理器
    
    封装核心AI功能，提供统一的企业级AI服务接口，包括：
    1. 模型管理（注册、加载、版本控制）
    2. AI推理（文本生成、图像分类、图像生成）
    3. 模型训练（Transformer、Vision、Diffusion模型）
    4. 企业级AI任务管理
    5. 多模型协同服务
    """
    
    def __init__(self, config: Optional[EnterpriseAIConfig] = None):
        """初始化企业级AI模型管理器
        
        Args:
            config: 企业AI配置
        """
        self.config = config or EnterpriseAIConfig()
        
        # 初始化核心AI服务
        self.model_manager = ModelManager(self.config.model_storage_path)
        self.inference_engine = InferenceEngine(self.model_manager)
        self.training_service = TrainingService(self.model_manager)
        
        # 企业级任务管理
        self.ai_tasks: Dict[str, AITask] = {}
        
        # 模型池管理
        self.model_pool: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> Dict[str, Any]:
        """初始化企业级AI服务
        
        Returns:
            初始化结果
        """
        try:
            # 初始化模型管理器
            model_init_result = await self.model_manager.initialize()
            if not model_init_result["success"]:
                return {
                    "success": False,
                    "error": f"模型管理器初始化失败: {model_init_result['error']}"
                }
            
            # 初始化模型池
            await self._initialize_model_pool()
            
            return {
                "success": True,
                "message": "企业级AI服务初始化成功",
                "model_pool_size": len(self.model_pool),
                "config": self.config.model_dump()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"企业级AI服务初始化失败: {str(e)}"
            }
    
    async def _initialize_model_pool(self) -> None:
        """初始化模型池，预加载常用模型"""
        try:
            # 获取所有可用模型
            model_statistics = await self.model_manager.get_model_statistics()
            if model_statistics["success"]:
                # 根据配置预加载常用模型
                for model_id, metadata in self.model_manager.model_metadata.items():
                    if metadata.get("is_enterprise_model", False):
                        # 预加载企业级模型
                        load_result = await self.model_manager.load_model(model_id)
                        if load_result["success"]:
                            self.model_pool[model_id] = {
                                "model": load_result["model"],
                                "metadata": load_result["metadata"],
                                "last_used": datetime.now(),
                                "usage_count": 0
                            }
        except Exception as e:
            print(f"初始化模型池失败: {str(e)}")
    
    async def create_ai_task(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """创建AI任务
        
        Args:
            task_type: 任务类型 (text_generation, image_classification, image_generation, model_training)
            params: 任务参数
            
        Returns:
            任务创建结果
        """
        try:
            task_id = f"ai_task_{datetime.now().timestamp()}_{id(self)}"
            
            # 创建任务
            ai_task = AITask(
                task_id=task_id,
                task_type=task_type,
                params=params,
                status=TaskStatus.PENDING,
                created_at=datetime.now()
            )
            
            self.ai_tasks[task_id] = ai_task
            
            # 异步执行任务
            await self._execute_ai_task(task_id)
            
            return {
                "success": True,
                "task_id": task_id,
                "message": "AI任务已创建并开始执行"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"创建AI任务失败: {str(e)}"
            }
    
    async def _execute_ai_task(self, task_id: str) -> None:
        """执行AI任务
        
        Args:
            task_id: 任务ID
        """
        try:
            task = self.ai_tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            result = None
            
            # 根据任务类型执行不同的AI功能
            if task.task_type == "text_generation":
                result = await self._execute_text_generation_task(task.params)
            elif task.task_type == "image_classification":
                result = await self._execute_image_classification_task(task.params)
            elif task.task_type == "image_generation":
                result = await self._execute_image_generation_task(task.params)
            elif task.task_type == "model_training":
                result = await self._execute_model_training_task(task.params)
            else:
                raise ValueError(f"未知的任务类型: {task.task_type}")
            
            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            task.metadata = {
                "processing_time": (task.completed_at - task.started_at).total_seconds()
            }
            
        except Exception as e:
            # 更新任务失败状态
            task = self.ai_tasks[task_id]
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
    
    async def _execute_text_generation_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行文本生成任务
        
        Args:
            params: 文本生成参数
            
        Returns:
            文本生成结果
        """
        # 调用推理引擎进行文本生成
        result = await self.inference_engine.text_generation(
            model_id=params["model_id"],
            prompt=params["prompt"],
            max_length=params.get("max_length", 100),
            temperature=params.get("temperature", 1.0),
            repetition_penalty=params.get("repetition_penalty", 1.0),
            num_return_sequences=params.get("num_return_sequences", 1),
            beam_search=params.get("beam_search", False),
            beam_width=params.get("beam_width", 5),
            early_stopping=params.get("early_stopping", True),
            no_repeat_ngram_size=params.get("no_repeat_ngram_size", 0),
            do_sample=params.get("do_sample", False),
            top_p=params.get("top_p", 1.0),
            top_k=params.get("top_k", 0)
        )
        
        return result.model_dump()
    
    async def _execute_image_classification_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行图像分类任务
        
        Args:
            params: 图像分类参数
            
        Returns:
            图像分类结果
        """
        # 调用推理引擎进行图像分类
        result = await self.inference_engine.image_classification(
            model_id=params["model_id"],
            image=params["image"],
            top_k=params.get("top_k", 5)
        )
        
        return result.model_dump()
    
    async def _execute_image_generation_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行图像生成任务
        
        Args:
            params: 图像生成参数
            
        Returns:
            图像生成结果
        """
        # 调用推理引擎进行图像生成
        result = await self.inference_engine.image_generation(
            model_id=params["model_id"],
            num_samples=params.get("num_samples", 1),
            image_size=params.get("image_size", 256),
            guidance_scale=params.get("guidance_scale", 7.5)
        )
        
        return result.model_dump()
    
    async def _execute_model_training_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行模型训练任务
        
        Args:
            params: 模型训练参数
            
        Returns:
            模型训练结果
        """
        # 根据模型类型执行不同的训练逻辑
        model_type = params.get("model_type", "transformer")
        
        if model_type == "transformer":
            result = self.training_service.train_transformer_model(
                model_id=params["model_id"],
                train_data=params["train_data"],
                train_labels=params["train_labels"],
                val_data=params.get("val_data"),
                val_labels=params.get("val_labels"),
                config=params.get("config"),
                create_new_version=params.get("create_new_version", True)
            )
        elif model_type == "vision":
            result = self.training_service.train_vision_model(
                model_id=params["model_id"],
                train_data=params["train_data"],
                train_labels=params["train_labels"],
                val_data=params.get("val_data"),
                val_labels=params.get("val_labels"),
                config=params.get("config"),
                create_new_version=params.get("create_new_version", True)
            )
        elif model_type == "diffusion":
            result = self.training_service.train_diffusion_model(
                model_id=params["model_id"],
                train_data=params["train_data"],
                config=params.get("config"),
                create_new_version=params.get("create_new_version", True)
            )
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        return result
    
    async def get_ai_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取AI任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态
        """
        try:
            if task_id not in self.ai_tasks:
                return {
                    "success": False,
                    "error": "任务不存在"
                }
            
            task = self.ai_tasks[task_id]
            return {
                "success": True,
                "task_id": task.task_id,
                "status": task.status.value,
                "task_type": task.task_type,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "metadata": task.metadata,
                "result": task.result,
                "error": task.error
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"获取任务状态失败: {str(e)}"
            }
    
    async def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息
        
        Returns:
            模型统计信息
        """
        try:
            # 获取基础模型统计
            base_stats = await self.model_manager.get_model_statistics()
            
            if not base_stats["success"]:
                return base_stats
            
            # 添加企业级统计信息
            enterprise_stats = {
                "model_pool_size": len(self.model_pool),
                "active_tasks": len([t for t in self.ai_tasks.values() if t.status == TaskStatus.RUNNING]),
                "total_tasks": len(self.ai_tasks),
                "model_usage": {model_id: pool_item["usage_count"] for model_id, pool_item in self.model_pool.items()}
            }
            
            base_stats["statistics"]["enterprise"] = enterprise_stats
            
            return base_stats
        except Exception as e:
            return {
                "success": False,
                "error": f"获取模型统计信息失败: {str(e)}"
            }
    
    async def batch_inference(self, task_type: str, batch_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量推理
        
        Args:
            task_type: 任务类型
            batch_params: 批量推理参数列表
            
        Returns:
            批量推理结果
        """
        try:
            results = []
            
            for params in batch_params:
                # 执行单个推理
                if task_type == "text_generation":
                    result = await self.inference_engine.text_generation(**params)
                elif task_type == "image_classification":
                    result = await self.inference_engine.image_classification(**params)
                elif task_type == "image_generation":
                    result = await self.inference_engine.image_generation(**params)
                else:
                    raise ValueError(f"未知的任务类型: {task_type}")
                
                results.append(result.model_dump())
            
            return {
                "success": True,
                "results": results,
                "total_count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"批量推理失败: {str(e)}"
            }
    
    async def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """更新模型元数据
        
        Args:
            model_id: 模型ID
            metadata: 要更新的元数据
            
        Returns:
            更新结果
        """
        try:
            # 调用模型管理器更新元数据
            result = await self.model_manager.update_model_metadata(model_id, metadata)
            
            # 如果模型在池中，更新池中的元数据
            if model_id in self.model_pool:
                self.model_pool[model_id]["metadata"].update(metadata)
            
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"更新模型元数据失败: {str(e)}"
            }
    
    async def shutdown(self) -> Dict[str, Any]:
        """关闭企业级AI服务
        
        Returns:
            关闭结果
        """
        try:
            # 清空模型池
            for model_id in list(self.model_pool.keys()):
                # 卸载模型
                await self.model_manager.unload_model(model_id)
                del self.model_pool[model_id]
            
            # 清空任务
            self.ai_tasks.clear()
            
            return {
                "success": True,
                "message": "企业级AI服务已成功关闭"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"关闭企业级AI服务失败: {str(e)}"
            }
