"""
模型管理器服务
提供AI模型的管理、训练、推理等核心功能
"""

import os
import json
import hashlib
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class ModelManager:
    """模型管理器类"""
    
    def __init__(self, model_storage_path: str = "./models"):
        """初始化模型管理器"""
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(exist_ok=True)
        
        # 模型缓存
        self.model_cache = {}
        self.model_metadata = {}  # 存储模型元数据，键为 model_base_id
        self.model_versions = {}   # 存储模型版本信息，键为 model_base_id
        
        # 训练任务管理
        self.training_tasks = {}
        
    async def initialize(self):
        """初始化模型管理器"""
        try:
            # 加载已存在的模型元数据
            await self._load_model_metadata()
            
            # 预加载常用模型
            await self._preload_common_models()
            
            return {"success": True, "message": "模型管理器初始化成功"}
        except Exception as e:
            return {"success": False, "error": f"初始化失败: {str(e)}"}
    
    async def _load_model_metadata(self):
        """加载模型元数据"""
        metadata_file = self.model_storage_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 兼容旧版本数据
                if 'models' in data:
                    self.model_metadata = data['models']
                else:
                    self.model_metadata = data
                
                # 加载版本信息
                if 'versions' in data:
                    self.model_versions = data['versions']
                else:
                    # 为旧数据创建版本信息
                    self.model_versions = {}
                    for model_id, metadata in self.model_metadata.items():
                        base_id = self._get_base_id(model_id)
                        if base_id not in self.model_versions:
                            self.model_versions[base_id] = []
                        self.model_versions[base_id].append({
                            'model_id': model_id,
                            'version': metadata.get('version', '1.0.0'),
                            'created_at': metadata.get('created_at'),
                            'status': metadata.get('status', 'ready')
                        })
    
    async def _save_model_metadata(self):
        """保存模型元数据"""
        metadata_file = self.model_storage_path / "metadata.json"
        data = {
            'models': self.model_metadata,
            'versions': self.model_versions
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def save_model_metadata(self):
        """保存模型元数据的公共方法"""
        await self._save_model_metadata()
    
    async def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        try:
            total_models = len(self.model_metadata)
            loaded_models = len(self.model_cache)
            training_tasks = len(self.training_tasks)
            
            # 按类型统计
            type_stats = {}
            for model in self.model_metadata.values():
                model_type = model.get("type", "unknown")
                if model_type not in type_stats:
                    type_stats[model_type] = 0
                type_stats[model_type] += 1
            
            # 按状态统计
            status_stats = {}
            for model in self.model_metadata.values():
                status = model.get("status", "unknown")
                if status not in status_stats:
                    status_stats[status] = 0
                status_stats[status] += 1
            
            # 按框架统计
            framework_stats = {}
            for model in self.model_metadata.values():
                framework = model.get("framework", "unknown")
                if framework not in framework_stats:
                    framework_stats[framework] = 0
                framework_stats[framework] += 1
            
            return {
                "success": True,
                "statistics": {
                    "total_models": total_models,
                    "loaded_models": loaded_models,
                    "training_tasks": training_tasks,
                    "by_type": type_stats,
                    "by_status": status_stats,
                    "by_framework": framework_stats
                }
            }
        except Exception as e:
            return {"success": False, "error": f"获取统计信息失败: {str(e)}"}
    
    async def search_models(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """搜索模型"""
        try:
            filters = filters or {}
            results = []
            
            for model_id, metadata in self.model_metadata.items():
                # 检查是否匹配搜索条件
                matches = False
                
                # 文本搜索
                if query:
                    query_lower = query.lower()
                    if (query_lower in model_id.lower() or
                        query_lower in metadata.get("name", "").lower() or
                        query_lower in metadata.get("description", "").lower() or
                        query_lower in metadata.get("type", "").lower()):
                        matches = True
                else:
                    matches = True  # 如果没有查询条件，匹配所有
                
                # 过滤器匹配
                if matches and filters:
                    for filter_key, filter_value in filters.items():
                        if filter_key in metadata:
                            if metadata[filter_key] != filter_value:
                                matches = False
                                break
                        else:
                            matches = False
                            break
                
                if matches:
                    results.append(metadata)
            
            return {
                "success": True,
                "results": results,
                "total_count": len(results),
                "query": query,
                "filters": filters
            }
        except Exception as e:
            return {"success": False, "error": f"搜索模型失败: {str(e)}"}
    
    async def export_model(self, model_id: str, format: str = "onnx") -> Dict[str, Any]:
        """导出模型到指定格式"""
        try:
            if model_id not in self.model_metadata:
                return {"success": False, "error": "模型不存在"}
            
            model_record = self.model_metadata[model_id]
            
            # 模拟导出过程
            await asyncio.sleep(0.5)  # 模拟导出延迟
            
            # 生成导出文件路径
            export_filename = f"{model_id}.{format}"
            export_path = str(self.model_storage_path / export_filename)
            
            # 模拟导出结果
            export_info = {
                "model_id": model_id,
                "format": format,
                "export_path": export_path,
                "file_size": 1024 * 1024,  # 模拟1MB文件大小
                "exported_at": datetime.now().isoformat(),
                "compatibility": {
                    "framework": model_record.get("framework", "pytorch"),
                    "version": "1.0.0",
                    "supported_platforms": ["cpu", "gpu"]
                }
            }
            
            return {
                "success": True,
                "message": f"模型导出成功，格式: {format}",
                "export_info": export_info
            }
        except Exception as e:
            return {"success": False, "error": f"导出模型失败: {str(e)}"}
    
    async def import_model_from_file(self, file_path: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """从文件导入模型"""
        try:
            import os
            
            if not os.path.exists(file_path):
                return {"success": False, "error": "文件不存在"}
            
            # 生成模型ID
            model_id = model_info.get("model_id") or f"imported_{os.path.basename(file_path).split('.')[0]}"
            
            # 检查模型是否已存在
            if model_id in self.model_metadata:
                return {"success": False, "error": "模型已存在"}
            
            # 获取文件信息
            file_size = os.path.getsize(file_path)
            
            # 创建模型记录
            model_data = {
                "name": model_info.get("name", model_id),
                "type": model_info.get("type", "transformer"),
                "framework": model_info.get("framework", "pytorch"),
                "version": model_info.get("version", "1.0.0"),
                "status": "imported",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "metadata": model_info.get("metadata", {}),
                "file_path": str(self.model_storage_path / f"{model_id}.pth"),
                "imported_from": file_path,
                "original_size": file_size
            }
            
            # 模拟文件复制过程
            await asyncio.sleep(0.3)  # 模拟文件复制延迟
            
            # 注册模型
            result = await self.register_model(model_id, model_data)
            
            if result["success"]:
                return {
                    "success": True,
                    "model_id": model_id,
                    "message": "模型导入成功",
                    "import_info": {
                        "original_file": file_path,
                        "file_size": file_size,
                        "imported_at": datetime.now().isoformat()
                    }
                }
            else:
                return result
                
        except Exception as e:
            return {"success": False, "error": f"导入模型失败: {str(e)}"}
    
    async def backup_model_metadata(self, backup_path: str = None) -> Dict[str, Any]:
        """备份模型元数据"""
        try:
            if backup_path is None:
                backup_path = str(self.model_storage_path / "backup" / f"metadata_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            # 确保备份目录存在
            backup_dir = os.path.dirname(backup_path)
            os.makedirs(backup_dir, exist_ok=True)
            
            # 创建备份数据
            backup_data = {
                "backup_created_at": datetime.now().isoformat(),
                "total_models": len(self.model_metadata),
                "metadata": self.model_metadata,
                "versions": self.model_versions
            }
            
            # 保存备份文件
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "message": "模型元数据备份成功",
                "backup_info": {
                    "backup_path": backup_path,
                    "backup_size": os.path.getsize(backup_path),
                    "total_models": len(self.model_metadata),
                    "backup_time": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {"success": False, "error": f"备份模型元数据失败: {str(e)}"}
    
    async def restore_model_metadata(self, backup_path: str) -> Dict[str, Any]:
        """从备份恢复模型元数据"""
        try:
            if not os.path.exists(backup_path):
                return {"success": False, "error": "备份文件不存在"}
            
            # 读取备份数据
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # 恢复元数据
            self.model_metadata = backup_data.get("metadata", {})
            self.model_versions = backup_data.get("versions", {})
            
            # 保存恢复后的元数据
            await self._save_model_metadata()
            
            return {
                "success": True,
                "message": "模型元数据恢复成功",
                "restore_info": {
                    "backup_path": backup_path,
                    "restored_models": len(self.model_metadata),
                    "restore_time": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {"success": False, "error": f"恢复模型元数据失败: {str(e)}"}
    
    def _get_base_id(self, model_id: str) -> str:
        """从完整模型ID中提取基础模型ID"""
        # 假设模型ID格式为 base_id_vX.Y.Z
        parts = model_id.rsplit('_v', 1)
        if len(parts) == 2:
            return parts[0]
        return model_id
    
    def _get_next_version(self, base_id: str) -> str:
        """获取下一个版本号"""
        if base_id not in self.model_versions or not self.model_versions[base_id]:
            return "1"
        
        # 按版本号排序（支持纯数字版本号）
        def version_key(v):
            try:
                return int(v['version'])
            except ValueError:
                # 兼容旧的语义化版本号
                return tuple(map(int, v['version'].split('.')))
        
        versions = sorted(self.model_versions[base_id], key=version_key)
        latest_version = versions[-1]['version']
        
        # 生成下一个版本号 (简单的递增整数版本号)
        try:
            next_version = int(latest_version) + 1
            return str(next_version)
        except ValueError:
            # 兼容旧的语义化版本号
            major, minor, patch = map(int, latest_version.split('.'))
            return f"{major}.{minor}.{patch + 1}"
    
    async def _preload_common_models(self):
        """预加载常用模型"""
        # 这里可以预加载一些基础模型
        common_models = [
            "agriculture_classification_v1",
            "resource_optimization_v1", 
            "risk_assessment_v1"
        ]
        
        for model_id in common_models:
            if model_id in self.model_metadata:
                await self.load_model(model_id)
    
    async def register_model(self, model_id: str, model_data: Dict[str, Any], is_new_version: bool = False) -> Dict[str, Any]:
        """注册新模型"""
        try:
            base_id = self._get_base_id(model_id)
            version = model_data.get("version")
            
            if is_new_version:
                # 创建新版本
                # 对于版本创建，model_id参数是原始模型的ID，不应该被当作带版本号的ID来处理
                original_model_id = model_id
                if original_model_id not in self.model_metadata:
                    return {"success": False, "error": "原始模型不存在"}
                
                # 获取原始模型的base_id用于版本管理
                base_id = self._get_base_id(original_model_id)
                
                # 生成新版本号
                if not version:
                    version = self._get_next_version(base_id)
                
                # 创建带版本号的模型ID
                new_model_id = f"{base_id}_v{version}"
                
                # 确保新版本模型ID不存在
                if new_model_id in self.model_metadata:
                    return {"success": False, "error": "该版本模型已存在"}
                
                # 创建模型记录
                model_record = {
                    "model_id": new_model_id,
                    "name": model_data.get("name", self.model_metadata[original_model_id]["name"]),
                    "type": model_data.get("type", self.model_metadata[original_model_id]["type"]),
                    "framework": model_data.get("framework", self.model_metadata[original_model_id]["framework"]),
                    "version": version,
                    "status": "registered",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "metadata": model_data.get("metadata", self.model_metadata[original_model_id]["metadata"]),
                    "file_path": str(self.model_storage_path / f"{new_model_id}.pth"),
                    "hash": hashlib.sha256(new_model_id.encode()).hexdigest()
                }
                
                # 保存模型文件（如果有）
                if "model_file" in model_data:
                    model_file_path = self.model_storage_path / f"{new_model_id}.pth"
                    with open(model_file_path, 'wb') as f:
                        f.write(model_data["model_file"])
                
                # 更新元数据
                self.model_metadata[new_model_id] = model_record
                
                # 更新版本信息
                if base_id not in self.model_versions:
                    self.model_versions[base_id] = []
                self.model_versions[base_id].append({
                    'model_id': new_model_id,
                    'version': version,
                    'created_at': model_record['created_at'],
                    'status': model_record['status']
                })
                
                await self._save_model_metadata()
                
                return {
                    "success": True,
                    "model_id": new_model_id,
                    "message": "新版本模型注册成功"
                }
            else:
                # 创建全新模型
                if model_id in self.model_metadata:
                    return {"success": False, "error": "模型已存在"}
                
                # 确定版本号
                if not version:
                    version = "1.0.0"
                
                # 创建模型记录
                model_record = {
                    "model_id": model_id,
                    "name": model_data.get("name", model_id),
                    "type": model_data.get("type", "classification"),
                    "framework": model_data.get("framework", "pytorch"),
                    "version": version,
                    "status": "registered",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "metadata": model_data.get("metadata", {}),
                    "file_path": str(self.model_storage_path / f"{model_id}.pth"),
                    "hash": hashlib.sha256(model_id.encode()).hexdigest()
                }
                
                # 保存模型文件（如果有）
                if "model_file" in model_data:
                    model_file_path = self.model_storage_path / f"{model_id}.pth"
                    with open(model_file_path, 'wb') as f:
                        f.write(model_data["model_file"])
                
                # 更新元数据
                self.model_metadata[model_id] = model_record
                
                # 更新版本信息
                if base_id not in self.model_versions:
                    self.model_versions[base_id] = []
                self.model_versions[base_id].append({
                    'model_id': model_id,
                    'version': version,
                    'created_at': model_record['created_at'],
                    'status': model_record['status']
                })
                
                await self._save_model_metadata()
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "message": "模型注册成功"
                }
        except Exception as e:
            return {"success": False, "error": f"模型注册失败: {str(e)}"}
    
    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """加载模型到内存"""
        try:
            if model_id not in self.model_metadata:
                # 调试信息：打印可用的模型列表
                print(f"DEBUG: 请求的模型ID {model_id} 不存在")
                print(f"DEBUG: 可用的模型ID: {list(self.model_metadata.keys())}")
                return {"success": False, "error": "模型不存在"}
            
            model_record = self.model_metadata[model_id]
            
            # 如果模型已在缓存中，直接返回
            if model_id in self.model_cache:
                return {
                    "success": True,
                    "model_id": model_id,
                    "message": "模型已加载",
                    "from_cache": True,
                    "model": self.model_cache[model_id],
                    "metadata": model_record
                }
            
            # 模拟模型加载过程
            # 在实际应用中，这里应该加载实际的模型文件
            # 例如：model = torch.load(model_record["file_path"])
            
            # 模拟加载延迟
            await asyncio.sleep(0.1)
            
            # 创建模拟模型对象
            model_obj = {
                "loaded_at": datetime.now().isoformat(),
                "model_type": model_record["type"],
                "status": "loaded",
                "params": {"params": {}}  # 模拟模型参数结构，兼容InferenceEngine
            }
            
            # 将模型添加到缓存
            self.model_cache[model_id] = model_obj
            
            return {
                "success": True,
                "model_id": model_id,
                "message": "模型加载成功",
                "from_cache": False,
                "model": model_obj,
                "metadata": model_record
            }
        except Exception as e:
            return {"success": False, "error": f"模型加载失败: {str(e)}"}
    
    async def unload_model(self, model_id: str) -> Dict[str, Any]:
        """从内存卸载模型"""
        try:
            if model_id not in self.model_cache:
                return {"success": False, "error": "模型未加载"}
            
            # 从缓存中移除模型
            del self.model_cache[model_id]
            
            return {
                "success": True,
                "model_id": model_id,
                "message": "模型卸载成功"
            }
        except Exception as e:
            return {"success": False, "error": f"模型卸载失败: {str(e)}"}
    
    async def predict(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用模型进行预测"""
        try:
            # 确保模型已加载
            load_result = await self.load_model(model_id)
            if not load_result["success"]:
                return load_result
            
            # 模拟预测过程
            await asyncio.sleep(0.05)  # 模拟预测延迟
            
            # 根据模型类型生成预测结果
            model_record = self.model_metadata[model_id]
            model_type = model_record["type"]
            
            if model_type == "classification":
                prediction = self._simulate_classification(input_data)
            elif model_type == "regression":
                prediction = self._simulate_regression(input_data)
            elif model_type == "optimization":
                prediction = self._simulate_optimization(input_data)
            else:
                prediction = {"result": "unknown", "confidence": 0.5}
            
            return {
                "success": True,
                "model_id": model_id,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat(),
                "processing_time": "0.05s"
            }
        except Exception as e:
            return {"success": False, "error": f"预测失败: {str(e)}"}
    
    def _simulate_classification(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟分类预测"""
        features = input_data.get("features", {})
        
        # 简单的模拟分类逻辑
        if "temperature" in features and "humidity" in features:
            temp = features["temperature"]
            humidity = features["humidity"]
            
            if temp > 25 and humidity > 60:
                return {"class": "适合种植", "confidence": 0.85}
            else:
                return {"class": "需要调整", "confidence": 0.75}
        
        return {"class": "未知", "confidence": 0.5}
    
    def _simulate_regression(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟回归预测"""
        features = input_data.get("features", {})
        
        # 简单的模拟回归逻辑
        prediction = 0.0
        if "area" in features:
            prediction = features["area"] * 0.8
        
        return {"predicted_value": prediction, "confidence": 0.9}
    
    def _simulate_optimization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟优化预测"""
        resources = input_data.get("resources", {})
        constraints = input_data.get("constraints", {})
        
        # 简单的资源优化模拟
        optimized_allocation = {}
        for resource, amount in resources.items():
            optimized_allocation[resource] = amount * 0.9  # 优化为原资源的90%
        
        return {
            "optimized_allocation": optimized_allocation,
            "efficiency_improvement": 0.1,
            "confidence": 0.88
        }
    
    async def start_training(self, model_id: str, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """开始模型训练"""
        try:
            if model_id in self.training_tasks:
                return {"success": False, "error": "模型正在训练中"}
            
            # 创建训练任务
            task_id = f"{model_id}_{int(datetime.now().timestamp())}"
            
            # 模拟训练过程（异步执行）
            async def training_task():
                # 模拟训练进度更新（10个步骤）
                total_steps = 10
                for step in range(total_steps + 1):
                    # 更新进度
                    self.training_tasks[task_id]["progress"] = step * 10  # 0-100%
                    self.training_tasks[task_id]["current_step"] = step
                    self.training_tasks[task_id]["total_steps"] = total_steps
                    
                    # 根据不同阶段更新状态信息
                    if step < 3:
                        self.training_tasks[task_id]["stage"] = "数据准备"
                    elif step < 7:
                        self.training_tasks[task_id]["stage"] = "模型训练"
                    else:
                        self.training_tasks[task_id]["stage"] = "模型评估"
                    
                    # 更新模型状态
                    if model_id in self.model_metadata:
                        self.model_metadata[model_id]["status"] = "training"
                        self.model_metadata[model_id]["updated_at"] = datetime.now().isoformat()
                        await self._save_model_metadata()
                    
                    # 模拟每步训练时间
                    if step < total_steps:
                        await asyncio.sleep(0.5)  # 每步0.5秒，总共5秒
                
                # 训练完成后更新状态
                if model_id in self.model_metadata:
                    self.model_metadata[model_id]["status"] = "trained"
                    self.model_metadata[model_id]["updated_at"] = datetime.now().isoformat()
                    # 模拟训练完成后更新模型指标
                    if "metrics" not in self.model_metadata[model_id]:
                        self.model_metadata[model_id]["metrics"] = {}
                    self.model_metadata[model_id]["metrics"]["accuracy"] = 0.85
                    self.model_metadata[model_id]["metrics"]["loss"] = 0.23
                    self.model_metadata[model_id]["accuracy"] = 0.85
                    await self._save_model_metadata()
                
                # 更新训练任务状态
                self.training_tasks[task_id]["status"] = "completed"
                self.training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
            
            # 启动训练任务
            task = asyncio.create_task(training_task())
            self.training_tasks[task_id] = {
                "task": task,
                "model_id": model_id,
                "started_at": datetime.now().isoformat(),
                "status": "training",
                "progress": 0,
                "current_step": 0,
                "total_steps": 10,
                "stage": "初始化",
                "training_data": training_data
            }
            
            return {
                "success": True,
                "task_id": task_id,
                "model_id": model_id,
                "message": "训练任务已启动",
                "status": "training",
                "progress": 0,
                "stage": "初始化"
            }
        except Exception as e:
            return {"success": False, "error": f"训练启动失败: {str(e)}"}
    
    async def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """获取训练状态"""
        try:
            if task_id not in self.training_tasks:
                return {"success": False, "error": "训练任务不存在"}
            
            task_info = self.training_tasks[task_id]
            
            # 检查任务是否完成
            if task_info["task"].done() and task_info["status"] != "completed":
                task_info["status"] = "completed"
                task_info["completed_at"] = datetime.now().isoformat()
            
            # 构造详细的状态信息
            status_response = {
                "success": True,
                "task_id": task_id,
                "model_id": task_info["model_id"],
                "status": task_info["status"],
                "progress": task_info.get("progress", 0),
                "stage": task_info.get("stage", "未知"),
                "current_step": task_info.get("current_step", 0),
                "total_steps": task_info.get("total_steps", 0),
                "started_at": task_info["started_at"],
                "completed_at": task_info.get("completed_at")
            }
            
            # 如果训练完成，添加模型指标信息
            if task_info["status"] == "completed":
                model_id = task_info["model_id"]
                if model_id in self.model_metadata:
                    status_response["metrics"] = self.model_metadata[model_id].get("metrics", {})
                    status_response["model_status"] = self.model_metadata[model_id].get("status", "unknown")
            
            return status_response
        except Exception as e:
            return {"success": False, "error": f"获取训练状态失败: {str(e)}"}
    
    async def list_models(self) -> Dict[str, Any]:
        """列出所有模型"""
        try:
            models = list(self.model_metadata.values())
            
            # 确保每个模型包含大模型和量化相关字段
            for model in models:
                if "framework" not in model:
                    model["framework"] = "pytorch"
                if "is_pretrained" not in model:
                    model["is_pretrained"] = False
                if "pretrained_source" not in model:
                    model["pretrained_source"] = None
                if "model_format" not in model:
                    model["model_format"] = "pytorch"
                if "quantization" not in model:
                    model["quantization"] = {
                        "enabled": False
                    }
            
            return {
                "success": True,
                "models": models,
                "total_count": len(models),
                "loaded_count": len(self.model_cache)
            }
        except Exception as e:
            return {"success": False, "error": f"获取模型列表失败: {str(e)}"}
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        try:
            if model_id not in self.model_metadata:
                return {"success": False, "error": "模型不存在"}
            
            model_info = self.model_metadata[model_id].copy()
            
            # 确保包含大模型相关字段
            if "framework" not in model_info:
                model_info["framework"] = "pytorch"
            if "is_pretrained" not in model_info:
                model_info["is_pretrained"] = False
            if "pretrained_source" not in model_info:
                model_info["pretrained_source"] = None
            if "model_format" not in model_info:
                model_info["model_format"] = "pytorch"
            if "quantization" not in model_info:
                model_info["quantization"] = {
                    "enabled": False
                }
            
            # 添加加载状态
            model_info["is_loaded"] = model_id in self.model_cache
            
            return {
                "success": True,
                "model": model_info
            }
        except Exception as e:
            return {"success": False, "error": f"获取模型信息失败: {str(e)}"}
    
    async def delete_model(self, model_id: str) -> Dict[str, Any]:
        """删除模型"""
        try:
            if model_id not in self.model_metadata:
                return {"success": False, "error": "模型不存在"}
            
            # 如果模型已加载，先卸载
            if model_id in self.model_cache:
                await self.unload_model(model_id)
            
            # 删除模型文件
            model_file = self.model_storage_path / f"{model_id}.pth"
            if model_file.exists():
                model_file.unlink()
            
            # 删除元数据
            del self.model_metadata[model_id]
            await self._save_model_metadata()
            
            return {
                "success": True,
                "model_id": model_id,
                "message": "模型删除成功"
            }
        except Exception as e:
            return {"success": False, "error": f"模型删除失败: {str(e)}"}
    
    async def quantize_model(self, model_id: str, quantization_type: str = "int8") -> Dict[str, Any]:
        """对模型进行量化处理"""
        try:
            if model_id not in self.model_metadata:
                return {"success": False, "error": "模型不存在"}
            
            base_id = self._get_base_id(model_id)
            original_model = await self.load_model(model_id)
            
            if not original_model["success"]:
                return original_model
            
            # 检查模型是否支持量化（基于模型类型）
            model_type = self.model_metadata[model_id].get("framework", "")
            if model_type != "transformer":
                return {"success": False, "error": "该模型不支持量化"}
            
            # 生成量化模型的版本信息
            version = self._get_next_version(base_id)
            quantized_model_id = f"{base_id}_v{version}_quantized_{quantization_type}"
            
            # 模拟量化过程
            await asyncio.sleep(1.0)  # 模拟量化延迟
            
            # 创建量化模型记录
            quantized_record = self.model_metadata[model_id].copy()
            quantized_record.update({
                "model_id": quantized_model_id,
                "version": version,
                "status": "quantized",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "file_path": str(self.model_storage_path / f"{quantized_model_id}.pth"),
                "hash": hashlib.sha256(quantized_model_id.encode()).hexdigest(),
                "quantization": {
                    "enabled": True,
                    "type": quantization_type,
                    "original_model_id": model_id,
                    "compression_ratio": 0.5,  # 模拟压缩率
                    "accuracy_drop": 0.01,     # 模拟精度下降
                    "inference_speedup": 2.0   # 模拟推理加速倍数
                }
            })
            
            # 更新元数据
            self.model_metadata[quantized_model_id] = quantized_record
            
            # 更新版本信息
            if base_id not in self.model_versions:
                self.model_versions[base_id] = []
            self.model_versions[base_id].append({
                'model_id': quantized_model_id,
                'version': version,
                'created_at': quantized_record['created_at'],
                'status': quantized_record['status']
            })
            
            await self._save_model_metadata()
            
            return {
                "success": True,
                "model_id": quantized_model_id,
                "message": f"模型量化成功，量化类型: {quantization_type}",
                "quantization": quantized_record["quantization"]
            }
        except Exception as e:
            return {"success": False, "error": f"模型量化失败: {str(e)}"}
    
    async def predict_with_quantization(self, model_id: str, input_data: Dict[str, Any], quantization_type: str = "int8") -> Dict[str, Any]:
        """使用量化模型进行预测"""
        try:
            # 确保模型已加载
            load_result = await self.load_model(model_id)
            if not load_result["success"]:
                return load_result
            
            # 检查模型是否支持量化推理（基于量化状态）
            if not self.model_metadata[model_id].get("quantization", {}).get("enabled", False):
                return {"success": False, "error": "该模型不支持量化推理"}
            
            # 模拟量化预测过程
            await asyncio.sleep(0.02)  # 模拟量化推理延迟（比普通推理快）
            
            # 根据模型类型生成预测结果
            model_record = self.model_metadata[model_id]
            model_type = model_record["type"]
            
            # 模拟量化预测结果（与普通预测类似，但速度更快）
            if model_type == "classification":
                prediction = self._simulate_classification(input_data)
            elif model_type == "regression":
                prediction = self._simulate_regression(input_data)
            elif model_type == "optimization":
                prediction = self._simulate_optimization(input_data)
            elif model_type == "transformer":
                # 大模型预测模拟
                prediction = {
                    "generated_text": "这是一个使用量化模型生成的文本示例。",
                    "confidence": 0.82,
                    "token_count": 25
                }
            else:
                prediction = {"result": "unknown", "confidence": 0.5}
            
            return {
                "success": True,
                "model_id": model_id,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat(),
                "processing_time": "0.02s",  # 量化推理速度更快
                "quantization_type": quantization_type
            }
        except Exception as e:
            return {"success": False, "error": f"量化预测失败: {str(e)}"}
    
    async def get_model_versions(self, model_id: str) -> Dict[str, Any]:
        """获取模型的版本历史"""
        try:
            # 获取基础模型ID
            base_id = self._get_base_id(model_id)
            
            if base_id not in self.model_versions:
                return {"success": False, "error": "模型不存在或没有版本历史"}
            
            # 获取所有版本信息
            versions = self.model_versions[base_id]
            
            # 为每个版本获取详细信息
            version_details = []
            for version_info in versions:
                ver_model_id = version_info["model_id"]
                if ver_model_id in self.model_metadata:
                    version_details.append({
                        **version_info,
                        "metadata": self.model_metadata[ver_model_id]
                    })
            
            # 按版本号排序（从新到旧）
            version_details.sort(
                key=lambda v: tuple(map(int, v["version"].split('.'))),
                reverse=True
            )
            
            return {
                "success": True,
                "versions": version_details,
                "base_id": base_id
            }
        except Exception as e:
            return {"success": False, "error": f"获取模型版本历史失败: {str(e)}"}
    
    async def load_pretrained_model(self, pretrained_info: Dict[str, Any]) -> Dict[str, Any]:
        """加载预训练模型
        
        支持的格式：
        - Hugging Face模型 (model_format='huggingface')
        - ONNX模型 (model_format='onnx')
        - TensorFlow SavedModel (model_format='tensorflow')
        - PyTorch模型 (model_format='pytorch')
        
        参数:
            pretrained_info: 预训练模型信息，包含以下字段：
                - model_name_or_path: 模型名称或本地路径
                - model_format: 模型格式
                - model_type: 模型类型 (classification, regression, transformer等)
                - name: 模型显示名称
                - metadata: 额外的元数据
        
        返回:
            模型注册结果
        """
        try:
            # 解析预训练模型信息
            model_name_or_path = pretrained_info.get('model_name_or_path')
            model_format = pretrained_info.get('model_format', 'huggingface')
            model_type = pretrained_info.get('model_type', 'transformer')
            name = pretrained_info.get('name', model_name_or_path)
            metadata = pretrained_info.get('metadata', {})
            
            if not model_name_or_path:
                return {"success": False, "error": "模型名称或路径不能为空"}
            
            # 生成唯一的模型ID
            model_id = f"pretrained_{model_format}_{hashlib.sha256(model_name_or_path.encode()).hexdigest()[:8]}"
            base_id = model_id
            
            # 加载预训练模型
            model_data = {
                "name": name,
                "type": model_type,
                "framework": model_format,
                "version": "1.0.0",
                "status": "pretrained",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "metadata": {
                    "pretrained_source": model_name_or_path,
                    "model_format": model_format,
                    **metadata
                },
                "file_path": str(self.model_storage_path / f"{model_id}.{model_format}")
            }
            
            # 模拟不同格式的模型加载过程
            if model_format == 'huggingface':
                # 模拟从Hugging Face下载和加载模型
                await asyncio.sleep(1.0)  # 模拟下载延迟
                model_data["metadata"]["model_type"] = "huggingface"
                
            elif model_format == 'onnx':
                # 模拟加载ONNX模型
                await asyncio.sleep(0.5)  # 模拟加载延迟
                model_data["metadata"]["model_type"] = "onnx"
                
            elif model_format == 'tensorflow':
                # 模拟加载TensorFlow模型
                await asyncio.sleep(0.7)
                model_data["metadata"]["model_type"] = "tensorflow"
                
            elif model_format == 'pytorch':
                # 模拟加载PyTorch模型
                await asyncio.sleep(0.6)
                model_data["metadata"]["model_type"] = "pytorch"
            
            # 注册模型
            register_result = await self.register_model(model_id, model_data, is_new_version=False)
            
            if register_result["success"]:
                # 将预训练模型添加到模型缓存
                self.model_cache[model_id] = {
                    "loaded_at": datetime.now().isoformat(),
                    "model_type": model_type,
                    "status": "loaded",
                    "format": model_format
                }
                
            return {
                "success": True,
                "model_id": register_result["model_id"],
                "message": f"预训练模型加载成功",
                "model_format": model_format
            }
            
        except Exception as e:
            return {"success": False, "error": f"预训练模型加载失败: {str(e)}"}

    async def close(self):
        """关闭模型管理器"""
        # 清理资源
        self.model_cache.clear()
        
        # 取消所有训练任务
        for task_id, task_info in self.training_tasks.items():
            if not task_info["task"].done():
                task_info["task"].cancel()
        
        self.training_tasks.clear()
        
        print("模型管理器已关闭")
    
    def create_model(self, model_type: str, model_id: str, hyperparameters: dict, 
                   description: str, file_path: str = None, name: str = None, 
                   status: str = "ready", version: str = "v1.0.0") -> "ModelMetadata":
        """创建模型的方法，用于API路由调用"""
        # 定义内部ModelMetadata类用于兼容API路由返回格式
        class ModelMetadata:
            def __init__(self, data: dict):
                self.data = data
                
            def to_dict(self):
                return self.data
        
        # 生成模型名称
        model_name = name or model_id
        
        # 创建模型元数据
        model_metadata = {
            "model_id": model_id,
            "name": model_name,
            "type": model_type,
            "model_type": model_type,
            "version": version,
            "description": description,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "hyperparameters": hyperparameters,
            "metrics": {"accuracy": 0.0},
            "size": None,
            "framework": "pytorch",  # 默认框架
            "is_pretrained": False,
            "pretrained_source": None,
            "model_format": "pytorch"  # 默认模型格式
        }
        
        # 如果提供了文件路径，计算文件大小
        if file_path and os.path.exists(file_path):
            try:
                model_size = os.path.getsize(file_path)
                model_metadata["size"] = model_size
            except Exception as e:
                print(f"计算模型大小失败: {str(e)}")
        
        # 保存到模型元数据字典
        self.model_metadata[model_id] = model_metadata
        
        # 保存元数据到文件
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环已运行，使用create_task
            loop.create_task(self._save_model_metadata())
        else:
            # 如果事件循环未运行，直接执行
            loop.run_until_complete(self._save_model_metadata())
        
        # 返回兼容的ModelMetadata对象
        return ModelMetadata(model_metadata)
    
    def update_model_metrics(self, model_id: str, metrics: dict):
        """更新模型指标的同步方法，用于API路由调用"""
        if model_id not in self.model_metadata:
            raise ValueError(f"模型 {model_id} 不存在")
        
        # 更新指标
        if "metrics" not in self.model_metadata[model_id]:
            self.model_metadata[model_id]["metrics"] = {}
        
        self.model_metadata[model_id]["metrics"].update(metrics)
        
        # 如果accuracy在指标中，也单独保存为兼容API
        if "accuracy" in metrics:
            self.model_metadata[model_id]["accuracy"] = metrics["accuracy"]
        
        # 更新时间
        self.model_metadata[model_id]["updated_at"] = datetime.now().isoformat()
        
        # 保存元数据到文件
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环已运行，使用create_task
            loop.create_task(self._save_model_metadata())
        else:
            # 如果事件循环未运行，直接执行
            loop.run_until_complete(self._save_model_metadata())
    
