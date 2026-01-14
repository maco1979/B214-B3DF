"""模型仓库服务

提供完整的模型管理功能，包括：
1. 模型版本管理
2. 模型搜索和筛选
3. 模型部署和监控
4. 模型共享和访问控制
5. 模型元数据管理
"""

import os
import json
import hashlib
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRepository:
    """模型仓库类
    
    提供完整的模型管理功能
    """
    
    def __init__(self, storage_path: str = "./model_repository"):
        """初始化模型仓库"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # 目录结构
        self.models_dir = self.storage_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.storage_path / "repository_metadata.json"
        
        # 内存缓存
        self.repository_metadata = {
            "models": {},
            "versions": {},
            "tags": {},
            "deployments": {},
            "statistics": {
                "total_models": 0,
                "total_versions": 0,
                "total_deployments": 0
            }
        }
        
        # 加载现有元数据
        self._load_metadata()
    
    def _load_metadata(self):
        """加载仓库元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.repository_metadata = json.load(f)
    
    def _save_metadata(self):
        """保存仓库元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.repository_metadata, f, indent=2, ensure_ascii=False)
    
    def add_model(self, model_id: str, model_data: Dict[str, Any], model_file: Optional[bytes] = None) -> Dict[str, Any]:
        """添加模型到仓库
        
        Args:
            model_id: 模型ID
            model_data: 模型数据
            model_file: 模型文件内容
            
        Returns:
            Dict[str, Any]: 添加结果
        """
        try:
            # 检查模型是否已存在
            if model_id in self.repository_metadata["models"]:
                return {"success": False, "error": "模型已存在"}
            
            # 准备模型数据
            model_record = {
                "model_id": model_id,
                "name": model_data.get("name", model_id),
                "type": model_data.get("type", "unknown"),
                "industry": model_data.get("industry", "agriculture"),
                "framework": model_data.get("framework", "unknown"),
                "version": model_data.get("version", "1.0.0"),
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "metadata": model_data.get("metadata", {}),
                "tags": model_data.get("tags", []),
                "metrics": model_data.get("metrics", {}),
                "size": model_data.get("size", 0),
                "is_pretrained": model_data.get("is_pretrained", False),
                "pretrained_source": model_data.get("pretrained_source", None),
                "model_format": model_data.get("model_format", "pytorch"),
                "quantization": model_data.get("quantization", {"enabled": False})
            }
            
            # 保存模型文件（如果提供）
            if model_file:
                model_file_path = self.models_dir / f"{model_id}.pth"
                with open(model_file_path, 'wb') as f:
                    f.write(model_file)
                model_record["file_path"] = str(model_file_path)
                model_record["size"] = len(model_file)
            
            # 添加到仓库
            self.repository_metadata["models"][model_id] = model_record
            
            # 处理版本信息
            base_id = self._get_base_id(model_id)
            if base_id not in self.repository_metadata["versions"]:
                self.repository_metadata["versions"][base_id] = []
            
            self.repository_metadata["versions"][base_id].append({
                "model_id": model_id,
                "version": model_record["version"],
                "created_at": model_record["created_at"],
                "status": model_record["status"]
            })
            
            # 处理标签
            for tag in model_record["tags"]:
                if tag not in self.repository_metadata["tags"]:
                    self.repository_metadata["tags"][tag] = []
                self.repository_metadata["tags"][tag].append(model_id)
            
            # 更新统计信息
            self._update_statistics()
            
            # 保存元数据
            self._save_metadata()
            
            return {
                "success": True,
                "message": "模型添加成功",
                "model_id": model_id,
                "model_record": model_record
            }
        except Exception as e:
            logger.error(f"添加模型失败: {e}")
            return {"success": False, "error": f"添加模型失败: {str(e)}"}
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """更新模型信息"""
        try:
            if model_id not in self.repository_metadata["models"]:
                return {"success": False, "error": "模型不存在"}
            
            model_record = self.repository_metadata["models"][model_id]
            
            # 更新模型数据
            for key, value in updates.items():
                if key in model_record:
                    model_record[key] = value
            
            # 更新时间
            model_record["updated_at"] = datetime.now().isoformat()
            
            # 更新统计信息
            self._update_statistics()
            
            # 保存元数据
            self._save_metadata()
            
            return {
                "success": True,
                "message": "模型更新成功",
                "model_id": model_id,
                "model_record": model_record
            }
        except Exception as e:
            logger.error(f"更新模型失败: {e}")
            return {"success": False, "error": f"更新模型失败: {str(e)}"}
    
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """删除模型"""
        try:
            if model_id not in self.repository_metadata["models"]:
                return {"success": False, "error": "模型不存在"}
            
            model_record = self.repository_metadata["models"][model_id]
            
            # 删除模型文件（如果存在）
            if "file_path" in model_record:
                file_path = Path(model_record["file_path"])
                if file_path.exists():
                    file_path.unlink()
            
            # 从版本信息中删除
            base_id = self._get_base_id(model_id)
            if base_id in self.repository_metadata["versions"]:
                self.repository_metadata["versions"][base_id] = [
                    v for v in self.repository_metadata["versions"][base_id] 
                    if v["model_id"] != model_id
                ]
            
            # 从标签中删除
            for tag in model_record["tags"]:
                if tag in self.repository_metadata["tags"]:
                    self.repository_metadata["tags"][tag] = [
                        mid for mid in self.repository_metadata["tags"][tag] 
                        if mid != model_id
                    ]
            
            # 从模型列表中删除
            del self.repository_metadata["models"][model_id]
            
            # 更新统计信息
            self._update_statistics()
            
            # 保存元数据
            self._save_metadata()
            
            return {
                "success": True,
                "message": "模型删除成功",
                "model_id": model_id
            }
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return {"success": False, "error": f"删除模型失败: {str(e)}"}
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """获取模型详情"""
        try:
            if model_id not in self.repository_metadata["models"]:
                return {"success": False, "error": "模型不存在"}
            
            return {
                "success": True,
                "model": self.repository_metadata["models"][model_id]
            }
        except Exception as e:
            logger.error(f"获取模型失败: {e}")
            return {"success": False, "error": f"获取模型失败: {str(e)}"}
    
    def list_models(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """列出模型"""
        try:
            models = list(self.repository_metadata["models"].values())
            
            # 应用筛选条件
            if filters:
                filtered_models = []
                for model in models:
                    match = True
                    for key, value in filters.items():
                        if key not in model or model[key] != value:
                            match = False
                            break
                    if match:
                        filtered_models.append(model)
                models = filtered_models
            
            return {
                "success": True,
                "models": models,
                "total_count": len(models)
            }
        except Exception as e:
            logger.error(f"列出模型失败: {e}")
            return {"success": False, "error": f"列出模型失败: {str(e)}"}
    
    def search_models(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """搜索模型"""
        try:
            query = query.lower()
            results = []
            
            for model in self.repository_metadata["models"].values():
                # 文本搜索
                matches_query = (
                    query in model["model_id"].lower() or
                    query in model["name"].lower() or
                    query in model["type"].lower() or
                    query in model["industry"].lower() or
                    query in model["framework"].lower()
                )
                
                # 标签搜索
                if not matches_query:
                    matches_query = any(query in tag.lower() for tag in model["tags"])
                
                # 元数据搜索
                if not matches_query and "metadata" in model:
                    for key, value in model["metadata"].items():
                        if query in str(value).lower():
                            matches_query = True
                            break
                
                # 应用筛选条件
                matches_filters = True
                if filters:
                    for key, value in filters.items():
                        if key not in model or model[key] != value:
                            matches_filters = False
                            break
                
                if matches_query and matches_filters:
                    results.append(model)
            
            return {
                "success": True,
                "models": results,
                "total_count": len(results)
            }
        except Exception as e:
            logger.error(f"搜索模型失败: {e}")
            return {"success": False, "error": f"搜索模型失败: {str(e)}"}
    
    def add_model_version(self, model_id: str, version_data: Dict[str, Any]) -> Dict[str, Any]:
        """添加模型版本"""
        try:
            # 检查基础模型是否存在
            if model_id not in self.repository_metadata["models"]:
                return {"success": False, "error": "基础模型不存在"}
            
            base_model = self.repository_metadata["models"][model_id]
            base_id = self._get_base_id(model_id)
            
            # 生成新版本号
            version = version_data.get("version", self._get_next_version(base_id))
            new_model_id = f"{base_id}_v{version}"
            
            # 准备新版本模型数据
            new_model_record = base_model.copy()
            new_model_record.update({
                "model_id": new_model_id,
                "version": version,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                **version_data
            })
            
            # 添加新版本
            self.repository_metadata["models"][new_model_id] = new_model_record
            
            # 更新版本信息
            self.repository_metadata["versions"][base_id].append({
                "model_id": new_model_id,
                "version": version,
                "created_at": new_model_record["created_at"],
                "status": new_model_record["status"]
            })
            
            # 更新统计信息
            self._update_statistics()
            
            # 保存元数据
            self._save_metadata()
            
            return {
                "success": True,
                "message": "模型版本添加成功",
                "model_id": new_model_id,
                "version": version
            }
        except Exception as e:
            logger.error(f"添加模型版本失败: {e}")
            return {"success": False, "error": f"添加模型版本失败: {str(e)}"}
    
    def get_model_versions(self, base_id: str) -> Dict[str, Any]:
        """获取模型版本列表"""
        try:
            if base_id not in self.repository_metadata["versions"]:
                return {"success": False, "error": "模型不存在"}
            
            versions = self.repository_metadata["versions"][base_id]
            version_details = []
            
            for version_info in versions:
                model_id = version_info["model_id"]
                if model_id in self.repository_metadata["models"]:
                    version_details.append({
                        **version_info,
                        "model": self.repository_metadata["models"][model_id]
                    })
            
            return {
                "success": True,
                "versions": version_details,
                "total_count": len(version_details)
            }
        except Exception as e:
            logger.error(f"获取模型版本失败: {e}")
            return {"success": False, "error": f"获取模型版本失败: {str(e)}"}
    
    def add_model_tag(self, model_id: str, tag: str) -> Dict[str, Any]:
        """添加模型标签"""
        try:
            if model_id not in self.repository_metadata["models"]:
                return {"success": False, "error": "模型不存在"}
            
            model = self.repository_metadata["models"][model_id]
            
            if tag not in model["tags"]:
                model["tags"].append(tag)
                
                # 更新标签索引
                if tag not in self.repository_metadata["tags"]:
                    self.repository_metadata["tags"][tag] = []
                if model_id not in self.repository_metadata["tags"][tag]:
                    self.repository_metadata["tags"][tag].append(model_id)
                
                # 更新时间
                model["updated_at"] = datetime.now().isoformat()
                
                # 保存元数据
                self._save_metadata()
            
            return {
                "success": True,
                "message": "标签添加成功",
                "tag": tag
            }
        except Exception as e:
            logger.error(f"添加模型标签失败: {e}")
            return {"success": False, "error": f"添加模型标签失败: {str(e)}"}
    
    def remove_model_tag(self, model_id: str, tag: str) -> Dict[str, Any]:
        """移除模型标签"""
        try:
            if model_id not in self.repository_metadata["models"]:
                return {"success": False, "error": "模型不存在"}
            
            model = self.repository_metadata["models"][model_id]
            
            if tag in model["tags"]:
                model["tags"].remove(tag)
                
                # 更新标签索引
                if tag in self.repository_metadata["tags"]:
                    self.repository_metadata["tags"][tag] = [
                        mid for mid in self.repository_metadata["tags"][tag] 
                        if mid != model_id
                    ]
                    
                    # 如果标签没有关联模型，删除标签
                    if not self.repository_metadata["tags"][tag]:
                        del self.repository_metadata["tags"][tag]
                
                # 更新时间
                model["updated_at"] = datetime.now().isoformat()
                
                # 保存元数据
                self._save_metadata()
            
            return {
                "success": True,
                "message": "标签移除成功",
                "tag": tag
            }
        except Exception as e:
            logger.error(f"移除模型标签失败: {e}")
            return {"success": False, "error": f"移除模型标签失败: {str(e)}"}
    
    def get_repository_statistics(self) -> Dict[str, Any]:
        """获取仓库统计信息"""
        try:
            # 更新统计信息
            self._update_statistics()
            
            return {
                "success": True,
                "statistics": self.repository_metadata["statistics"]
            }
        except Exception as e:
            logger.error(f"获取仓库统计失败: {e}")
            return {"success": False, "error": f"获取仓库统计失败: {str(e)}"}
    
    def get_models_by_industry(self, industry: str) -> Dict[str, Any]:
        """按行业获取模型"""
        try:
            models = [
                model for model in self.repository_metadata["models"].values() 
                if model["industry"] == industry
            ]
            
            return {
                "success": True,
                "models": models,
                "total_count": len(models),
                "industry": industry
            }
        except Exception as e:
            logger.error(f"按行业获取模型失败: {e}")
            return {"success": False, "error": f"按行业获取模型失败: {str(e)}"}
    
    def _get_base_id(self, model_id: str) -> str:
        """获取基础模型ID"""
        # 假设模型ID格式为 base_id_vX.Y.Z
        parts = model_id.rsplit('_v', 1)
        if len(parts) == 2:
            return parts[0]
        return model_id
    
    def _get_next_version(self, base_id: str) -> str:
        """获取下一个版本号"""
        if base_id not in self.repository_metadata["versions"] or not self.repository_metadata["versions"][base_id]:
            return "1.0.0"
        
        versions = sorted(
            self.repository_metadata["versions"][base_id],
            key=lambda v: tuple(map(int, v["version"].split('.')))
        )
        latest_version = versions[-1]["version"]
        
        # 生成下一个版本号
        major, minor, patch = map(int, latest_version.split('.'))
        return f"{major}.{minor}.{patch + 1}"
    
    def _update_statistics(self):
        """更新仓库统计信息"""
        total_models = len(self.repository_metadata["models"])
        total_versions = sum(len(v) for v in self.repository_metadata["versions"].values())
        total_deployments = len(self.repository_metadata["deployments"])
        
        # 按类型统计
        type_stats = {}
        for model in self.repository_metadata["models"].values():
            model_type = model.get("type", "unknown")
            type_stats[model_type] = type_stats.get(model_type, 0) + 1
        
        # 按行业统计
        industry_stats = {}
        for model in self.repository_metadata["models"].values():
            industry = model.get("industry", "unknown")
            industry_stats[industry] = industry_stats.get(industry, 0) + 1
        
        # 按框架统计
        framework_stats = {}
        for model in self.repository_metadata["models"].values():
            framework = model.get("framework", "unknown")
            framework_stats[framework] = framework_stats.get(framework, 0) + 1
        
        # 按状态统计
        status_stats = {}
        for model in self.repository_metadata["models"].values():
            status = model.get("status", "unknown")
            status_stats[status] = status_stats.get(status, 0) + 1
        
        self.repository_metadata["statistics"].update({
            "total_models": total_models,
            "total_versions": total_versions,
            "total_deployments": total_deployments,
            "by_type": type_stats,
            "by_industry": industry_stats,
            "by_framework": framework_stats,
            "by_status": status_stats,
            "last_updated": datetime.now().isoformat()
        })
    
    async def backup_repository(self, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """备份仓库"""
        try:
            if not backup_path:
                backup_path = self.storage_path / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = Path(backup_path)
            backup_path.mkdir(exist_ok=True)
            
            # 备份元数据
            metadata_backup = backup_path / "metadata.json"
            with open(metadata_backup, 'w', encoding='utf-8') as f:
                json.dump(self.repository_metadata, f, indent=2, ensure_ascii=False)
            
            # 备份模型文件
            models_backup_path = backup_path / "models"
            models_backup_path.mkdir(exist_ok=True)
            
            for model in self.repository_metadata["models"].values():
                if "file_path" in model:
                    src_file = Path(model["file_path"])
                    if src_file.exists():
                        dst_file = models_backup_path / f"{model['model_id']}.pth"
                        dst_file.write_bytes(src_file.read_bytes())
            
            return {
                "success": True,
                "message": "仓库备份成功",
                "backup_path": str(backup_path),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"备份仓库失败: {e}")
            return {"success": False, "error": f"备份仓库失败: {str(e)}"}
    
    async def restore_repository(self, backup_path: str) -> Dict[str, Any]:
        """从备份恢复仓库"""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                return {"success": False, "error": "备份路径不存在"}
            
            # 恢复元数据
            metadata_backup = backup_path / "metadata.json"
            if not metadata_backup.exists():
                return {"success": False, "error": "备份元数据文件不存在"}
            
            with open(metadata_backup, 'r', encoding='utf-8') as f:
                self.repository_metadata = json.load(f)
            
            # 恢复模型文件
            models_backup_path = backup_path / "models"
            if models_backup_path.exists():
                self.models_dir.mkdir(exist_ok=True)
                
                for src_file in models_backup_path.glob("*.pth"):
                    model_id = src_file.stem
                    dst_file = self.models_dir / src_file.name
                    dst_file.write_bytes(src_file.read_bytes())
                    
                    # 更新文件路径
                    if model_id in self.repository_metadata["models"]:
                        self.repository_metadata["models"][model_id]["file_path"] = str(dst_file)
            
            # 保存元数据
            self._save_metadata()
            
            return {
                "success": True,
                "message": "仓库恢复成功",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"恢复仓库失败: {e}")
            return {"success": False, "error": f"恢复仓库失败: {str(e)}"}
