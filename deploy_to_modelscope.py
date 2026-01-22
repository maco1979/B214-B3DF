#!/usr/bin/env python3
"""
ModelScope模型部署脚本
用于将本地模型快速部署到ModelScope平台

使用方法:
python deploy_to_modelscope.py --model_id agriculture_classification_v1 --repo_id chenshuo1979/aifactory
"""

import argparse
import os
import json
import requests
import hashlib
from pathlib import Path

class ModelScopeDeployer:
    """ModelScope模型部署器"""
    
    def __init__(self, repo_id, model_id, model_path=None, metadata_path=None, api_token=None):
        """初始化部署器"""
        self.repo_id = repo_id
        self.model_id = model_id
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.api_token = api_token
        self.base_url = "https://modelscope.cn"
        
        # 默认路径
        if not self.metadata_path:
            self.metadata_path = Path("backend/models/metadata.json")
        if not self.model_path:
            self.model_path = Path("backend/models")
    
    def load_model_metadata(self):
        """加载模型元数据"""
        print(f"加载模型元数据: {self.metadata_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {self.metadata_path}")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if self.model_id not in metadata.get("models", {}):
            raise ValueError(f"模型ID {self.model_id} 不在元数据中")
        
        return metadata["models"][self.model_id]
    
    def get_model_file_path(self, model_metadata):
        """获取模型文件路径"""
        file_path = model_metadata.get("file_path")
        if not file_path:
            raise ValueError(f"模型 {self.model_id} 没有配置文件路径")
        
        # 处理相对路径 - 直接相对于项目根目录
        if not os.path.isabs(file_path):
            # 移除可能的重复 'models/' 前缀
            if file_path.startswith("models/") or file_path.startswith("models\\"):
                file_path = file_path[7:]
            file_path = Path("backend/models") / file_path
        else:
            file_path = Path(file_path)
        
        print(f"检查模型文件: {file_path}")
        if not file_path.exists():
            # 尝试其他可能的路径
            alternative_path = Path("backend") / file_path
            if alternative_path.exists():
                file_path = alternative_path
            else:
                # 列出目录内容帮助调试
                print(f"目录内容: {list(Path('backend/models').glob('*'))}")
                raise FileNotFoundError(f"模型文件不存在: {file_path}")
        
        return file_path
    
    def calculate_file_hash(self, file_path):
        """计算文件哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def prepare_deployment_package(self):
        """准备部署包"""
        print(f"准备部署包: {self.model_id}")
        
        # 加载元数据
        model_metadata = self.load_model_metadata()
        
        # 获取模型文件
        model_file = self.get_model_file_path(model_metadata)
        
        # 计算文件哈希
        file_hash = self.calculate_file_hash(model_file)
        
        # 构建部署信息
        deployment_info = {
            "model_id": self.model_id,
            "repo_id": self.repo_id,
            "name": model_metadata.get("name", self.model_id),
            "type": model_metadata.get("type", "vision"),
            "framework": model_metadata.get("framework", "pytorch"),
            "version": model_metadata.get("version", "1.0.0"),
            "description": model_metadata.get("description", "AI模型"),
            "file_path": str(model_file),
            "file_size": model_file.stat().st_size,
            "file_hash": file_hash,
            "metadata": model_metadata
        }
        
        return deployment_info
    
    def deploy_model(self):
        """执行部署"""
        print(f"\n=== 开始部署模型到 ModelScope ===")
        print(f"模型ID: {self.model_id}")
        print(f"仓库ID: {self.repo_id}")
        print(f"模型路径: {self.model_path}")
        
        try:
            # 加载元数据
            model_metadata = self.load_model_metadata()
            
            print("\n=== 模型信息 ===")
            print(f"名称: {model_metadata.get('name', self.model_id)}")
            print(f"类型: {model_metadata.get('type', 'vision')}")
            print(f"框架: {model_metadata.get('framework', 'pytorch')}")
            print(f"版本: {model_metadata.get('version', '1.0.0')}")
            print(f"描述: {model_metadata.get('description', 'AI模型')}")
            print(f"状态: {model_metadata.get('status', 'unknown')}")
            
            # 尝试获取模型文件
            has_model_file = False
            model_file_path = None
            
            try:
                model_file_path = self.get_model_file_path(model_metadata)
                has_model_file = True
                file_size = model_file_path.stat().st_size
                file_hash = self.calculate_file_hash(model_file_path)
                print(f"文件: {model_file_path}")
                print(f"大小: {file_size:,} 字节")
                print(f"哈希: {file_hash}")
            except FileNotFoundError as fe:
                print(f"⚠️  警告: 模型文件未找到 - {fe}")
                print("将生成元数据部署命令")
            
            # 构建部署信息
            deployment_info = {
                "model_id": self.model_id,
                "repo_id": self.repo_id,
                "name": model_metadata.get("name", self.model_id),
                "type": model_metadata.get("type", "vision"),
                "framework": model_metadata.get("framework", "pytorch"),
                "version": model_metadata.get("version", "1.0.0"),
                "description": model_metadata.get("description", "AI模型"),
                "metadata": model_metadata
            }
            
            if has_model_file and model_file_path:
                deployment_info["file_path"] = str(model_file_path)
                deployment_info["file_size"] = file_size
                deployment_info["file_hash"] = file_hash
            
            # 生成部署命令
            if has_model_file and model_file_path:
                deploy_command = f"ms upload --model-id {self.repo_id} --model-path {model_file_path}"
            else:
                # 如果没有模型文件，使用元数据部署
                deploy_command = f"ms upload --model-id {self.repo_id} --model-path ./backend/models"
            
            print("\n=== 部署命令 ===")
            print(deploy_command)
            print("\n=== 完整部署方案 ===")
            print("1. 安装 ModelScope SDK:")
            print("   pip install modelscope")
            print("   pip install modelscope[cv,nlp,multi-modal]")
            print()
            print("2. 配置 ModelScope 访问凭证:")
            print("   https://modelscope.cn/docs/%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8")
            print()
            print("3. 准备模型文件:")
            print("   将模型文件放置到: backend/models/ 目录下")
            print("   支持的模型格式: .pth, .pt, .onnx, .h5, .pb 等")
            print()
            print("4. 执行部署命令:")
            print(f"   {deploy_command}")
            print()
            print("5. 部署成功后访问:")
            print(f"   模型主页: {self.base_url}/models/{self.repo_id}")
            print(f"   文件管理: {self.base_url}/models/{self.repo_id}/file/")
            print(f"   API文档: {self.base_url}/models/{self.repo_id}/api")
            print()
            print("=== 批量部署多个模型 ===")
            print("如需部署所有模型，可以使用:")
            print(f"ms upload --model-id {self.repo_id} --model-path ./backend/models")
            print()
            # 保存部署信息
            with open(f"modelscope_deploy_info_{self.model_id}.json", 'w', encoding='utf-8') as f:
                json.dump(deployment_info, f, indent=2, ensure_ascii=False)
            
            print(f"=== 部署信息已保存到: modelscope_deploy_info_{self.model_id}.json ===")
            print()
            print("✅ 部署准备完成!")
            
            return {
                "success": True,
                "message": "部署准备完成，请按照上述说明执行部署",
                "deployment_info": deployment_info,
                "deploy_command": deploy_command,
                "has_model_file": has_model_file
            }
            
        except Exception as e:
            print(f"\n=== 部署失败 ===")
            print(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将模型部署到 ModelScope")
    parser.add_argument("--model-id", required=True, help="模型ID")
    parser.add_argument("--repo-id", default="chenshuo1979/aifactory", help="ModelScope 仓库ID")
    parser.add_argument("--model-path", help="模型文件目录")
    parser.add_argument("--metadata-path", help="元数据文件路径")
    parser.add_argument("--api-token", help="ModelScope API Token")
    
    args = parser.parse_args()
    
    deployer = ModelScopeDeployer(
        repo_id=args.repo_id,
        model_id=args.model_id,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        api_token=args.api_token
    )
    
    deployer.deploy_model()

if __name__ == "__main__":
    main()
