"""
智能合约封装类
提供AI模型管理和数据溯源相关的智能合约接口
"""

import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class ContractType(Enum):
    """智能合约类型"""
    MODEL_REGISTRY = "model_registry"
    DATA_PROVENANCE = "data_provenance"
    FEDERATED_LEARNING = "federated_learning"


class ModelRegistryContract:
    """模型注册智能合约封装"""
    
    def __init__(self, fabric_client):
        self.client = fabric_client
        self.contract_name = "modelregistry"
    
    async def register_model(self, 
                           model_id: str,
                           model_hash: str,
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """注册新模型到区块链"""
        args = [
            model_id,
            model_hash,
            json.dumps(metadata),
            datetime.utcnow().isoformat()
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "registerModel",
            args
        )
    
    async def update_model_version(self,
                                 model_id: str,
                                 new_hash: str,
                                 version_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """更新模型版本"""
        args = [
            model_id,
            new_hash,
            json.dumps(version_metadata),
            datetime.utcnow().isoformat()
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "updateModelVersion",
            args
        )
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """获取模型信息"""
        args = [model_id]
        
        result = await self.client.query_chaincode(
            self.contract_name,
            "getModelInfo",
            args
        )
        
        if result['success'] and result['data']:
            return {
                'success': True,
                'model_info': json.loads(result['data']),
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return result
    
    async def verify_model_integrity(self, model_id: str, current_hash: str) -> Dict[str, Any]:
        """验证模型完整性"""
        args = [model_id, current_hash]
        
        return await self.client.query_chaincode(
            self.contract_name,
            "verifyModelIntegrity",
            args
        )
    
    async def get_model_history(self, model_id: str) -> Dict[str, Any]:
        """获取模型版本历史"""
        args = [model_id]
        
        result = await self.client.query_chaincode(
            self.contract_name,
            "getModelHistory",
            args
        )
        
        if result['success'] and result['data']:
            return {
                'success': True,
                'history': json.loads(result['data']),
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return result


class DataProvenanceContract:
    """数据溯源智能合约封装"""
    
    def __init__(self, fabric_client):
        self.client = fabric_client
        self.contract_name = "dataprovenance"
    
    async def record_data_usage(self,
                              data_id: str,
                              operation: str,
                              model_id: str,
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """记录数据使用记录"""
        args = [
            data_id,
            operation,
            model_id,
            json.dumps(metadata),
            datetime.utcnow().isoformat()
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "recordDataUsage",
            args
        )
    
    async def get_data_provenance(self, data_id: str) -> Dict[str, Any]:
        """获取数据溯源记录"""
        args = [data_id]
        
        result = await self.client.query_chaincode(
            self.contract_name,
            "getDataProvenance",
            args
        )
        
        if result['success'] and result['data']:
            return {
                'success': True,
                'provenance': json.loads(result['data']),
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return result
    
    async def verify_data_authenticity(self, data_id: str, data_hash: str) -> Dict[str, Any]:
        """验证数据真实性"""
        args = [data_id, data_hash]
        
        return await self.client.query_chaincode(
            self.contract_name,
            "verifyDataAuthenticity",
            args
        )


class FederatedLearningContract:
    """联邦学习智能合约封装"""
    
    def __init__(self, fabric_client):
        self.client = fabric_client
        self.contract_name = "federatedlearning"
    
    async def start_federated_round(self,
                                  round_id: str,
                                  model_id: str,
                                  participants: List[str]) -> Dict[str, Any]:
        """开始联邦学习轮次"""
        args = [
            round_id,
            model_id,
            json.dumps(participants),
            datetime.utcnow().isoformat()
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "startFederatedRound",
            args
        )
    
    async def submit_model_update(self,
                                round_id: str,
                                participant_id: str,
                                model_hash: str,
                                contribution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """提交模型更新"""
        args = [
            round_id,
            participant_id,
            model_hash,
            json.dumps(contribution_metrics),
            datetime.utcnow().isoformat()
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "submitModelUpdate",
            args
        )
    
    async def complete_federated_round(self,
                                     round_id: str,
                                     aggregated_model_hash: str,
                                     aggregation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """完成联邦学习轮次"""
        args = [
            round_id,
            aggregated_model_hash,
            json.dumps(aggregation_metrics),
            datetime.utcnow().isoformat()
        ]
        
        return await self.client.invoke_chaincode(
            self.contract_name,
            "completeFederatedRound",
            args
        )
    
    async def get_round_status(self, round_id: str) -> Dict[str, Any]:
        """获取联邦学习轮次状态"""
        args = [round_id]
        
        result = await self.client.query_chaincode(
            self.contract_name,
            "getRoundStatus",
            args
        )
        
        if result['success'] and result['data']:
            return {
                'success': True,
                'round_status': json.loads(result['data']),
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return result