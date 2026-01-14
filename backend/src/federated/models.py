"""
联邦学习模型定义
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class ClientRegistration:
    """
    联邦客户端注册信息模型
    """
    def __init__(self, client_id: str, client_info: Dict[str, Any]):
        self.client_id = client_id
        self.client_info = client_info
        self.status = "pending"  # pending, approved, rejected, activated, suspended
        self.registered_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        
        # 1. 主体信息
        self.entity_info = client_info.get("entity_info", {})
        self.entity_type = self.entity_info.get("type", "")  # organization, device, proxy
        
        # 2. 资质核验信息
        self.verification = {
            "status": "pending",  # pending, verified, failed
            "verification_results": {},
            "verified_at": None
        }
        
        # 3. 能力评估信息
        self.capability_assessment = {
            "status": "pending",  # pending, passed, failed
            "assessment_results": {},
            "assessed_at": None
        }
        
        # 4. 权限配置
        self.permissions = {
            "role": client_info.get("role", "training_node"),  # training_node, aggregation_node, validation_node
            "allowed_tasks": client_info.get("allowed_tasks", []),
            "permission_level": "basic"  # basic, advanced, admin
        }
        
        # 5. 安全认证
        self.security = {
            "certificate": None,
            "public_key": None,
            "last_auth_time": None,
            "auth_status": "pending"  # pending, authenticated, expired
        }
        
        # 6. 合规备案
        self.compliance = {
            "agreements_signed": [],
            "privacy_commitments": [],
            "compliance_documents": [],
            "compliance_status": "pending"  # pending, compliant, non_compliant
        }
        
        # 7. 试运营信息
        self.trial_operation = {
            "status": "pending",  # pending, in_progress, passed, failed
            "trial_results": {},
            "trial_completed_at": None
        }
        
        # 8. 审核信息
        self.review = {
            "reviewer_id": None,
            "review_comments": [],
            "review_status": "pending",  # pending, approved, rejected
            "reviewed_at": None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "client_id": self.client_id,
            "status": self.status,
            "registered_at": self.registered_at,
            "updated_at": self.updated_at,
            "entity_info": self.entity_info,
            "entity_type": self.entity_type,
            "verification": self.verification,
            "capability_assessment": self.capability_assessment,
            "permissions": self.permissions,
            "security": self.security,
            "compliance": self.compliance,
            "trial_operation": self.trial_operation,
            "review": self.review
        }


class ClientCapabilityAssessment:
    """
    客户端能力评估结果模型
    """
    def __init__(self):
        self.hardware = {
            "cpu_model": None,
            "gpu_model": None,
            "memory_gb": None,
            "storage_gb": None,
            "pass": False
        }
        self.network = {
            "upload_bandwidth_mbps": None,
            "download_bandwidth_mbps": None,
            "latency_ms": None,
            "stability_score": None,
            "pass": False
        }
        self.data = {
            "data_type": None,
            "data_size": None,
            "data_distribution": None,
            "data_compliance": None,
            "data_quality_score": None,
            "pass": False
        }
        self.software = {
            "federated_framework": None,
            "framework_version": None,
            "dependencies": [],
            "compatibility_score": None,
            "pass": False
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "hardware": self.hardware,
            "network": self.network,
            "data": self.data,
            "software": self.software,
            "overall_pass": all([
                self.hardware["pass"],
                self.network["pass"],
                self.data["pass"],
                self.software["pass"]
            ])
        }


class ClientVerificationResult:
    """
    客户端身份核验结果模型
    """
    def __init__(self):
        self.basic_info = {
            "entity_name": None,
            "entity_id": None,
            "contact_info": None,
            "verified": False
        }
        self.qualification_docs = {
            "business_license": {
                "verified": False,
                "expired": False
            },
            "industry_qualifications": [],
            "compliance_certificates": []
        }
        self.identity_auth = {
            "public_account_verified": False,
            "online_verification": False,
            "unique_hardware_id": None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "basic_info": self.basic_info,
            "qualification_docs": self.qualification_docs,
            "identity_auth": self.identity_auth,
            "overall_verified": self.basic_info["verified"] and \
                               self.identity_auth["public_account_verified"] and \
                               self.identity_auth["online_verification"]
        }
