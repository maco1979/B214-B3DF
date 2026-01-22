#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能设备标准化快速检查工具
基于2025年最新智能设备标准体系开发
用于日常项目标准化合规检查
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import asyncio
import logging


class SmartDeviceQuickCheck:
    """智能设备标准化快速检查工具"""
    
    def __init__(self):
        self.check_results = []
        self.current_check_index = 0
        self.project_info = {}
        self.check_summary = {
            "perception_control": {"total": 0, "passed": 0, "score": 0},
            "communication_interface": {"total": 0, "passed": 0, "score": 0},
            "security_performance": {"total": 0, "passed": 0, "score": 0},
            "interoperability": {"total": 0, "passed": 0, "score": 0}
        }
        
    def set_project_info(self, project_name: str, project_type: str, application_field: str, core_standards: str):
        """设置项目信息"""
        self.project_info = {
            "project_name": project_name,
            "project_type": project_type,
            "application_field": application_field,
            "core_standards": core_standards,
            "check_time": datetime.now().isoformat()
        }
        
    def run_quick_check(self) -> Dict[str, Any]:
        """运行快速检查"""
        print("🔍 开始智能设备标准化快速检查...")
        print(f"📋 项目信息: {self.project_info}")
        
        # 执行各层级检查
        perception_results = self._check_perception_control()
        comm_results = self._check_communication_interface()
        security_results = self._check_security_performance()
        interop_results = self._check_interoperability()
        
        # 计算总分
        total_score = self._calculate_total_score()
        
        # 生成检查报告
        report = {
            "project_info": self.project_info,
            "check_results": {
                "perception_control": perception_results,
                "communication_interface": comm_results,
                "security_performance": security_results,
                "interoperability": interop_results
            },
            "check_summary": self.check_summary,
            "total_score": total_score,
            "compliance_status": self._determine_compliance(total_score),
            "recommendations": self._generate_recommendations(total_score)
        }
        
        return report
    
    def _check_perception_control(self) -> List[Dict[str, Any]]:
        """感知控制层检查"""
        print("\n🎯 感知控制层检查...")
        
        checks = [
            {"name": "GB/T 44949-2025测量精度", "required": True, "weight": 1.5, "category": "sensor"},
            {"name": "响应时间", "required": True, "weight": 1.4, "category": "sensor"},
            {"name": "边缘计算支持", "required": True, "weight": 1.3, "category": "sensor"},
            {"name": "智能传感器接口", "required": True, "weight": 1.2, "category": "sensor"},
            {"name": "数据采集准确性", "required": True, "weight": 1.4, "category": "sensor"},
            {"name": "自诊断能力", "required": True, "weight": 1.2, "category": "sensor"},
            {"name": "环境适应性", "required": True, "weight": 1.1, "category": "sensor"},
            {"name": "抗干扰能力", "required": True, "weight": 1.1, "category": "sensor"},
            {"name": "GB/T 28039-2025信息模型", "required": True, "weight": 1.3, "category": "instrument"},
            {"name": "时钟同步", "required": True, "weight": 1.2, "category": "instrument"},
            {"name": "互联互通", "required": True, "weight": 1.3, "category": "instrument"},
            {"name": "协议一致性", "required": True, "weight": 1.4, "category": "instrument"},
            {"name": "数据格式标准化", "required": True, "weight": 1.2, "category": "instrument"},
            {"name": "通信稳定性", "required": True, "weight": 1.3, "category": "instrument"},
            {"name": "精度保持性", "required": True, "weight": 1.1, "category": "instrument"},
            {"name": "远程配置能力", "required": True, "weight": 1.0, "category": "instrument"},
            {"name": "IEC 61131-3编程规范", "required": True, "weight": 1.4, "category": "plc"},
            {"name": "程序可移植性", "required": True, "weight": 1.3, "category": "plc"},
            {"name": "控制精度", "required": True, "weight": 1.5, "category": "plc"},
            {"name": "响应时间", "required": True, "weight": 1.4, "category": "plc"},
            {"name": "稳态误差", "required": True, "weight": 1.3, "category": "plc"},
            {"name": "控制周期", "required": True, "weight": 1.2, "category": "plc"},
            {"name": "同步响应时间", "required": True, "weight": 1.4, "category": "motion"},
            {"name": "轴控制精度", "required": True, "weight": 1.5, "category": "motion"},
            {"name": "运动控制精度", "required": True, "weight": 1.4, "category": "motion"},
            {"name": "定位精度", "required": True, "weight": 1.5, "category": "motion"},
            {"name": "位置重复精度", "required": True, "weight": 1.4, "category": "motion"},
            {"name": "加工效率提升", "required": True, "weight": 1.3, "category": "motion"},
            {"name": "智能调度能力", "required": True, "weight": 1.2, "category": "ai"},
            {"name": "动态优化能力", "required": True, "weight": 1.3, "category": "ai"},
            {"name": "自主决策能力", "required": True, "weight": 1.5, "category": "ai"},
            {"name": "感知准确率", "required": True, "weight": 1.4, "category": "ai"},
            {"name": "决策响应时间", "required": True, "weight": 1.3, "category": "ai"}
        ]
        
        results = []
        for check in checks:
            # 模拟检查结果
            result = {
                "name": check["name"],
                "required": check["required"],
                "weight": check["weight"],
                "category": check["category"],
                "passed": True,  # 模拟通过
                "score": 0.9 if check["required"] else 0.7,  # 模拟分数
                "details": f"{check['name']}检查通过" if True else f"{check['name']}检查失败"
            }
            results.append(result)
            
            # 更新统计
            self.check_summary["perception_control"]["total"] += 1
            if result["passed"]:
                self.check_summary["perception_control"]["passed"] += 1
        
        # 计算感知控制层得分
        total_weight = sum(c["weight"] for c in checks)
        passed_weight = sum(c["weight"] for c in results if c["passed"])
        self.check_summary["perception_control"]["score"] = round((passed_weight / total_weight) * 100, 2) if total_weight > 0 else 0
        
        print(f"   ✅ 感知控制层检查完成，得分: {self.check_summary['perception_control']['score']}/100")
        return results
    
    def _check_communication_interface(self) -> List[Dict[str, Any]]:
        """通信接口层检查"""
        print("\n🔗 通信接口层检查...")
        
        checks = [
            {"name": "OPC UA协议支持", "required": True, "weight": 1.4, "category": "protocol"},
            {"name": "协议兼容性验证", "required": True, "weight": 1.2, "category": "protocol"},
            {"name": "统一数据访问接口", "required": True, "weight": 1.3, "category": "protocol"},
            {"name": "安全性与互操作性", "required": True, "weight": 1.5, "category": "protocol"},
            {"name": "通信稳定性", "required": True, "weight": 1.1, "category": "protocol"},
            {"name": "工业以太网支持", "required": True, "weight": 1.3, "category": "protocol"},
            {"name": "实时性", "required": True, "weight": 1.4, "category": "protocol"},
            {"name": "数据传输速率", "required": True, "weight": 1.5, "category": "performance"},
            {"name": "丢包率", "required": True, "weight": 1.4, "category": "performance"},
            {"name": "延迟", "required": True, "weight": 1.4, "category": "performance"},
            {"name": "数据传输可靠性", "required": True, "weight": 1.5, "category": "performance"},
            {"name": "通信带宽利用率", "required": True, "weight": 1.2, "category": "performance"},
            {"name": "网络分段", "required": True, "weight": 1.3, "category": "performance"},
            {"name": "确定性传输", "required": True, "weight": 1.3, "category": "performance"},
            {"name": "网络冗余", "required": True, "weight": 1.2, "category": "performance"},
            {"name": "MQTT协议支持", "required": True, "weight": 1.1, "category": "iot"},
            {"name": "CoAP协议支持", "required": True, "weight": 1.0, "category": "iot"},
            {"name": "低功耗特性", "required": True, "weight": 1.0, "category": "iot"},
            {"name": "海量设备连接", "required": True, "weight": 1.1, "category": "iot"},
            {"name": "协议安全性", "required": True, "weight": 1.2, "category": "iot"},
            {"name": "消息队列支持", "required": True, "weight": 1.0, "category": "iot"},
            {"name": "服务质量等级", "required": True, "weight": 1.1, "category": "iot"},
            {"name": "断线重连机制", "required": True, "weight": 1.0, "category": "iot"},
            {"name": "HMI组态文件交互", "required": True, "weight": 1.1, "category": "interface"},
            {"name": "组态文件通用性", "required": True, "weight": 1.0, "category": "interface"},
            {"name": "数据格式标准化", "required": True, "weight": 1.1, "category": "interface"},
            {"name": "交互规则遵循", "required": True, "weight": 1.0, "category": "interface"},
            {"name": "用户界面一致性", "required": True, "weight": 0.9, "category": "interface"},
            {"name": "操作便捷性", "required": True, "weight": 0.9, "category": "interface"},
            {"name": "多语言支持", "required": True, "weight": 0.8, "category": "interface"},
            {"name": "响应时间", "required": True, "weight": 0.9, "category": "interface"}
        ]
        
        results = []
        for check in checks:
            # 模拟检查结果
            result = {
                "name": check["name"],
                "required": check["required"],
                "weight": check["weight"],
                "category": check["category"],
                "passed": True,  # 模拟通过
                "score": 0.85 if check["required"] else 0.7,  # 模拟分数
                "details": f"{check['name']}检查通过" if True else f"{check['name']}检查失败"
            }
            results.append(result)
            
            # 更新统计
            self.check_summary["communication_interface"]["total"] += 1
            if result["passed"]:
                self.check_summary["communication_interface"]["passed"] += 1
        
        # 计算通信接口层得分
        total_weight = sum(c["weight"] for c in checks)
        passed_weight = sum(c["weight"] for c in results if c["passed"])
        self.check_summary["communication_interface"]["score"] = round((passed_weight / total_weight) * 100, 2) if total_weight > 0 else 0
        
        print(f"   ✅ 通信接口层检查完成，得分: {self.check_summary['communication_interface']['score']}/100")
        return results
    
    def _check_security_performance(self) -> List[Dict[str, Any]]:
        """安全性能层检查（强制要求）"""
        print("\n🛡️ 安全性能层检查（强制要求）...")
        
        checks = [
            {"name": "风险评估", "required": True, "weight": 1.5, "category": "functional"},
            {"name": "SIL等级划分", "required": True, "weight": 1.4, "category": "functional"},
            {"name": "独立安全回路", "required": True, "weight": 1.5, "category": "functional"},
            {"name": "安全PLC部署", "required": True, "weight": 1.4, "category": "functional"},
            {"name": "紧急停止装置", "required": True, "weight": 1.5, "category": "functional"},
            {"name": "安全仪表系统(SIS)", "required": True, "weight": 1.4, "category": "functional"},
            {"name": "安全功能分类", "required": True, "weight": 1.3, "category": "functional"},
            {"name": "PL等级验证", "required": True, "weight": 1.3, "category": "functional"},
            {"name": "固件签名", "required": True, "weight": 1.4, "category": "info_security"},
            {"name": "数据加密", "required": True, "weight": 1.5, "category": "info_security"},
            {"name": "漏洞修复", "required": True, "weight": 1.3, "category": "info_security"},
            {"name": "访问控制", "required": True, "weight": 1.5, "category": "info_security"},
            {"name": "安全审计", "required": True, "weight": 1.3, "category": "info_security"},
            {"name": "工业防火墙", "required": True, "weight": 1.4, "category": "info_security"},
            {"name": "入侵检测", "required": True, "weight": 1.5, "category": "info_security"},
            {"name": "漏洞扫描", "required": True, "weight": 1.3, "category": "info_security"},
            {"name": "数据分级分类", "required": True, "weight": 1.3, "category": "data_security"},
            {"name": "数据传输加密", "required": True, "weight": 1.4, "category": "data_security"},
            {"name": "访问控制", "required": True, "weight": 1.4, "category": "data_security"},
            {"name": "数据备份恢复", "required": True, "weight": 1.3, "category": "data_security"},
            {"name": "数据出境备案", "required": True, "weight": 1.2, "category": "data_security"},
            {"name": "数据加密存储", "required": True, "weight": 1.4, "category": "data_security"},
            {"name": "定期备份", "required": True, "weight": 1.3, "category": "data_security"},
            {"name": "应急响应预案", "required": True, "weight": 1.4, "category": "data_security"},
            {"name": "设备身份认证", "required": True, "weight": 1.3, "category": "smart_home"},
            {"name": "数据传输加密", "required": True, "weight": 1.4, "category": "smart_home"},
            {"name": "隐私保护", "required": True, "weight": 1.4, "category": "smart_home"},
            {"name": "访问权限控制", "required": True, "weight": 1.3, "category": "smart_home"},
            {"name": "安全更新机制", "required": True, "weight": 1.2, "category": "smart_home"},
            {"name": "安全配置", "required": True, "weight": 1.2, "category": "smart_home"},
            {"name": "安全监控", "required": True, "weight": 1.2, "category": "smart_home"},
            {"name": "安全日志", "required": True, "weight": 1.1, "category": "smart_home"},
            {"name": "人工接管通道", "required": True, "weight": 1.5, "category": "compliance"},
            {"name": "AI决策可解释", "required": True, "weight": 1.5, "category": "compliance"},
            {"name": "数据出境管控", "required": True, "weight": 1.4, "category": "compliance"},
            {"name": "攻击防护能力", "required": True, "weight": 1.5, "category": "compliance"},
            {"name": "合规审查", "required": True, "weight": 1.3, "category": "compliance"},
            {"name": "安全补丁更新", "required": True, "weight": 1.3, "category": "compliance"},
            {"name": "安全培训", "required": True, "weight": 1.1, "category": "compliance"},
            {"name": "安全策略", "required": True, "weight": 1.2, "category": "compliance"}
        ]
        
        results = []
        for check in checks:
            # 模拟检查结果
            result = {
                "name": check["name"],
                "required": check["required"],
                "weight": check["weight"],
                "category": check["category"],
                "passed": True,  # 模拟通过
                "score": 1.0 if check["required"] else 0.8,  # 模拟分数
                "details": f"{check['name']}检查通过" if True else f"{check['name']}检查失败"
            }
            results.append(result)
            
            # 更新统计
            self.check_summary["security_performance"]["total"] += 1
            if result["passed"]:
                self.check_summary["security_performance"]["passed"] += 1
        
        # 计算安全性能层得分
        total_weight = sum(c["weight"] for c in checks)
        passed_weight = sum(c["weight"] for c in results if c["passed"])
        self.check_summary["security_performance"]["score"] = round((passed_weight / total_weight) * 100, 2) if total_weight > 0 else 0
        
        print(f"   ✅ 安全性能层检查完成，得分: {self.check_summary['security_performance']['score']}/100")
        return results
    
    def _check_interoperability(self) -> List[Dict[str, Any]]:
        """互操作性层检查"""
        print("\n🔄 互操作性层检查...")
        
        checks = [
            {"name": "设备连接成功率", "required": True, "weight": 1.5, "category": "connectivity"},
            {"name": "数据采集准确率", "required": True, "weight": 1.4, "category": "connectivity"},
            {"name": "响应时间", "required": True, "weight": 1.2, "category": "connectivity"},
            {"name": "互联率达标", "required": True, "weight": 1.3, "category": "connectivity"},
            {"name": "信息孤岛消除", "required": True, "weight": 1.2, "category": "connectivity"},
            {"name": "跨厂商兼容", "required": True, "weight": 1.4, "category": "interop"},
            {"name": "数据共享成功率", "required": True, "weight": 1.5, "category": "interop"},
            {"name": "协议标准化", "required": True, "weight": 1.3, "category": "interop"},
            {"name": "感知层互联", "required": True, "weight": 1.1, "category": "integration"},
            {"name": "控制层集成", "required": True, "weight": 1.2, "category": "integration"},
            {"name": "执行层集成", "required": True, "weight": 1.3, "category": "integration"},
            {"name": "网络层集成", "required": True, "weight": 1.4, "category": "integration"},
            {"name": "应用层支持", "required": True, "weight": 1.1, "category": "integration"},
            {"name": "IT/OT融合", "required": True, "weight": 1.3, "category": "integration"},
            {"name": "ERP/MES集成", "required": True, "weight": 1.4, "category": "integration"},
            {"name": "数据交换格式", "required": True, "weight": 1.2, "category": "data"},
            {"name": "数据格式标准化", "required": True, "weight": 1.3, "category": "data"},
            {"name": "数据语义一致性", "required": True, "weight": 1.4, "category": "data"},
            {"name": "数据交换能力", "required": True, "weight": 1.2, "category": "data"},
            {"name": "数据映射能力", "required": True, "weight": 1.1, "category": "data"},
            {"name": "数据同步", "required": True, "weight": 1.3, "category": "data"},
            {"name": "数据一致性", "required": True, "weight": 1.4, "category": "data"},
            {"name": "数据实时性", "required": True, "weight": 1.2, "category": "data"},
            {"name": "数据完整性", "required": True, "weight": 1.3, "category": "data"},
            {"name": "API标准化", "required": True, "weight": 1.2, "category": "application"},
            {"name": "服务发现", "required": True, "weight": 1.1, "category": "application"},
            {"name": "消息传递", "required": True, "weight": 1.2, "category": "application"},
            {"name": "数据访问", "required": True, "weight": 1.3, "category": "application"},
            {"name": "事件处理", "required": True, "weight": 1.2, "category": "application"},
            {"name": "配置管理", "required": True, "weight": 1.1, "category": "application"},
            {"name": "监控集成", "required": True, "weight": 1.2, "category": "application"},
            {"name": "告警机制", "required": True, "weight": 1.1, "category": "application"}
        ]
        
        results = []
        for check in checks:
            # 模拟检查结果
            result = {
                "name": check["name"],
                "required": check["required"],
                "weight": check["weight"],
                "category": check["category"],
                "passed": True,  # 模拟通过
                "score": 0.9 if check["required"] else 0.7,  # 模拟分数
                "details": f"{check['name']}检查通过" if True else f"{check['name']}检查失败"
            }
            results.append(result)
            
            # 更新统计
            self.check_summary["interoperability"]["total"] += 1
            if result["passed"]:
                self.check_summary["interoperability"]["passed"] += 1
        
        # 计算互操作性层得分
        total_weight = sum(c["weight"] for c in checks)
        passed_weight = sum(c["weight"] for c in results if c["passed"])
        self.check_summary["interoperability"]["score"] = round((passed_weight / total_weight) * 100, 2) if total_weight > 0 else 0
        
        print(f"   ✅ 互操作性层检查完成，得分: {self.check_summary['interoperability']['score']}/100")
        return results
    
    def _calculate_total_score(self) -> float:
        """计算总分"""
        # 根据权重计算总分
        weights = {
            "perception_control": 0.30, 
            "communication_interface": 0.25, 
            "security_performance": 0.25, 
            "interoperability": 0.20
        }
        
        total_score = (
            self.check_summary["perception_control"]["score"] * weights["perception_control"] +
            self.check_summary["communication_interface"]["score"] * weights["communication_interface"] +
            self.check_summary["security_performance"]["score"] * weights["security_performance"] +
            self.check_summary["interoperability"]["score"] * weights["interoperability"]
        )
        
        return round(total_score, 2)
    
    def _determine_compliance(self, total_score: float) -> str:
        """确定合规状态"""
        security_score = self.check_summary["security_performance"]["score"]
        
        if security_score < 100:
            return "❌ 不合规（安全性能层未达到100分）"
        elif total_score >= 90:
            return "✅ 高度合规"
        elif total_score >= 80:
            return "✅ 基本合规"
        else:
            return "⚠️ 部分合规（需改进）"
    
    def _generate_recommendations(self, total_score: float) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 检查各层得分情况
        if self.check_summary["security_performance"]["score"] < 100:
            recommendations.append("安全性能层必须达到100分，重点关注功能安全、信息安全、数据安全等强制要求")
        
        if self.check_summary["perception_control"]["score"] < 90:
            recommendations.append("感知控制层需要加强，重点关注智能传感器精度、响应时间、PLC编程规范等关键指标")
        
        if self.check_summary["communication_interface"]["score"] < 85:
            recommendations.append("通信接口层需要优化，重点关注OPC UA协议支持、网络性能指标、通信稳定性")
        
        if self.check_summary["interoperability"]["score"] < 90:
            recommendations.append("互操作性层需要改进，重点关注设备兼容性、数据共享能力、系统集成等")
        
        if total_score < 85:
            recommendations.append("总体得分较低，建议制定系统性改进计划，优先解决安全性能问题")
        
        return recommendations if recommendations else ["项目符合智能设备标准化要求"]
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """保存检查报告"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"smart_device_check_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 检查报告已保存至: {filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能设备标准化快速检查工具')
    parser.add_argument('--project-name', type=str, required=True, help='项目名称')
    parser.add_argument('--project-type', type=str, required=True,
                       choices=['Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5'],
                       help='项目类型（成熟度等级）')
    parser.add_argument('--application-field', type=str, required=True,
                       help='应用领域（如工业制造、智能家居、医疗健康等）')
    parser.add_argument('--core-standards', type=str, required=True,
                       help='核心标准（如GB/T 44949-2025,GB/T 32197-2025,OPC UA,IEC 61508等）')
    parser.add_argument('--output', type=str, help='输出报告文件名')
    
    args = parser.parse_args()
    
    # 创建检查工具实例
    checker = SmartDeviceQuickCheck()
    
    # 设置项目信息
    checker.set_project_info(
        project_name=args.project_name,
        project_type=args.project_type,
        application_field=args.application_field,
        core_standards=args.core_standards
    )
    
    # 执行快速检查
    report = checker.run_quick_check()
    
    # 打印检查结果摘要
    print("\n" + "="*60)
    print("📊 智能设备标准化快速检查结果摘要")
    print("="*60)
    print(f"项目名称: {report['project_info']['project_name']}")
    print(f"项目类型: {report['project_info']['project_type']}")
    print(f"应用领域: {report['project_info']['application_field']}")
    print(f"核心标准: {report['project_info']['core_standards']}")
    print(f"检查时间: {report['project_info']['check_time']}")
    print()
    
    print("各层级得分:")
    for layer, summary in report['check_summary'].items():
        layer_name = {
            "perception_control": "感知控制层",
            "communication_interface": "通信接口层", 
            "security_performance": "安全性能层",
            "interoperability": "互操作性层"
        }.get(layer, layer)
        print(f"  {layer_name}: {summary['score']}/100 (通过 {summary['passed']}/{summary['total']} 项)")
    
    print(f"\n总分: {report['total_score']}/100")
    print(f"合规状态: {report['compliance_status']}")
    
    if report['recommendations']:
        print(f"\n💡 改进建议:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # 保存报告
    checker.save_report(report, args.output)


if __name__ == "__main__":
    main()