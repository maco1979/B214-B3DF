#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心有机智能体标准化快速检查工具
基于生物-机器融合标准体系开发
用于日常项目标准化合规检查
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse


class CoreOrganicAgentQuickCheck:
    """核心有机智能体标准化快速检查工具"""
    
    def __init__(self):
        self.check_results = []
        self.current_check_index = 0
        self.project_info = {}
        
    def set_project_info(self, project_name: str, system_type: str, application_field: str, bio_component_type: str):
        """设置项目信息"""
        self.project_info = {
            "project_name": project_name,
            "system_type": system_type,
            "application_field": application_field,
            "bio_component_type": bio_component_type,
            "check_time": datetime.now().isoformat()
        }
        
    def run_quick_check(self) -> Dict[str, Any]:
        """运行快速检查"""
        print("🔍 开始核心有机智能体标准化快速检查...")
        print(f"📋 项目信息: {self.project_info['project_name']} ({self.project_info['system_type']})")
        print(f"🧬 生物组件: {self.project_info['bio_component_type']}")
        print(f"🌍 应用领域: {self.project_info['application_field']}")
        print("-" * 70)
        
        # 执行各项检查
        self._check_bio_safety_standards()
        self._check_ethics_compliance()
        self._check_bio_machine_compatibility()
        self._check_core_capabilities()
        self._check_three_layer_protection()
        self._check_evaluation_metrics()
        
        # 生成检查报告
        report = self._generate_check_report()
        
        print(f"\n✅ 快速检查完成！")
        print(f"📊 总检查项: {len(self.check_results)}")
        print(f"✅ 通过项: {len([r for r in self.check_results if r['result']])}")
        print(f"❌ 未通过项: {len([r for r in self.check_results if not r['result']])}")
        print(f"📈 通过率: {report['summary']['pass_rate']:.1%}")
        
        return report
    
    def _check_bio_safety_standards(self):
        """检查生物安全标准符合性"""
        print("\n🛡️  1. 生物安全标准检查")
        
        # 生物安全等级
        result = self._manual_check("确定生物安全等级", 
                                   "是否已确定项目的生物安全等级(P1-P4)？")
        self._record_check_result("生物安全等级确定", result)
        
        result = self._manual_check("实验室符合安全等级", 
                                   "实验室是否符合确定的安全等级要求？")
        self._record_check_result("实验室安全等级", result)
        
        # 生物组件认证
        result = self._manual_check("生物组件活性检测", 
                                   "生物组件活性是否≥90%？")
        self._record_check_result("生物组件活性", result)
        
        result = self._manual_check("生物组件纯度检测", 
                                   "生物组件纯度是否≥99.9%？")
        self._record_check_result("生物组件纯度", result)
        
        result = self._manual_check("病原体筛查", 
                                   "是否完成病原体筛查且结果为阴性？")
        self._record_check_result("病原体筛查", result)
        
        result = self._manual_check("废弃物处理规范", 
                                   "是否建立生物废弃物处理规范？")
        self._record_check_result("废弃物处理", result)
    
    def _check_ethics_compliance(self):
        """检查伦理合规性"""
        print("\n🧭 2. 伦理合规检查")
        
        result = self._manual_check("伦理审查申请", 
                                   "是否已提交伦理审查申请？")
        self._record_check_result("伦理审查申请", result)
        
        result = self._manual_check("伦理委员会批准", 
                                   "是否获得伦理委员会审查批准？")
        self._record_check_result("伦理批准", result)
        
        result = self._manual_check("最小伤害原则", 
                                   "是否遵循最小伤害原则？")
        self._record_check_result("最小伤害原则", result)
        
        result = self._manual_check("生态风险评估", 
                                   "是否完成生态风险评估？")
        self._record_check_result("生态风险评估", result)
        
        result = self._manual_check("知情同意机制", 
                                   "是否建立知情同意机制？")
        self._record_check_result("知情同意", result)
    
    def _check_bio_machine_compatibility(self):
        """检查生物-机器兼容性"""
        print("\n🔧 3. 生物-机器兼容性检查")
        
        result = self._manual_check("生物-电子接口", 
                                   "生物-电子接口是否稳定可靠？")
        self._record_check_result("生物-电子接口", result)
        
        result = self._manual_check("信号转换效率", 
                                   "生物-电子信号转换效率是否≥95%？")
        self._record_check_result("信号转换效率", result)
        
        result = self._manual_check("免疫排斥反应", 
                                   "是否评估免疫排斥反应且在可接受范围内？")
        self._record_check_result("免疫排斥反应", result)
        
        result = self._manual_check("生物相容性", 
                                   "组件间是否无排斥反应？")
        self._record_check_result("生物相容性", result)
    
    def _check_core_capabilities(self):
        """检查核心能力"""
        print("\n⚡ 4. 核心能力检查")
        
        # 自主性
        result = self._manual_check("独立决策比例", 
                                   "独立决策比例是否≥90%？")
        self._record_check_result("独立决策比例", result)
        
        result = self._manual_check("人工干预频率", 
                                   "人工干预频率是否≤1次/100小时？")
        self._record_check_result("人工干预频率", result)
        
        # 适应性
        result = self._manual_check("环境变化响应", 
                                   "环境变化响应时间是否≤5分钟？")
        self._record_check_result("环境响应时间", result)
        
        result = self._manual_check("学习效率", 
                                   "学习效率是否≥85%？")
        self._record_check_result("学习效率", result)
        
        # 稳定性
        result = self._manual_check("性能衰减率", 
                                   "连续运行30天性能衰减率是否≤10%？")
        self._record_check_result("性能衰减率", result)
        
        result = self._manual_check("故障恢复率", 
                                   "故障自动恢复率是否≥95%？")
        self._record_check_result("故障恢复率", result)
    
    def _check_three_layer_protection(self):
        """检查三层防护机制"""
        print("\n🛡️  5. 三层防护机制检查")
        
        # 参数校验层
        result = self._manual_check("参数校验层实现", 
                                   "是否实现参数校验层？")
        self._record_check_result("参数校验层", result)
        
        result = self._manual_check("生物组件参数验证", 
                                   "是否验证生物组件参数有效性？")
        self._record_check_result("生物组件参数验证", result)
        
        result = self._manual_check("连接参数验证", 
                                   "是否验证连接参数有效性？")
        self._record_check_result("连接参数验证", result)
        
        # 判空层
        result = self._manual_check("生物-机器判空层", 
                                   "是否实现生物-机器判空层？")
        self._record_check_result("判空层", result)
        
        result = self._manual_check("生物组件活性验证", 
                                   "是否验证生物组件活性(≥90%)？")
        self._record_check_result("生物组件活性验证", result)
        
        result = self._manual_check("接口连接有效性", 
                                   "是否验证生物-机器接口连接有效性？")
        self._record_check_result("接口连接验证", result)
        
        # 异常层
        result = self._manual_check("全局异常处理", 
                                   "是否实现全局异常处理？")
        self._record_check_result("全局异常处理", result)
        
        result = self._manual_check("生物反应异常捕获", 
                                   "是否捕获生物反应异常？")
        self._record_check_result("生物反应异常捕获", result)
        
        result = self._manual_check("顶层异常兜底", 
                                   "是否实现顶层异常兜底机制？")
        self._record_check_result("顶层异常兜底", result)
    
    def _check_evaluation_metrics(self):
        """检查评估指标达成"""
        print("\n📊 6. 评估指标检查")
        
        print("   请根据实际测试结果评估：")
        
        # 任务成功率
        success_rate = self._get_numeric_input("任务成功率 (%)", 0, 100, 95)
        result = success_rate >= 95
        self._record_check_result("任务成功率(≥95%)", result, detail=f"实际: {success_rate}%")
        
        # 生存周期
        survival_rate = self._get_numeric_input("系统生存周期达标率 (%)", 0, 100, 90)
        result = survival_rate >= 90
        self._record_check_result("生存周期达标率(≥90%)", result, detail=f"实际: {survival_rate}%")
        
        # 可用性
        availability = self._get_numeric_input("系统可用性 (%)", 0, 100, 99.9)
        result = availability >= 99.9
        self._record_check_result("系统可用性(≥99.9%)", result, detail=f"实际: {availability}%")
        
        # 安全测试通过率
        security_pass_rate = self._get_numeric_input("安全测试通过率 (%)", 0, 100, 100)
        result = security_pass_rate >= 100
        self._record_check_result("安全测试通过率(100%)", result, detail=f"实际: {security_pass_rate}%")
    
    def _manual_check(self, check_item: str, prompt: str) -> bool:
        """手动检查项"""
        while True:
            response = input(f"   {check_item}? {prompt} (y/n): ").lower().strip()
            if response in ['y', 'yes', '是', 'Y', 'YES', '']:
                return True
            elif response in ['n', 'no', '否', 'N', 'NO']:
                return False
            else:
                print("   请输入 y(是) 或 n(否)")
    
    def _get_numeric_input(self, prompt: str, min_val: float, max_val: float, default: float) -> float:
        """获取数值输入"""
        while True:
            try:
                value_str = input(f"   {prompt} (默认{default}, 范围{min_val}-{max_val}): ").strip()
                if not value_str:
                    return default
                value = float(value_str)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"   数值应在 {min_val} 到 {max_val} 之间")
            except ValueError:
                print("   请输入有效数字")
    
    def _record_check_result(self, check_name: str, result: bool, detail: str = ""):
        """记录检查结果"""
        check_result = {
            "index": self.current_check_index,
            "check_name": check_name,
            "result": result,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
            "status": "✅" if result else "❌"
        }
        self.check_results.append(check_result)
        self.current_check_index += 1
        
        status_icon = "✅" if result else "❌"
        print(f"   {status_icon} {check_name}: {'通过' if result else '未通过'} {detail}")
    
    def _generate_check_report(self) -> Dict[str, Any]:
        """生成检查报告"""
        total_checks = len(self.check_results)
        passed_checks = len([r for r in self.check_results if r["result"]])
        pass_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        # 计算合规等级
        if pass_rate >= 0.95:
            compliance_level = "优秀"
            level_code = "EXCELLENT"
        elif pass_rate >= 0.85:
            compliance_level = "良好"
            level_code = "GOOD"
        elif pass_rate >= 0.70:
            compliance_level = "合格"
            level_code = "ACCEPTABLE"
        else:
            compliance_level = "不合格"
            level_code = "UNACCEPTABLE"
        
        report = {
            "project_info": self.project_info,
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": total_checks - passed_checks,
                "pass_rate": pass_rate,
                "compliance_level": compliance_level,
                "level_code": level_code,
                "check_time": datetime.now().isoformat()
            },
            "detailed_results": self.check_results,
            "recommendations": self._generate_recommendations()
        }
        
        # 保存报告
        filename = f"core_organic_agent_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 检查报告已保存至: {filename}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 检查未通过的项目
        failed_checks = [r for r in self.check_results if not r["result"]]
        if failed_checks:
            recommendations.append(f"建议优先处理 {len(failed_checks)} 个未通过的检查项")
        
        # 生物安全检查
        bio_safety_failed = [r for r in self.check_results if "生物安全" in r["check_name"] and not r["result"]]
        if bio_safety_failed:
            recommendations.append("生物安全存在风险，需立即处理")
        
        # 伦理合规检查
        ethics_failed = [r for r in self.check_results if "伦理" in r["check_name"] and not r["result"]]
        if ethics_failed:
            recommendations.append("伦理合规存在风险，需立即处理")
        
        # 三层防护机制
        protection_failed = [r for r in self.check_results if "防护" in r["check_name"] and not r["result"]]
        if protection_failed:
            recommendations.append("三层防护机制存在缺陷，需立即完善")
        
        # 评估指标
        metric_failed = [r for r in self.check_results if "率" in r["check_name"] and not r["result"]]
        if metric_failed:
            recommendations.append("部分评估指标未达标，建议优化性能")
        
        if not recommendations:
            recommendations.append("✅ 恭喜！项目基本符合核心有机智能体标准化要求")
        
        return recommendations
    
    def print_summary_report(self, report: Dict[str, Any]):
        """打印摘要报告"""
        print("\n" + "="*70)
        print("📋 核心有机智能体标准化检查摘要报告")
        print("="*70)
        print(f"项目名称: {report['project_info']['project_name']}")
        print(f"系统类型: {report['project_info']['system_type']}")
        print(f"生物组件: {report['project_info']['bio_component_type']}")
        print(f"应用领域: {report['project_info']['application_field']}")
        print(f"检查时间: {report['project_info']['check_time']}")
        print("-"*70)
        print(f"总检查项: {report['summary']['total_checks']}")
        print(f"通过项:   {report['summary']['passed_checks']}")
        print(f"未通过项: {report['summary']['failed_checks']}")
        print(f"通过率:   {report['summary']['pass_rate']:.1%}")
        print(f"合规等级: {report['summary']['compliance_level']}")
        print("-"*70)
        
        if report['recommendations']:
            print("💡 改进建议:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("="*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='核心有机智能体标准化快速检查工具')
    parser.add_argument('--project-name', required=True, help='项目名称')
    parser.add_argument('--system-type', required=True, 
                       choices=['自然生物智能体', '生物混合智能体', '合成生物学智能体'],
                       help='系统类型')
    parser.add_argument('--field', default='企业服务', help='应用领域')
    parser.add_argument('--bio-component', default='神经元网络', 
                       choices=['神经元网络', '活体细胞', '微生物群落', '其他'],
                       help='生物组件类型')
    
    args = parser.parse_args()
    
    # 创建检查工具实例
    checker = CoreOrganicAgentQuickCheck()
    
    # 设置项目信息
    checker.set_project_info(args.project_name, args.system_type, args.field, args.bio_component)
    
    # 运行快速检查
    report = checker.run_quick_check()
    
    # 打印摘要报告
    checker.print_summary_report(report)
    
    # 输出合规状态
    level = report['summary']['compliance_level']
    if level in ['优秀', '良好']:
        print(f"\n🎉 项目符合核心有机智能体标准化要求！({level})")
    elif level == '合格':
        print(f"\n⚠️  项目基本符合要求，但仍有改进空间 ({level})")
        print("   建议关注改进建议并尽快整改")
    else:
        print(f"\n🚨 项目不符合核心有机智能体标准化要求！({level})")
        print("   建议立即整改不合规项后重新检查")


if __name__ == "__main__":
    main()