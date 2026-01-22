#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电子设备标准化快速检查工具
实现自动化合规检查功能，覆盖安全性能、EMC性能、接口兼容性、可靠性、环保要求五大核心领域
"""

import json
import argparse
import datetime
from typing import Dict, List, Any, Tuple
import random


class ElectronicDeviceStandardChecker:
    """
    电子设备标准化快速检查工具类
    涵盖安全性能、EMC性能、接口兼容性、可靠性、环保要求五大核心领域
    """
    
    def __init__(self):
        self.check_weights = {
            'safety_performance': 0.30,  # 安全性能权重30%
            'emc_performance': 0.25,     # EMC性能权重25%
            'interface_compatibility': 0.20,  # 接口兼容性权重20%
            'reliability': 0.15,         # 可靠性权重15%
            'environmental_requirements': 0.10  # 环保要求权重10%
        }
        
        self.compliance_thresholds = {
            'overall': 0.95,  # 总体合规率阈值95%
            'safety_performance': 1.00,  # 安全性能检查项通过率100%
            'emc_performance': 0.98,     # EMC性能检查项通过率98%
            'interface_compatibility': 0.95,  # 接口兼容性检查项通过率95%
            'reliability': 0.90,         # 可靠性检查项通过率90%
            'environmental_requirements': 1.00  # 环保要求检查项通过率100%
        }
    
    def check_safety_performance(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查安全性能 - 权重30%
        """
        print("正在检查安全性能...")
        
        # 模拟安全性能检查项目
        safety_checks = [
            {
                'id': '2.1.1',
                'name': '危害分析与风险评估(HARA)',
                'required': True,
                'passed': device_info.get('has_hara_analysis', True),
                'standard': 'IEC 62368-1:2023'
            },
            {
                'id': '2.1.2',
                'name': '电气安全要求(绝缘电阻≥10MΩ)',
                'required': True,
                'passed': device_info.get('insulation_resistance', 20) >= 10,
                'standard': 'IEC 62368-1:2023',
                'measured_value': device_info.get('insulation_resistance', 20),
                'unit': 'MΩ'
            },
            {
                'id': '2.1.3',
                'name': '电气安全要求(漏电流≤0.75mA)',
                'required': True,
                'passed': device_info.get('leakage_current', 0.5) <= 0.75,
                'standard': 'IEC 62368-1:2023',
                'measured_value': device_info.get('leakage_current', 0.5),
                'unit': 'mA'
            },
            {
                'id': '2.1.4',
                'name': '抗电强度(≥1500V)',
                'required': True,
                'passed': device_info.get('dielectric_strength', 2000) >= 1500,
                'standard': 'IEC 62368-1:2023',
                'measured_value': device_info.get('dielectric_strength', 2000),
                'unit': 'V'
            },
            {
                'id': '2.1.5',
                'name': '温升限制',
                'required': True,
                'passed': device_info.get('temperature_rise', 40) <= 60,
                'standard': 'IEC 62368-1:2023',
                'measured_value': device_info.get('temperature_rise', 40),
                'unit': 'K'
            },
            {
                'id': '2.1.6',
                'name': '电池安全',
                'required': True,
                'passed': device_info.get('battery_safe', True),
                'standard': 'IEC 62368-1:2023'
            }
        ]
        
        total_checks = len(safety_checks)
        passed_checks = sum(1 for check in safety_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'safety_performance',
            'name': '安全性能',
            'weight': self.check_weights['safety_performance'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['safety_performance'],
            'checks': safety_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['safety_performance'],
            'score': compliance_rate * self.check_weights['safety_performance']
        }
    
    def check_emc_performance(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查EMC性能 - 权重25%
        """
        print("正在检查EMC性能...")
        
        # 模拟EMC性能检查项目
        emc_checks = [
            {
                'id': '3.1.1',
                'name': '传导发射(150kHz-30MHz ≤ 40dBμV)',
                'required': True,
                'passed': device_info.get('conducted_emission', 35) <= 40,
                'standard': 'CISPR 32/GB/T 9254-2021',
                'measured_value': device_info.get('conducted_emission', 35),
                'unit': 'dBμV'
            },
            {
                'id': '3.1.2',
                'name': '辐射发射(30MHz-6GHz ≤ 30dBμV/m)',
                'required': True,
                'passed': device_info.get('radiated_emission', 25) <= 30,
                'standard': 'CISPR 32/GB/T 9254-2021',
                'measured_value': device_info.get('radiated_emission', 25),
                'unit': 'dBμV/m'
            },
            {
                'id': '3.2.1',
                'name': '静电放电抗扰度(接触放电≥±4kV)',
                'required': True,
                'passed': device_info.get('esd_contact', 6) >= 4,
                'standard': 'IEC 61000-4-2',
                'measured_value': device_info.get('esd_contact', 6),
                'unit': 'kV'
            },
            {
                'id': '3.2.2',
                'name': '静电放电抗扰度(空气放电≥±8kV)',
                'required': True,
                'passed': device_info.get('esd_air', 10) >= 8,
                'standard': 'IEC 61000-4-2',
                'measured_value': device_info.get('esd_air', 10),
                'unit': 'kV'
            },
            {
                'id': '3.2.3',
                'name': '射频电磁场辐射抗扰度(≥10V/m)',
                'required': True,
                'passed': device_info.get('rf_immunity', 12) >= 10,
                'standard': 'IEC 61000-4-3',
                'measured_value': device_info.get('rf_immunity', 12),
                'unit': 'V/m'
            },
            {
                'id': '3.2.4',
                'name': '电快速瞬变脉冲群抗扰度(≥2kV)',
                'required': True,
                'passed': device_info.get('eft_burst', 3) >= 2,
                'standard': 'IEC 61000-4-4',
                'measured_value': device_info.get('eft_burst', 3),
                'unit': 'kV'
            },
            {
                'id': '3.2.5',
                'name': '浪涌(冲击)抗扰度',
                'required': True,
                'passed': device_info.get('surge_immunity', True),
                'standard': 'IEC 61000-4-5'
            }
        ]
        
        total_checks = len(emc_checks)
        passed_checks = sum(1 for check in emc_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'emc_performance',
            'name': 'EMC性能',
            'weight': self.check_weights['emc_performance'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['emc_performance'],
            'checks': emc_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['emc_performance'],
            'score': compliance_rate * self.check_weights['emc_performance']
        }
    
    def check_interface_compatibility(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查接口兼容性 - 权重20%
        """
        print("正在检查接口兼容性...")
        
        # 模拟接口兼容性检查项目
        interface_checks = [
            {
                'id': '4.1.1',
                'name': 'USB Type-C可逆插拔设计',
                'required': True,
                'passed': device_info.get('usb_type_c_reversible', True),
                'standard': 'USB Type-C'
            },
            {
                'id': '4.1.2',
                'name': 'USB Type-C数据传输速率(≥40Gbps)',
                'required': True,
                'passed': device_info.get('usb_type_c_speed', 40) >= 40,
                'standard': 'USB4 Version 2.0',
                'measured_value': device_info.get('usb_type_c_speed', 40),
                'unit': 'Gbps'
            },
            {
                'id': '4.1.3',
                'name': 'USB Type-C功率输出(≥240W)',
                'required': False,  # 非强制要求
                'passed': device_info.get('usb_type_c_power', 240) >= 240,
                'standard': 'USB PD 3.1',
                'measured_value': device_info.get('usb_type_c_power', 240),
                'unit': 'W'
            },
            {
                'id': '4.1.4',
                'name': '协议兼容性(USB 2.0/3.2/DP/TBT)',
                'required': True,
                'passed': device_info.get('usb_protocol_compatibility', True),
                'standard': 'USB Type-C'
            },
            {
                'id': '4.2.1',
                'name': 'HDMI 2.1视频传输能力(8K/60Hz或4K/120Hz)',
                'required': False,  # 非强制要求
                'passed': device_info.get('hdmi_8k_capability', True),
                'standard': 'HDMI 2.1'
            },
            {
                'id': '4.2.2',
                'name': 'HDMI 2.1带宽性能(≥48Gbps)',
                'required': False,  # 非强制要求
                'passed': device_info.get('hdmi_bandwidth', 48) >= 48,
                'standard': 'HDMI 2.1',
                'measured_value': device_info.get('hdmi_bandwidth', 48),
                'unit': 'Gbps'
            },
            {
                'id': '4.3.1',
                'name': '无线充电标准(Qi 2.0)',
                'required': False,  # 非强制要求
                'passed': device_info.get('wireless_charging_qi2', True),
                'standard': 'Qi 2.0'
            },
            {
                'id': '4.4.1',
                'name': '工业通信协议(OPC UA/Modbus)',
                'required': False,  # 仅对工业设备强制
                'passed': device_info.get('industrial_communication', True),
                'standard': 'OPC UA/Modbus',
                'applicable': device_info.get('device_type') == 'industrial'
            }
        ]
        
        # 过滤适用的检查项
        applicable_checks = [
            check for check in interface_checks 
            if check.get('applicable', True) or not check.get('applicable') is False
        ]
        
        total_checks = len(applicable_checks)
        passed_checks = sum(1 for check in applicable_checks if check['passed'] or not check['required'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'interface_compatibility',
            'name': '接口兼容性',
            'weight': self.check_weights['interface_compatibility'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['interface_compatibility'],
            'checks': applicable_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['interface_compatibility'],
            'score': compliance_rate * self.check_weights['interface_compatibility']
        }
    
    def check_reliability(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查可靠性 - 权重15%
        """
        print("正在检查可靠性...")
        
        # 模拟可靠性检查项目
        reliability_checks = [
            {
                'id': '5.1.1',
                'name': '高温试验(+55°C下正常工作)',
                'required': True,
                'passed': device_info.get('high_temp_test_passed', True),
                'standard': 'IEC 60068-2-2'
            },
            {
                'id': '5.1.2',
                'name': '低温试验(-10°C下正常工作)',
                'required': True,
                'passed': device_info.get('low_temp_test_passed', True),
                'standard': 'IEC 60068-2-1'
            },
            {
                'id': '5.1.3',
                'name': '湿热试验(40°C/93%RH下正常工作)',
                'required': True,
                'passed': device_info.get('humidity_test_passed', True),
                'standard': 'IEC 60068-2-78'
            },
            {
                'id': '5.1.4',
                'name': '振动试验(10-500Hz振动下正常工作)',
                'required': True,
                'passed': device_info.get('vibration_test_passed', True),
                'standard': 'IEC 60068-2-6'
            },
            {
                'id': '5.2.1',
                'name': '恒定应力加速寿命(MTBF≥50,000小时)',
                'required': True,
                'passed': device_info.get('mtbf', 60000) >= 50000,
                'standard': 'GB/T 5080.7-2012',
                'measured_value': device_info.get('mtbf', 60000),
                'unit': '小时'
            },
            {
                'id': '5.3.1',
                'name': '服务器能效(符合GB 20943-2013)',
                'required': False,  # 仅对服务器强制
                'passed': device_info.get('server_efficiency_compliant', True),
                'standard': 'GB 20943-2013',
                'applicable': device_info.get('device_type') == 'server'
            },
            {
                'id': '5.3.2',
                'name': '待机功耗(≤0.5W)',
                'required': True,
                'passed': device_info.get('standby_power', 0.3) <= 0.5,
                'standard': '能效标准',
                'measured_value': device_info.get('standby_power', 0.3),
                'unit': 'W'
            }
        ]
        
        # 过滤适用的检查项
        applicable_checks = [
            check for check in reliability_checks 
            if check.get('applicable', True) or not check.get('applicable') is False
        ]
        
        total_checks = len(applicable_checks)
        passed_checks = sum(1 for check in applicable_checks if check['passed'] or not check['required'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'reliability',
            'name': '可靠性',
            'weight': self.check_weights['reliability'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['reliability'],
            'checks': applicable_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['reliability'],
            'score': compliance_rate * self.check_weights['reliability']
        }
    
    def check_environmental_requirements(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查环保要求 - 权重10%
        """
        print("正在检查环保要求...")
        
        # 模拟环保要求检查项目
        environmental_checks = [
            {
                'id': '6.1.1',
                'name': '铅含量(≤1000ppm)',
                'required': True,
                'passed': device_info.get('lead_content', 500) <= 1000,
                'standard': 'RoHS (GB/T 26572-2011)',
                'measured_value': device_info.get('lead_content', 500),
                'unit': 'ppm'
            },
            {
                'id': '6.1.2',
                'name': '汞含量(≤1000ppm)',
                'required': True,
                'passed': device_info.get('mercury_content', 500) <= 1000,
                'standard': 'RoHS (GB/T 26572-2011)',
                'measured_value': device_info.get('mercury_content', 500),
                'unit': 'ppm'
            },
            {
                'id': '6.1.3',
                'name': '镉含量(≤100ppm)',
                'required': True,
                'passed': device_info.get('cadmium_content', 50) <= 100,
                'standard': 'RoHS (GB/T 26572-2011)',
                'measured_value': device_info.get('cadmium_content', 50),
                'unit': 'ppm'
            },
            {
                'id': '6.1.4',
                'name': '六价铬含量(≤1000ppm)',
                'required': True,
                'passed': device_info.get('hexavalent_chromium_content', 500) <= 1000,
                'standard': 'RoHS (GB/T 26572-2011)',
                'measured_value': device_info.get('hexavalent_chromium_content', 500),
                'unit': 'ppm'
            },
            {
                'id': '6.1.5',
                'name': '多溴联苯含量(≤1000ppm)',
                'required': True,
                'passed': device_info.get('pbb_content', 500) <= 1000,
                'standard': 'RoHS (GB/T 26572-2011)',
                'measured_value': device_info.get('pbb_content', 500),
                'unit': 'ppm'
            },
            {
                'id': '6.1.6',
                'name': '多溴二苯醚含量(≤1000ppm)',
                'required': True,
                'passed': device_info.get('pbde_content', 500) <= 1000,
                'standard': 'RoHS (GB/T 26572-2011)',
                'measured_value': device_info.get('pbde_content', 500),
                'unit': 'ppm'
            },
            {
                'id': '6.2.1',
                'name': '回收率(≥85%)',
                'required': True,
                'passed': device_info.get('recycling_rate', 90) >= 85,
                'standard': 'WEEE (GB/T 26125-2011)',
                'measured_value': device_info.get('recycling_rate', 90),
                'unit': '%'
            },
            {
                'id': '6.2.2',
                'name': '再利用率(≥75%)',
                'required': True,
                'passed': device_info.get('reuse_rate', 80) >= 75,
                'standard': 'WEEE (GB/T 26125-2011)',
                'measured_value': device_info.get('reuse_rate', 80),
                'unit': '%'
            }
        ]
        
        total_checks = len(environmental_checks)
        passed_checks = sum(1 for check in environmental_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'environmental_requirements',
            'name': '环保要求',
            'weight': self.check_weights['environmental_requirements'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['environmental_requirements'],
            'checks': environmental_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['environmental_requirements'],
            'score': compliance_rate * self.check_weights['environmental_requirements']
        }
    
    def generate_device_info_template(self) -> Dict[str, Any]:
        """
        生成设备信息模板，用于指导用户提供必要的检查数据
        """
        return {
            "device_id": "设备唯一标识符",
            "device_name": "设备名称",
            "device_type": "设备类型 (如: it, audio_video, communication, industrial, consumer)",
            "manufacturer": "制造商",
            "model": "型号",
            "production_date": "生产日期",
            # 安全性能相关
            "has_hara_analysis": True,  # 是否完成HARA分析
            "insulation_resistance": 20,  # 绝缘电阻 (MΩ)
            "leakage_current": 0.5,  # 漏电流 (mA)
            "dielectric_strength": 2000,  # 抗电强度 (V)
            "temperature_rise": 40,  # 温升 (K)
            "battery_safe": True,  # 电池是否安全
            # EMC性能相关
            "conducted_emission": 35,  # 传导发射 (dBμV)
            "radiated_emission": 25,  # 辐射发射 (dBμV/m)
            "esd_contact": 6,  # 静电放电接触放电 (kV)
            "esd_air": 10,  # 静电放电空气放电 (kV)
            "rf_immunity": 12,  # 射频电磁场辐射抗扰度 (V/m)
            "eft_burst": 3,  # 电快速瞬变脉冲群 (kV)
            "surge_immunity": True,  # 浪涌抗扰度
            # 接口兼容性相关
            "usb_type_c_reversible": True,  # USB Type-C是否可逆
            "usb_type_c_speed": 40,  # USB Type-C速度 (Gbps)
            "usb_type_c_power": 240,  # USB Type-C功率 (W)
            "usb_protocol_compatibility": True,  # USB协议兼容性
            "hdmi_8k_capability": True,  # HDMI 8K能力
            "hdmi_bandwidth": 48,  # HDMI带宽 (Gbps)
            "wireless_charging_qi2": True,  # 无线充电Qi 2.0
            "industrial_communication": True,  # 工业通信协议
            # 可靠性相关
            "high_temp_test_passed": True,  # 高温测试是否通过
            "low_temp_test_passed": True,  # 低温测试是否通过
            "humidity_test_passed": True,  # 湿热测试是否通过
            "vibration_test_passed": True,  # 振动测试是否通过
            "mtbf": 60000,  # 平均无故障时间 (小时)
            "server_efficiency_compliant": True,  # 服务器能效是否合规
            "standby_power": 0.3,  # 待机功耗 (W)
            # 环保要求相关
            "lead_content": 500,  # 铅含量 (ppm)
            "mercury_content": 500,  # 汞含量 (ppm)
            "cadmium_content": 50,  # 镉含量 (ppm)
            "hexavalent_chromium_content": 500,  # 六价铬含量 (ppm)
            "pbb_content": 500,  # 多溴联苯含量 (ppm)
            "pbde_content": 500,  # 多溴二苯醚含量 (ppm)
            "recycling_rate": 90,  # 回收率 (%)
            "reuse_rate": 80  # 再利用率 (%)
        }
    
    def run_comprehensive_check(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行综合检查
        """
        print(f"开始对设备 {device_info.get('device_id', 'Unknown')} 进行标准化检查...")
        print("="*60)
        
        # 执行各项检查
        safety_result = self.check_safety_performance(device_info)
        emc_result = self.check_emc_performance(device_info)
        interface_result = self.check_interface_compatibility(device_info)
        reliability_result = self.check_reliability(device_info)
        environmental_result = self.check_environmental_requirements(device_info)
        
        # 计算总体合规性
        all_results = [safety_result, emc_result, interface_result, reliability_result, environmental_result]
        overall_score = sum(result['score'] for result in all_results)
        overall_compliance_rate = overall_score / sum(self.check_weights.values())
        
        # 检查是否满足总体合规阈值
        overall_compliant = (
            overall_compliance_rate >= self.compliance_thresholds['overall'] and
            all(result['compliant'] for result in all_results)
        )
        
        # 生成详细报告
        report = {
            'device_info': {
                'id': device_info.get('device_id', 'Unknown'),
                'name': device_info.get('device_name', 'Unknown Device'),
                'type': device_info.get('device_type', 'Unknown Type'),
                'manufacturer': device_info.get('manufacturer', 'Unknown'),
                'model': device_info.get('model', 'Unknown Model'),
                'production_date': device_info.get('production_date', 'Unknown')
            },
            'check_date': datetime.datetime.now().isoformat(),
            'overall_compliance': {
                'score': overall_score,
                'compliance_rate': overall_compliance_rate,
                'threshold': self.compliance_thresholds['overall'],
                'compliant': overall_compliant,
                'status': 'COMPLIANT' if overall_compliant else 'NON-COMPLIANT'
            },
            'category_results': {
                'safety_performance': safety_result,
                'emc_performance': emc_result,
                'interface_compatibility': interface_result,
                'reliability': reliability_result,
                'environmental_requirements': environmental_result
            },
            'summary': {
                'total_categories': len(all_results),
                'compliant_categories': sum(1 for result in all_results if result['compliant']),
                'total_checks': sum(result['total_checks'] for result in all_results),
                'passed_checks': sum(result['passed_checks'] for result in all_results),
                'overall_compliance_rate': overall_compliance_rate
            },
            'recommendations': self._generate_recommendations(all_results)
        }
        
        return report
    
    def _generate_recommendations(self, all_results: List[Dict[str, Any]]) -> List[str]:
        """
        根据检查结果生成改进建议
        """
        recommendations = []
        
        for category_result in all_results:
            if not category_result['compliant']:
                category_name = category_result['name']
                non_compliant_checks = [
                    check for check in category_result['checks'] 
                    if not check['passed'] and check['required']
                ]
                
                for check in non_compliant_checks:
                    recommendations.append(
                        f"- {category_name}: {check['name']} 不符合要求，"
                        f"标准: {check['standard']}, "
                        f"实际值: {check.get('measured_value', 'N/A')}{check.get('unit', '')}"
                    )
        
        # 添加总体建议
        if not all(result['compliant'] for result in all_results):
            recommendations.append("- 建议优先解决安全性能和环保要求方面的问题，这两项是强制性要求")
            recommendations.append("- 对于EMC性能，建议加强电磁屏蔽和滤波设计")
            recommendations.append("- 考虑进行额外的环境可靠性测试")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """
        打印格式化的检查报告
        """
        print("\n" + "="*60)
        print("电子设备标准化检查报告")
        print("="*60)
        
        device = report['device_info']
        print(f"设备信息: {device['name']} ({device['model']}) - {device['manufacturer']}")
        print(f"设备ID: {device['id']}")
        print(f"检查日期: {report['check_date']}")
        print()
        
        # 总体合规性
        overall = report['overall_compliance']
        print(f"总体合规性: {overall['status']}")
        print(f"合规率: {overall['compliance_rate']:.2%} (阈值: {overall['threshold']:.2%})")
        print(f"综合得分: {overall['score']:.3f}")
        print()
        
        # 各类别详细结果
        print("各类别检查结果:")
        print("-" * 60)
        
        for category_key, category_result in report['category_results'].items():
            status = "✓" if category_result['compliant'] else "✗"
            print(f"{status} {category_result['name']}:")
            print(f"  合规率: {category_result['compliance_rate']:.2%} (阈值: {category_result['threshold']:.2%})")
            print(f"  检查项: {category_result['passed_checks']}/{category_result['total_checks']} 通过")
            print(f"  加权得分: {category_result['score']:.3f}")
            print()
        
        # 汇总信息
        summary = report['summary']
        print("检查汇总:")
        print(f"  检查类别: {summary['compliant_categories']}/{summary['total_categories']} 合规")
        print(f"  检查项目: {summary['passed_checks']}/{summary['total_checks']} 通过")
        print()
        
        # 改进建议
        if report['recommendations']:
            print("改进建议:")
            print("-" * 60)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        else:
            print("改进建议: 未发现需要改进的项目，设备符合所有标准要求！")
        
        print("="*60)
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """
        保存检查报告到JSON文件
        """
        if filename is None:
            device_id = report['device_info']['id'].replace(' ', '_').replace('/', '_')
            filename = f"electronic_device_check_report_{device_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存至: {filename}")


def main():
    """
    主函数，处理命令行参数并执行检查
    """
    parser = argparse.ArgumentParser(description='电子设备标准化快速检查工具')
    parser.add_argument('--device-info', type=str, help='设备信息JSON文件路径')
    parser.add_argument('--template', action='store_true', help='生成设备信息模板')
    parser.add_argument('--output', type=str, help='输出报告文件路径')
    parser.add_argument('--generate-sample', action='store_true', help='生成示例设备信息')
    
    args = parser.parse_args()
    
    checker = ElectronicDeviceStandardChecker()
    
    if args.template:
        # 生成模板
        template = checker.generate_device_info_template()
        print("设备信息模板 (请根据实际设备情况填写):")
        print(json.dumps(template, ensure_ascii=False, indent=2))
        return
    
    if args.generate_sample:
        # 生成示例设备信息
        sample_info = checker.generate_device_info_template()
        # 随机生成一些示例值
        sample_info.update({
            "device_id": "ED-2025-001",
            "device_name": "智能终端设备",
            "device_type": "consumer",
            "manufacturer": "示例电子有限公司",
            "model": "STD-2025",
            "production_date": "2025-01-01",
            # 随机生成一些值用于演示
            "insulation_resistance": round(random.uniform(15, 25), 2),
            "leakage_current": round(random.uniform(0.3, 0.7), 2),
            "dielectric_strength": random.randint(1800, 2500),
            "temperature_rise": random.randint(35, 50),
            "conducted_emission": round(random.uniform(30, 45), 2),
            "radiated_emission": round(random.uniform(20, 35), 2),
            "mtbf": random.randint(50000, 80000),
            "standby_power": round(random.uniform(0.2, 0.6), 2),
            "lead_content": random.randint(100, 800),
            "recycling_rate": random.randint(85, 95)
        })
        print("示例设备信息:")
        print(json.dumps(sample_info, ensure_ascii=False, indent=2))
        return
    
    # 如果没有提供设备信息文件，则使用默认示例
    if args.device_info:
        try:
            with open(args.device_info, 'r', encoding='utf-8') as f:
                device_info = json.load(f)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {args.device_info}")
            return
        except json.JSONDecodeError:
            print(f"错误: {args.device_info} 不是有效的JSON文件")
            return
    else:
        # 使用默认示例设备信息
        print("未提供设备信息文件，使用默认示例设备信息进行检查...")
        device_info = checker.generate_device_info_template()
        device_info.update({
            "device_id": "ED-2025-001",
            "device_name": "智能终端设备",
            "device_type": "consumer",
            "manufacturer": "示例电子有限公司",
            "model": "STD-2025",
            "production_date": "2025-01-01"
        })
    
    # 执行检查
    report = checker.run_comprehensive_check(device_info)
    
    # 打印报告
    checker.print_report(report)
    
    # 保存报告
    checker.save_report(report, args.output)


if __name__ == "__main__":
    main()