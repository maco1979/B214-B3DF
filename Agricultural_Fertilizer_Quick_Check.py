#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
农业室内有机肥/室外化肥标准化快速检查工具
实现自动化合规检查功能，覆盖选型、检测、施用、监测、风险控制五大核心领域
"""

import json
import argparse
import datetime
from typing import Dict, List, Any, Tuple
import random


class AgriculturalFertilizerStandardChecker:
    """
    农业室内有机肥/室外化肥标准化快速检查工具类
    涵盖选型、检测、施用、监测、风险控制五大核心领域
    """
    
    def __init__(self):
        self.check_weights = {
            'indoor_organic_fertilizer_selection': 0.20,  # 室内有机肥选型权重20%
            'outdoor_chemical_fertilizer_selection': 0.20,  # 室外化肥选型权重20%
            'testing_verification': 0.20,  # 检测验证权重20%
            'application_operation': 0.20,  # 施用操作权重20%
            'monitoring_risk_control': 0.20  # 监测风险控制权重20%
        }
        
        self.compliance_thresholds = {
            'overall': 0.95,  # 总体合规率阈值95%
            'indoor_organic_fertilizer_selection': 1.00,  # 室内有机肥选型检查项通过率100%
            'outdoor_chemical_fertilizer_selection': 1.00,  # 室外化肥选型检查项通过率100%
            'testing_verification': 1.00,  # 检测验证检查项通过率100%
            'application_operation': 0.95,  # 施用操作检查项通过率95%
            'monitoring_risk_control': 0.90  # 监测风险控制检查项通过率90%
        }
    
    def check_indoor_organic_fertilizer_selection(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查室内有机肥选型 - 权重20%
        """
        print("正在检查室内有机肥选型...")
        
        # 模拟室内有机肥选型检查项目
        selection_checks = [
            {
                'id': '2.1.1',
                'name': '有机肥类型选择',
                'required': True,
                'passed': field_info.get('indoor_organic_fertilizer_standard_compliant', True),
                'standard': 'NY/T 525-2021',
                'measured_value': '符合标准',
                'unit': ''
            },
            {
                'id': '2.1.2',
                'name': '有机质含量',
                'required': True,
                'passed': field_info.get('indoor_organic_matter_content', 35) >= 30,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_organic_matter_content', 35),
                'unit': '%'
            },
            {
                'id': '2.1.3',
                'name': '总养分含量',
                'required': True,
                'passed': field_info.get('indoor_total_nutrients_content', 5.0) >= 4.0,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_total_nutrients_content', 5.0),
                'unit': '%'
            },
            {
                'id': '2.1.4',
                'name': '水分含量',
                'required': True,
                'passed': field_info.get('indoor_moisture_content', 25) <= 30,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_moisture_content', 25),
                'unit': '%'
            },
            {
                'id': '2.1.5',
                'name': 'pH值范围',
                'required': True,
                'passed': 5.5 <= field_info.get('indoor_ph_value', 7.0) <= 8.5,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_ph_value', 7.0),
                'unit': ''
            },
            {
                'id': '2.2.1',
                'name': '肥料登记证',
                'required': True,
                'passed': field_info.get('fertilizer_registration_certificate', True),
                'standard': 'NY/T 525-2021',
                'measured_value': '具备证书',
                'unit': ''
            },
            {
                'id': '2.2.2',
                'name': '生产许可证',
                'required': True,
                'passed': field_info.get('production_license', True),
                'standard': 'NY/T 525-2021',
                'measured_value': '具备证书',
                'unit': ''
            },
            {
                'id': '2.3.1',
                'name': '有机水溶肥料',
                'required': True,
                'passed': field_info.get('indoor_liquid_organic_fertilizer_standard', True),
                'standard': 'NY/T 3162-2017',
                'measured_value': '符合标准',
                'unit': ''
            },
            {
                'id': '2.3.2',
                'name': '有机质含量(液体)',
                'required': True,
                'passed': field_info.get('indoor_liquid_organic_carbon', 35) >= 30,
                'standard': 'NY/T 3162-2017',
                'measured_value': field_info.get('indoor_liquid_organic_carbon', 35),
                'unit': 'g/L'
            },
            {
                'id': '2.3.3',
                'name': '大量元素含量',
                'required': True,
                'passed': field_info.get('indoor_liquid_macronutrients', 65) >= 60,
                'standard': 'NY/T 3162-2017',
                'measured_value': field_info.get('indoor_liquid_macronutrients', 65),
                'unit': 'g/L'
            },
            {
                'id': '2.3.4',
                'name': '无菌化处理',
                'required': True,
                'passed': field_info.get('indoor_sterilization_process', True),
                'standard': 'NY/T 3162-2017',
                'measured_value': '已处理',
                'unit': ''
            }
        ]
        
        total_checks = len(selection_checks)
        passed_checks = sum(1 for check in selection_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'indoor_organic_fertilizer_selection',
            'name': '室内有机肥选型',
            'weight': self.check_weights['indoor_organic_fertilizer_selection'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['indoor_organic_fertilizer_selection'],
            'checks': selection_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['indoor_organic_fertilizer_selection'],
            'score': compliance_rate * self.check_weights['indoor_organic_fertilizer_selection']
        }
    
    def check_outdoor_chemical_fertilizer_selection(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查室外化肥选型 - 权重20%
        """
        print("正在检查室外化肥选型...")
        
        # 模拟室外化肥选型检查项目
        selection_checks = [
            {
                'id': '3.1.1',
                'name': '复合肥类型选择',
                'required': True,
                'passed': field_info.get('outdoor_compound_fertilizer_standard', True),
                'standard': 'GB/T 15063-2020',
                'measured_value': '符合标准',
                'unit': ''
            },
            {
                'id': '3.1.2',
                'name': '高浓度复合肥',
                'required': True,
                'passed': field_info.get('outdoor_high_concentration_nutrients', 42) >= 40,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_high_concentration_nutrients', 42),
                'unit': '%'
            },
            {
                'id': '3.1.3',
                'name': '中浓度复合肥',
                'required': True,
                'passed': field_info.get('outdoor_medium_concentration_nutrients', 32) >= 30,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_medium_concentration_nutrients', 32),
                'unit': '%'
            },
            {
                'id': '3.1.4',
                'name': '低浓度复合肥',
                'required': True,
                'passed': field_info.get('outdoor_low_concentration_nutrients', 26) >= 25,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_low_concentration_nutrients', 26),
                'unit': '%'
            },
            {
                'id': '3.1.5',
                'name': '单一养分最低含量',
                'required': True,
                'passed': field_info.get('outdoor_single_nutrient_min', 4.5) >= 4.0,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_single_nutrient_min', 4.5),
                'unit': '%'
            },
            {
                'id': '3.2.1',
                'name': '尿素选择',
                'required': True,
                'passed': field_info.get('outdoor_urea_standard', True),
                'standard': 'GB 2440-2017',
                'measured_value': '符合标准',
                'unit': ''
            },
            {
                'id': '3.2.2',
                'name': '磷酸一铵/二铵',
                'required': True,
                'passed': field_info.get('outdoor_phosphate_fertilizer_standard', True),
                'standard': 'GB/T 10205-2009',
                'measured_value': '符合标准',
                'unit': ''
            },
            {
                'id': '3.2.3',
                'name': '氯化钾/硫酸钾',
                'required': True,
                'passed': field_info.get('outdoor_potash_fertilizer_standard', True),
                'standard': 'GB 6549-2011/GB 20406-2006',
                'measured_value': '符合标准',
                'unit': ''
            },
            {
                'id': '3.2.4',
                'name': '水溶性磷比例',
                'required': True,
                'passed': field_info.get('outdoor_water_soluble_phosphorus_ratio', 65) >= 60,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_water_soluble_phosphorus_ratio', 65),
                'unit': '%'
            },
            {
                'id': '3.3.1',
                'name': '有机无机复混肥',
                'required': True,
                'passed': field_info.get('outdoor_organic_inorganic_standard', True),
                'standard': 'GB/T 18877-2020',
                'measured_value': '符合标准',
                'unit': ''
            },
            {
                'id': '3.3.2',
                'name': '有机质含量',
                'required': True,
                'passed': field_info.get('outdoor_organic_inorganic_organic_matter', 22) >= 20,
                'standard': 'GB/T 18877-2020',
                'measured_value': field_info.get('outdoor_organic_inorganic_organic_matter', 22),
                'unit': '%'
            },
            {
                'id': '3.3.3',
                'name': '总养分含量',
                'required': True,
                'passed': field_info.get('outdoor_organic_inorganic_nutrients', 26) >= 25,
                'standard': 'GB/T 18877-2020',
                'measured_value': field_info.get('outdoor_organic_inorganic_nutrients', 26),
                'unit': '%'
            },
            {
                'id': '3.3.4',
                'name': '氯离子含量',
                'required': True,
                'passed': field_info.get('outdoor_organic_inorganic_chloride', 2.5) <= 3.0,
                'standard': 'GB/T 18877-2020',
                'measured_value': field_info.get('outdoor_organic_inorganic_chloride', 2.5),
                'unit': '%'
            }
        ]
        
        total_checks = len(selection_checks)
        passed_checks = sum(1 for check in selection_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'outdoor_chemical_fertilizer_selection',
            'name': '室外化肥选型',
            'weight': self.check_weights['outdoor_chemical_fertilizer_selection'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['outdoor_chemical_fertilizer_selection'],
            'checks': selection_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['outdoor_chemical_fertilizer_selection'],
            'score': compliance_rate * self.check_weights['outdoor_chemical_fertilizer_selection']
        }
    
    def check_testing_verification(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查检测验证 - 权重20%
        """
        print("正在检查检测验证...")
        
        # 模拟检测验证检查项目
        testing_checks = [
            {
                'id': '4.1.1',
                'name': '有机质含量检测',
                'required': True,
                'passed': field_info.get('indoor_organic_matter_tested', 35) >= 30,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_organic_matter_tested', 35),
                'unit': '%'
            },
            {
                'id': '4.1.2',
                'name': '总养分检测',
                'required': True,
                'passed': field_info.get('indoor_total_nutrients_tested', 5.0) >= 4.0,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_total_nutrients_tested', 5.0),
                'unit': '%'
            },
            {
                'id': '4.1.3',
                'name': '水分检测',
                'required': True,
                'passed': field_info.get('indoor_moisture_tested', 25) <= 30,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_moisture_tested', 25),
                'unit': '%'
            },
            {
                'id': '4.1.4',
                'name': 'pH值检测',
                'required': True,
                'passed': 5.5 <= field_info.get('indoor_ph_tested', 7.0) <= 8.5,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_ph_tested', 7.0),
                'unit': ''
            },
            {
                'id': '4.1.5',
                'name': '粪大肠菌群检测',
                'required': True,
                'passed': field_info.get('indoor_fecal_coliform_tested', 50) <= 100,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_fecal_coliform_tested', 50),
                'unit': '个/g'
            },
            {
                'id': '4.1.6',
                'name': '蛔虫卵死亡率检测',
                'required': True,
                'passed': field_info.get('indoor_ascaris_mortality_tested', 98) >= 95,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_ascaris_mortality_tested', 98),
                'unit': '%'
            },
            {
                'id': '4.1.7',
                'name': '沙门氏菌检测',
                'required': True,
                'passed': field_info.get('indoor_salmonella_tested', 'not_detected') == 'not_detected',
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_salmonella_tested', 'not_detected'),
                'unit': ''
            },
            {
                'id': '4.1.8',
                'name': '种子发芽指数检测',
                'required': True,
                'passed': field_info.get('indoor_seed_germination_tested', 75) >= 70,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_seed_germination_tested', 75),
                'unit': '%'
            },
            {
                'id': '4.1.9',
                'name': '砷(As)检测',
                'required': True,
                'passed': field_info.get('indoor_arsenic_tested', 10) <= 15,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_arsenic_tested', 10),
                'unit': 'mg/kg'
            },
            {
                'id': '4.1.10',
                'name': '汞(Hg)检测',
                'required': True,
                'passed': field_info.get('indoor_mercury_tested', 1) <= 2,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_mercury_tested', 1),
                'unit': 'mg/kg'
            },
            {
                'id': '4.1.11',
                'name': '铅(Pb)检测',
                'required': True,
                'passed': field_info.get('indoor_lead_tested', 40) <= 50,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_lead_tested', 40),
                'unit': 'mg/kg'
            },
            {
                'id': '4.1.12',
                'name': '镉(Cd)检测',
                'required': True,
                'passed': field_info.get('indoor_cadmium_tested', 2) <= 3,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_cadmium_tested', 2),
                'unit': 'mg/kg'
            },
            {
                'id': '4.1.13',
                'name': '铬(Cr)检测',
                'required': True,
                'passed': field_info.get('indoor_chromium_tested', 120) <= 150,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_chromium_tested', 120),
                'unit': 'mg/kg'
            },
            {
                'id': '4.2.1',
                'name': '总养分检测',
                'required': True,
                'passed': field_info.get('outdoor_total_nutrients_tested', 42) >= 40,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_total_nutrients_tested', 42),
                'unit': '%'
            },
            {
                'id': '4.2.2',
                'name': '单一养分检测',
                'required': True,
                'passed': field_info.get('outdoor_single_nutrient_tested', 4.5) >= 4.0,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_single_nutrient_tested', 4.5),
                'unit': '%'
            },
            {
                'id': '4.2.3',
                'name': '水溶性磷检测',
                'required': True,
                'passed': field_info.get('outdoor_water_soluble_phosphorus_tested', 65) >= 60,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_water_soluble_phosphorus_tested', 65),
                'unit': '%'
            },
            {
                'id': '4.2.4',
                'name': '硝态氮检测',
                'required': True,
                'passed': field_info.get('outdoor_nitrate_nitrogen_tested', 2.0) >= 1.5,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_nitrate_nitrogen_tested', 2.0),
                'unit': '%'
            },
            {
                'id': '4.2.5',
                'name': '氯离子检测',
                'required': True,
                'passed': field_info.get('outdoor_chloride_tested', 2.5) <= 3.0,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_chloride_tested', 2.5),
                'unit': '%'
            },
            {
                'id': '4.2.6',
                'name': '水分检测',
                'required': True,
                'passed': field_info.get('outdoor_moisture_tested', 1.5) <= 2.0,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_moisture_tested', 1.5),
                'unit': '%'
            },
            {
                'id': '4.2.7',
                'name': '粒度检测',
                'required': True,
                'passed': field_info.get('outdoor_particle_size_tested', 92) >= 90,
                'standard': 'GB/T 15063-2020',
                'measured_value': field_info.get('outdoor_particle_size_tested', 92),
                'unit': '%'
            },
            {
                'id': '4.3.1',
                'name': '氨氮含量检测',
                'required': True,
                'passed': field_info.get('indoor_ammonia_nitrogen_tested', 0.3) <= 0.5,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_ammonia_nitrogen_tested', 0.3),
                'unit': '%'
            },
            {
                'id': '4.3.2',
                'name': '碳氮比检测',
                'required': True,
                'passed': field_info.get('indoor_carbon_nitrogen_ratio_tested', 20) <= 25,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_carbon_nitrogen_ratio_tested', 20),
                'unit': ':1'
            },
            {
                'id': '4.3.3',
                'name': '发热试验',
                'required': True,
                'passed': field_info.get('indoor_heating_test', 40) <= 45,
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_heating_test', 40),
                'unit': '℃'
            },
            {
                'id': '4.3.4',
                'name': '气味检测',
                'required': True,
                'passed': field_info.get('indoor_odor_test', 'no_odor') == 'no_odor',
                'standard': 'NY/T 525-2021',
                'measured_value': field_info.get('indoor_odor_test', 'no_odor'),
                'unit': ''
            }
        ]
        
        total_checks = len(testing_checks)
        passed_checks = sum(1 for check in testing_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'testing_verification',
            'name': '检测验证',
            'weight': self.check_weights['testing_verification'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['testing_verification'],
            'checks': testing_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['testing_verification'],
            'score': compliance_rate * self.check_weights['testing_verification']
        }
    
    def check_application_operation(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查施用操作 - 权重20%
        """
        print("正在检查施用操作...")
        
        # 模拟施用操作检查项目
        application_checks = [
            {
                'id': '5.1.1',
                'name': '基肥施用方法',
                'required': True,
                'passed': field_info.get('indoor_base_fertilization_method', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '均匀混合',
                'unit': ''
            },
            {
                'id': '5.1.2',
                'name': '基肥施用量',
                'required': True,
                'passed': 2.0 <= field_info.get('indoor_base_fertilization_amount', 2.5) <= 3.0,
                'standard': 'NY/T 394-2023',
                'measured_value': field_info.get('indoor_base_fertilization_amount', 2.5),
                'unit': '吨/亩'
            },
            {
                'id': '5.1.3',
                'name': '追肥施用方法',
                'required': True,
                'passed': field_info.get('indoor_top_dressing_method', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '腐熟浸出液稀释',
                'unit': ''
            },
            {
                'id': '5.1.4',
                'name': '追肥稀释比例',
                'required': True,
                'passed': 5 <= field_info.get('indoor_top_dressing_dilution', 7) <= 10,
                'standard': 'NY/T 394-2023',
                'measured_value': field_info.get('indoor_top_dressing_dilution', 7),
                'unit': '倍'
            },
            {
                'id': '5.1.5',
                'name': '无土栽培基质添加',
                'required': True,
                'passed': 15 <= field_info.get('indoor_hydroponic_addition', 18) <= 20,
                'standard': 'NY/T 394-2023',
                'measured_value': field_info.get('indoor_hydroponic_addition', 18),
                'unit': 'kg/m³'
            },
            {
                'id': '5.1.6',
                'name': '防氨气措施',
                'required': True,
                'passed': field_info.get('indoor_ammonia_prevention', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '充分腐熟',
                'unit': ''
            },
            {
                'id': '5.1.7',
                'name': '施用防护措施',
                'required': True,
                'passed': field_info.get('indoor_application_protection', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '佩戴防护用品',
                'unit': ''
            },
            {
                'id': '5.2.1',
                'name': '测土配方施肥',
                'required': True,
                'passed': field_info.get('outdoor_soil_testing_fertilization', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '按配方施肥',
                'unit': ''
            },
            {
                'id': '5.2.2',
                'name': '氮肥深施',
                'required': True,
                'passed': 10 <= field_info.get('outdoor_nitrogen_deep_application', 12) <= 15,
                'standard': 'NY/T 394-2023',
                'measured_value': field_info.get('outdoor_nitrogen_deep_application', 12),
                'unit': 'cm'
            },
            {
                'id': '5.2.3',
                'name': '磷肥集中施用',
                'required': True,
                'passed': field_info.get('outdoor_phosphorus_concentrated_application', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '集中施用',
                'unit': ''
            },
            {
                'id': '5.2.4',
                'name': '施肥均匀性',
                'required': True,
                'passed': field_info.get('outdoor_fertilization_uniformity', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '均匀分布',
                'unit': ''
            },
            {
                'id': '5.2.5',
                'name': '安全间隔期',
                'required': True,
                'passed': field_info.get('outdoor_safe_interval', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '满足间隔期',
                'unit': ''
            },
            {
                'id': '5.2.6',
                'name': '基肥与追肥比例',
                'required': True,
                'passed': field_info.get('outdoor_base_top_dressing_ratio', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '按作物需求比例',
                'unit': ''
            },
            {
                'id': '5.2.7',
                'name': '施肥时间控制',
                'required': True,
                'passed': field_info.get('outdoor_fertilization_timing', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '适期施肥',
                'unit': ''
            },
            {
                'id': '5.3.1',
                'name': '水稻施肥比例',
                'required': True,
                'passed': field_info.get('outdoor_rice_fertilization_ratio', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '基肥40%+分蘖肥30%+穗肥30%',
                'unit': ''
            },
            {
                'id': '5.3.2',
                'name': '小麦施肥比例',
                'required': True,
                'passed': field_info.get('outdoor_wheat_fertilization_ratio', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '基肥60%+拔节肥30%+孕穗肥10%',
                'unit': ''
            },
            {
                'id': '5.3.3',
                'name': '玉米施肥比例',
                'required': True,
                'passed': field_info.get('outdoor_corn_fertilization_ratio', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '基肥50%+大喇叭口期追肥50%',
                'unit': ''
            },
            {
                'id': '5.3.4',
                'name': '叶菜类施肥',
                'required': True,
                'passed': field_info.get('outdoor_leafy_vegetables_fertilization', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '氮肥为主，配合磷钾肥',
                'unit': ''
            },
            {
                'id': '5.3.5',
                'name': '果菜类施肥',
                'required': True,
                'passed': field_info.get('outdoor_fruit_vegetables_fertilization', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '磷钾肥比例增加，花期追肥',
                'unit': ''
            },
            {
                'id': '5.3.6',
                'name': '果树施肥方法',
                'required': True,
                'passed': 20 <= field_info.get('outdoor_fruit_tree_application_depth', 25) <= 30,
                'standard': 'NY/T 394-2023',
                'measured_value': field_info.get('outdoor_fruit_tree_application_depth', 25),
                'unit': 'cm'
            },
            {
                'id': '5.3.7',
                'name': '中药材施肥',
                'required': True,
                'passed': field_info.get('outdoor_medicinal_plants_fertilization', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '基肥为主，避免化肥过量',
                'unit': ''
            }
        ]
        
        total_checks = len(application_checks)
        passed_checks = sum(1 for check in application_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'application_operation',
            'name': '施用操作',
            'weight': self.check_weights['application_operation'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['application_operation'],
            'checks': application_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['application_operation'],
            'score': compliance_rate * self.check_weights['application_operation']
        }
    
    def check_monitoring_risk_control(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查监测风险控制 - 权重20%
        """
        print("正在检查监测风险控制...")
        
        # 模拟监测风险控制检查项目
        monitoring_checks = [
            {
                'id': '6.1.1',
                'name': '土壤养分监测',
                'required': True,
                'passed': field_info.get('soil_nutrient_monitoring', True),
                'standard': 'GB/T 45196-2025',
                'measured_value': '定期监测',
                'unit': ''
            },
            {
                'id': '6.1.2',
                'name': '水质监测',
                'required': True,
                'passed': field_info.get('water_quality_monitoring', True),
                'standard': 'GB/T 45196-2025',
                'measured_value': '防止污染',
                'unit': ''
            },
            {
                'id': '6.1.3',
                'name': '作物养分监测',
                'required': True,
                'passed': field_info.get('crop_nutrient_monitoring', True),
                'standard': 'GB/T 45196-2025',
                'measured_value': '营养状况评估',
                'unit': ''
            },
            {
                'id': '6.1.4',
                'name': '空气质量监测',
                'required': True,
                'passed': field_info.get('air_quality_monitoring', True),
                'standard': 'GB/T 45196-2025',
                'measured_value': '气体检测',
                'unit': ''
            },
            {
                'id': '6.1.5',
                'name': '土壤pH监测',
                'required': True,
                'passed': field_info.get('soil_ph_monitoring', True),
                'standard': 'GB/T 45196-2025',
                'measured_value': '适合作物生长',
                'unit': ''
            },
            {
                'id': '6.1.6',
                'name': '土壤盐分监测',
                'required': True,
                'passed': field_info.get('soil_salinity_monitoring', True),
                'standard': 'GB/T 45196-2025',
                'measured_value': '防止盐渍化',
                'unit': ''
            },
            {
                'id': '6.2.1',
                'name': '质量风险控制',
                'required': True,
                'passed': field_info.get('quality_risk_control', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '选择正规厂家产品',
                'unit': ''
            },
            {
                'id': '6.2.2',
                'name': '索要检测报告',
                'required': True,
                'passed': field_info.get('request_test_reports', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '每批次检测报告',
                'unit': ''
            },
            {
                'id': '6.2.3',
                'name': '抽样复检',
                'required': True,
                'passed': field_info.get('sampling_retest', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '抽样检测',
                'unit': ''
            },
            {
                'id': '6.2.4',
                'name': '环境风险控制',
                'required': True,
                'passed': field_info.get('environmental_risk_control', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '测土配方施肥',
                'unit': ''
            },
            {
                'id': '6.2.5',
                'name': '有机肥与化肥配合',
                'required': True,
                'passed': field_info.get('organic_chemical_combination', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '按比例配合',
                'unit': ''
            },
            {
                'id': '6.2.6',
                'name': '深施覆土',
                'required': True,
                'passed': field_info.get('deep_application_covering', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '防止流失',
                'unit': ''
            },
            {
                'id': '6.2.7',
                'name': '安全风险控制',
                'required': True,
                'passed': field_info.get('safety_risk_control', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '有机肥充分腐熟',
                'unit': ''
            },
            {
                'id': '6.2.8',
                'name': '化肥安全间隔期',
                'required': True,
                'passed': field_info.get('chemical_fertilizer_safe_interval', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '按要求执行',
                'unit': ''
            },
            {
                'id': '6.2.9',
                'name': '防护措施到位',
                'required': True,
                'passed': field_info.get('protection_measures_in_place', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '佩戴防护用品',
                'unit': ''
            },
            {
                'id': '6.3.1',
                'name': '作物生长监测',
                'required': True,
                'passed': field_info.get('crop_growth_monitoring', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '正常生长',
                'unit': ''
            },
            {
                'id': '6.3.2',
                'name': '无氨气中毒现象',
                'required': True,
                'passed': field_info.get('no_ammonia_poisoning', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '无中毒现象',
                'unit': ''
            },
            {
                'id': '6.3.3',
                'name': '养分利用率监测',
                'required': True,
                'passed': field_info.get('nutrient_use_efficiency_monitoring', 35) >= 30,
                'standard': 'NY/T 394-2023',
                'measured_value': field_info.get('nutrient_use_efficiency_monitoring', 35),
                'unit': '%'
            },
            {
                'id': '6.3.4',
                'name': '环境影响评估',
                'required': True,
                'passed': field_info.get('environmental_impact_assessment', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '无污染',
                'unit': ''
            },
            {
                'id': '6.3.5',
                'name': '产量效果评估',
                'required': True,
                'passed': field_info.get('yield_effect_assessment', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '产量提升',
                'unit': ''
            },
            {
                'id': '6.3.6',
                'name': '品质效果评估',
                'required': True,
                'passed': field_info.get('quality_effect_assessment', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '品质达标',
                'unit': ''
            },
            {
                'id': '6.3.7',
                'name': '持续改进措施',
                'required': True,
                'passed': field_info.get('continuous_improvement_measures', True),
                'standard': 'NY/T 394-2023',
                'measured_value': '定期评估改进',
                'unit': ''
            }
        ]
        
        total_checks = len(monitoring_checks)
        passed_checks = sum(1 for check in monitoring_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'monitoring_risk_control',
            'name': '监测风险控制',
            'weight': self.check_weights['monitoring_risk_control'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['monitoring_risk_control'],
            'checks': monitoring_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['monitoring_risk_control'],
            'score': compliance_rate * self.check_weights['monitoring_risk_control']
        }
    
    def generate_field_info_template(self) -> Dict[str, Any]:
        """
        生成田块信息模板，用于指导用户提供必要的检查数据
        """
        return {
            "field_id": "田块唯一标识符",
            "field_name": "田块名称",
            "field_type": "种植类型 (indoor/outdoor)",
            "location": "田块位置",
            "area": "田块面积(亩)",
            "crop_type": "种植作物类型",
            # 室内有机肥选型相关
            "indoor_organic_fertilizer_standard_compliant": True,  # 是否符合NY/T 525-2021标准
            "indoor_organic_matter_content": 35,  # 有机质含量 (%)
            "indoor_total_nutrients_content": 5.0,  # 总养分含量 (%)
            "indoor_moisture_content": 25,  # 水分含量 (%)
            "indoor_ph_value": 7.0,  # pH值
            "fertilizer_registration_certificate": True,  # 肥料登记证
            "production_license": True,  # 生产许可证
            "indoor_liquid_organic_fertilizer_standard": True,  # 有机水溶肥料标准
            "indoor_liquid_organic_carbon": 35,  # 有机质含量(液体) (g/L)
            "indoor_liquid_macronutrients": 65,  # 大量元素含量 (g/L)
            "indoor_sterilization_process": True,  # 无菌化处理
            # 室外化肥选型相关
            "outdoor_compound_fertilizer_standard": True,  # 复合肥标准
            "outdoor_high_concentration_nutrients": 42,  # 高浓度养分 (%)
            "outdoor_medium_concentration_nutrients": 32,  # 中浓度养分 (%)
            "outdoor_low_concentration_nutrients": 26,  # 低浓度养分 (%)
            "outdoor_single_nutrient_min": 4.5,  # 单一养分最低含量 (%)
            "outdoor_urea_standard": True,  # 尿素标准
            "outdoor_phosphate_fertilizer_standard": True,  # 磷酸肥标准
            "outdoor_potash_fertilizer_standard": True,  # 钾肥标准
            "outdoor_water_soluble_phosphorus_ratio": 65,  # 水溶性磷比例 (%)
            "outdoor_organic_inorganic_standard": True,  # 有机无机复混肥标准
            "outdoor_organic_inorganic_organic_matter": 22,  # 有机质含量 (%)
            "outdoor_organic_inorganic_nutrients": 26,  # 总养分含量 (%)
            "outdoor_organic_inorganic_chloride": 2.5,  # 氯离子含量 (%)
            # 检测验证相关
            "indoor_organic_matter_tested": 35,  # 有机质含量检测 (%)
            "indoor_total_nutrients_tested": 5.0,  # 总养分检测 (%)
            "indoor_moisture_tested": 25,  # 水分检测 (%)
            "indoor_ph_tested": 7.0,  # pH值检测
            "indoor_fecal_coliform_tested": 50,  # 粪大肠菌群检测 (个/g)
            "indoor_ascaris_mortality_tested": 98,  # 蛔虫卵死亡率检测 (%)
            "indoor_salmonella_tested": "not_detected",  # 沙门氏菌检测
            "indoor_seed_germination_tested": 75,  # 种子发芽指数检测 (%)
            "indoor_arsenic_tested": 10,  # 砷检测 (mg/kg)
            "indoor_mercury_tested": 1,  # 汞检测 (mg/kg)
            "indoor_lead_tested": 40,  # 铅检测 (mg/kg)
            "indoor_cadmium_tested": 2,  # 镉检测 (mg/kg)
            "indoor_chromium_tested": 120,  # 铬检测 (mg/kg)
            "outdoor_total_nutrients_tested": 42,  # 总养分检测 (%)
            "outdoor_single_nutrient_tested": 4.5,  # 单一养分检测 (%)
            "outdoor_water_soluble_phosphorus_tested": 65,  # 水溶性磷检测 (%)
            "outdoor_nitrate_nitrogen_tested": 2.0,  # 硝态氮检测 (%)
            "outdoor_chloride_tested": 2.5,  # 氯离子检测 (%)
            "outdoor_moisture_tested": 1.5,  # 水分检测 (%)
            "outdoor_particle_size_tested": 92,  # 粒度检测 (%)
            "indoor_ammonia_nitrogen_tested": 0.3,  # 氨氮含量检测 (%)
            "indoor_carbon_nitrogen_ratio_tested": 20,  # 碳氮比检测
            "indoor_heating_test": 40,  # 发热试验 (℃)
            "indoor_odor_test": "no_odor",  # 气味检测
            # 施用操作相关
            "indoor_base_fertilization_method": True,  # 基肥施用方法
            "indoor_base_fertilization_amount": 2.5,  # 基肥施用量 (吨/亩)
            "indoor_top_dressing_method": True,  # 追肥施用方法
            "indoor_top_dressing_dilution": 7,  # 追肥稀释比例 (倍)
            "indoor_hydroponic_addition": 18,  # 无土栽培基质添加 (kg/m³)
            "indoor_ammonia_prevention": True,  # 防氨气措施
            "indoor_application_protection": True,  # 施用防护措施
            "outdoor_soil_testing_fertilization": True,  # 测土配方施肥
            "outdoor_nitrogen_deep_application": 12,  # 氮肥深施 (cm)
            "outdoor_phosphorus_concentrated_application": True,  # 磷肥集中施用
            "outdoor_fertilization_uniformity": True,  # 施肥均匀性
            "outdoor_safe_interval": True,  # 安全间隔期
            "outdoor_base_top_dressing_ratio": True,  # 基肥与追肥比例
            "outdoor_fertilization_timing": True,  # 施肥时间控制
            "outdoor_rice_fertilization_ratio": True,  # 水稻施肥比例
            "outdoor_wheat_fertilization_ratio": True,  # 小麦施肥比例
            "outdoor_corn_fertilization_ratio": True,  # 玉米施肥比例
            "outdoor_leafy_vegetables_fertilization": True,  # 叶菜类施肥
            "outdoor_fruit_vegetables_fertilization": True,  # 果菜类施肥
            "outdoor_fruit_tree_application_depth": 25,  # 果树施肥深度 (cm)
            "outdoor_medicinal_plants_fertilization": True,  # 中药材施肥
            # 监测风险控制相关
            "soil_nutrient_monitoring": True,  # 土壤养分监测
            "water_quality_monitoring": True,  # 水质监测
            "crop_nutrient_monitoring": True,  # 作物养分监测
            "air_quality_monitoring": True,  # 空气质量监测
            "soil_ph_monitoring": True,  # 土壤pH监测
            "soil_salinity_monitoring": True,  # 土壤盐分监测
            "quality_risk_control": True,  # 质量风险控制
            "request_test_reports": True,  # 索要检测报告
            "sampling_retest": True,  # 抽样复检
            "environmental_risk_control": True,  # 环境风险控制
            "organic_chemical_combination": True,  # 有机肥与化肥配合
            "deep_application_covering": True,  # 深施覆土
            "safety_risk_control": True,  # 安全风险控制
            "chemical_fertilizer_safe_interval": True,  # 化肥安全间隔期
            "protection_measures_in_place": True,  # 防护措施到位
            "crop_growth_monitoring": True,  # 作物生长监测
            "no_ammonia_poisoning": True,  # 无氨气中毒现象
            "nutrient_use_efficiency_monitoring": 35,  # 养分利用率监测 (%)
            "environmental_impact_assessment": True,  # 环境影响评估
            "yield_effect_assessment": True,  # 产量效果评估
            "quality_effect_assessment": True,  # 品质效果评估
            "continuous_improvement_measures": True  # 持续改进措施
        }
    
    def run_comprehensive_check(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行综合检查
        """
        print(f"开始对田块 {field_info.get('field_id', 'Unknown')} 进行肥料标准化检查...")
        print("="*60)
        
        # 执行各项检查
        indoor_selection_result = self.check_indoor_organic_fertilizer_selection(field_info)
        outdoor_selection_result = self.check_outdoor_chemical_fertilizer_selection(field_info)
        testing_result = self.check_testing_verification(field_info)
        application_result = self.check_application_operation(field_info)
        monitoring_result = self.check_monitoring_risk_control(field_info)
        
        # 计算总体合规性
        all_results = [indoor_selection_result, outdoor_selection_result, testing_result, application_result, monitoring_result]
        overall_score = sum(result['score'] for result in all_results)
        overall_compliance_rate = overall_score / sum(self.check_weights.values())
        
        # 检查是否满足总体合规阈值
        overall_compliant = (
            overall_compliance_rate >= self.compliance_thresholds['overall'] and
            all(result['compliant'] for result in all_results)
        )
        
        # 生成详细报告
        report = {
            'field_info': {
                'id': field_info.get('field_id', 'Unknown'),
                'name': field_info.get('field_name', 'Unknown Field'),
                'type': field_info.get('field_type', 'Unknown Type'),
                'crop_type': field_info.get('crop_type', 'Unknown Crop'),
                'location': field_info.get('location', 'Unknown'),
                'area': field_info.get('area', 'Unknown')
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
                'indoor_organic_fertilizer_selection': indoor_selection_result,
                'outdoor_chemical_fertilizer_selection': outdoor_selection_result,
                'testing_verification': testing_result,
                'application_operation': application_result,
                'monitoring_risk_control': monitoring_result
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
            recommendations.append("- 建议优先解决检测验证方面的问题，确保所有肥料产品均经过质量检测")
            recommendations.append("- 加强施用操作规范性，确保按标准方法进行施肥")
            recommendations.append("- 完善监测风险控制体系，建立定期监测机制")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """
        打印格式化的检查报告
        """
        print("\n" + "="*60)
        print("农业室内有机肥/室外化肥标准化检查报告")
        print("="*60)
        
        field = report['field_info']
        print(f"田块信息: {field['name']} ({field['type']}) - {field['crop_type']}")
        print(f"田块ID: {field['id']}")
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
            print("改进建议: 未发现需要改进的项目，田块符合所有标准要求！")
        
        print("="*60)
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """
        保存检查报告到JSON文件
        """
        if filename is None:
            field_id = report['field_info']['id'].replace(' ', '_').replace('/', '_')
            filename = f"agricultural_fertilizer_check_report_{field_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存至: {filename}")


def main():
    """
    主函数，处理命令行参数并执行检查
    """
    parser = argparse.ArgumentParser(description='农业室内有机肥/室外化肥标准化快速检查工具')
    parser.add_argument('--field-info', type=str, help='田块信息JSON文件路径')
    parser.add_argument('--template', action='store_true', help='生成田块信息模板')
    parser.add_argument('--output', type=str, help='输出报告文件路径')
    parser.add_argument('--generate-sample', action='store_true', help='生成示例田块信息')
    
    args = parser.parse_args()
    
    checker = AgriculturalFertilizerStandardChecker()
    
    if args.template:
        # 生成模板
        template = checker.generate_field_info_template()
        print("田块信息模板 (请根据实际田块情况填写):")
        print(json.dumps(template, ensure_ascii=False, indent=2))
        return
    
    if args.generate_sample:
        # 生成示例田块信息
        sample_info = checker.generate_field_info_template()
        # 随机生成一些示例值用于演示
        sample_info.update({
            "field_id": "FERT-2025-001",
            "field_name": "示范田块",
            "field_type": "indoor",
            "crop_type": "tomato",
            "location": "XX省XX市XX县XX村",
            "area": 1.5,
            # 随机生成一些值用于演示
            "indoor_organic_matter_content": round(random.uniform(30, 40), 1),
            "indoor_total_nutrients_content": round(random.uniform(4.0, 6.0), 1),
            "outdoor_total_nutrients_tested": round(random.uniform(40, 45), 1),
            "nutrient_use_efficiency_monitoring": round(random.uniform(30, 50), 1)
        })
        print("示例田块信息:")
        print(json.dumps(sample_info, ensure_ascii=False, indent=2))
        return
    
    # 如果没有提供田块信息文件，则使用默认示例
    if args.field_info:
        try:
            with open(args.field_info, 'r', encoding='utf-8') as f:
                field_info = json.load(f)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {args.field_info}")
            return
        except json.JSONDecodeError:
            print(f"错误: {args.field_info} 不是有效的JSON文件")
            return
    else:
        # 使用默认示例田块信息
        print("未提供田块信息文件，使用默认示例田块信息进行检查...")
        field_info = checker.generate_field_info_template()
        field_info.update({
            "field_id": "FERT-2025-001",
            "field_name": "示范田块",
            "field_type": "indoor",
            "crop_type": "tomato",
            "location": "XX省XX市XX县XX村",
            "area": 1.5
        })
    
    # 执行检查
    report = checker.run_comprehensive_check(field_info)
    
    # 打印报告
    checker.print_report(report)
    
    # 保存报告
    checker.save_report(report, args.output)


if __name__ == "__main__":
    main()