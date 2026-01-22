#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
农业室外种植环境标准化快速检查工具
实现自动化合规检查功能，覆盖土壤、灌溉水、空气、病虫害、投入品、气象灾害六大核心领域
"""

import json
import argparse
import datetime
from typing import Dict, List, Any, Tuple
import random


class AgriculturalOutdoorEnvironmentStandardChecker:
    """
    农业室外种植环境标准化快速检查工具类
    涵盖土壤、灌溉水、空气、病虫害、投入品、气象灾害六大核心领域
    """
    
    def __init__(self):
        self.check_weights = {
            'soil_environment': 0.25,  # 土壤环境权重25%
            'irrigation_water_environment': 0.20,  # 灌溉水环境权重20%
            'air_quality': 0.15,  # 空气质量权重15%
            'pest_control': 0.15,  # 病虫害防控权重15%
            'input_risk': 0.15,  # 投入品风险权重15%
            'weather_disaster': 0.10  # 气象灾害防控权重10%
        }
        
        self.compliance_thresholds = {
            'overall': 0.95,  # 总体合规率阈值95%
            'soil_environment': 1.00,  # 土壤环境检查项通过率100%
            'irrigation_water_environment': 0.98,  # 灌溉水环境检查项通过率98%
            'air_quality': 0.95,  # 空气质量检查项通过率95%
            'pest_control': 0.90,  # 病虫害防控检查项通过率90%
            'input_risk': 1.00,  # 投入品风险检查项通过率100%
            'weather_disaster': 0.85  # 气象灾害防控检查项通过率85%
        }
    
    def check_soil_environment(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查土壤环境 - 权重25%
        """
        print("正在检查土壤环境...")
        
        # 模拟土壤环境检查项目
        soil_checks = [
            {
                'id': '2.1.1',
                'name': '镉含量检测',
                'required': True,
                'passed': field_info.get('soil_cadmium', 0.2) <= 0.3,
                'standard': 'GB 15618-2018',
                'measured_value': field_info.get('soil_cadmium', 0.2),
                'unit': 'mg/kg'
            },
            {
                'id': '2.1.2',
                'name': '汞含量检测',
                'required': True,
                'passed': field_info.get('soil_mercury', 0.3) <= 0.5,
                'standard': 'GB 15618-2018',
                'measured_value': field_info.get('soil_mercury', 0.3),
                'unit': 'mg/kg'
            },
            {
                'id': '2.1.3',
                'name': '砷含量检测',
                'required': True,
                'passed': field_info.get('soil_arsenic', 10) <= 15,
                'standard': 'GB 15618-2018',
                'measured_value': field_info.get('soil_arsenic', 10),
                'unit': 'mg/kg'
            },
            {
                'id': '2.1.4',
                'name': '土壤pH值',
                'required': True,
                'passed': 5.5 <= field_info.get('soil_ph', 6.5) <= 7.5,
                'standard': 'NY/T 391-2013',
                'measured_value': field_info.get('soil_ph', 6.5),
                'unit': ''
            },
            {
                'id': '2.1.5',
                'name': '土壤有机质',
                'required': True,
                'passed': field_info.get('soil_organic_matter', 18) >= 15,
                'standard': 'NY/T 391-2013',
                'measured_value': field_info.get('soil_organic_matter', 18),
                'unit': 'g/kg'
            },
            {
                'id': '2.1.6',
                'name': '土壤孔隙度',
                'required': True,
                'passed': field_info.get('soil_porosity', 55) >= 50,
                'standard': 'DB32/T 5042-2025',
                'measured_value': field_info.get('soil_porosity', 55),
                'unit': '%'
            },
            {
                'id': '2.1.7',
                'name': '土壤容重',
                'required': True,
                'passed': field_info.get('soil_bulk_density', 1.2) <= 1.3,
                'standard': 'NY/T 391-2013',
                'measured_value': field_info.get('soil_bulk_density', 1.2),
                'unit': 'g/cm³'
            },
            {
                'id': '2.1.8',
                'name': '全盐量检测',
                'required': True,
                'passed': field_info.get('soil_salinity', 0.8) <= 1.0,
                'standard': 'NY/T 391-2013',
                'measured_value': field_info.get('soil_salinity', 0.8),
                'unit': 'g/kg'
            }
        ]
        
        total_checks = len(soil_checks)
        passed_checks = sum(1 for check in soil_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'soil_environment',
            'name': '土壤环境',
            'weight': self.check_weights['soil_environment'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['soil_environment'],
            'checks': soil_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['soil_environment'],
            'score': compliance_rate * self.check_weights['soil_environment']
        }
    
    def check_irrigation_water_environment(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查灌溉水环境 - 权重20%
        """
        print("正在检查灌溉水环境...")
        
        # 模拟灌溉水环境检查项目
        water_checks = [
            {
                'id': '3.1.1',
                'name': '灌溉水pH值',
                'required': True,
                'passed': 5.5 <= field_info.get('water_ph', 7.0) <= 8.5,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_ph', 7.0),
                'unit': ''
            },
            {
                'id': '3.1.2',
                'name': '全盐量检测',
                'required': True,
                'passed': field_info.get('water_salinity', 800) <= 1000,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_salinity', 800),
                'unit': 'mg/L'
            },
            {
                'id': '3.1.3',
                'name': '化学需氧量(COD)',
                'required': True,
                'passed': field_info.get('water_cod', 45) <= 60,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_cod', 45),
                'unit': 'mg/L'
            },
            {
                'id': '3.2.1',
                'name': '镉含量检测',
                'required': True,
                'passed': field_info.get('water_cadmium', 0.005) <= 0.01,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_cadmium', 0.005),
                'unit': 'mg/L'
            },
            {
                'id': '3.2.2',
                'name': '铅含量检测',
                'required': True,
                'passed': field_info.get('water_lead', 0.05) <= 0.1,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_lead', 0.05),
                'unit': 'mg/L'
            },
            {
                'id': '3.2.3',
                'name': '汞含量检测',
                'required': True,
                'passed': field_info.get('water_mercury', 0.0005) <= 0.001,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_mercury', 0.0005),
                'unit': 'mg/L'
            },
            {
                'id': '3.2.4',
                'name': '砷含量检测',
                'required': True,
                'passed': field_info.get('water_arsenic', 0.03) <= 0.05,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_arsenic', 0.03),
                'unit': 'mg/L'
            },
            {
                'id': '3.3.1',
                'name': '旱作悬浮物',
                'required': True,
                'passed': field_info.get('water_suspended_solids_dryland', 80) <= 100,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_suspended_solids_dryland', 80),
                'unit': 'mg/L'
            },
            {
                'id': '3.3.2',
                'name': '水作悬浮物',
                'required': True,
                'passed': field_info.get('water_suspended_solids_paddy', 60) <= 80,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_suspended_solids_paddy', 60),
                'unit': 'mg/L'
            },
            {
                'id': '3.4.1',
                'name': '粪大肠菌群',
                'required': True,
                'passed': field_info.get('water_fecal_coliform', 5000) <= 10000,
                'standard': 'GB 5084-2021',
                'measured_value': field_info.get('water_fecal_coliform', 5000),
                'unit': '个/L'
            }
        ]
        
        total_checks = len(water_checks)
        passed_checks = sum(1 for check in water_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'irrigation_water_environment',
            'name': '灌溉水环境',
            'weight': self.check_weights['irrigation_water_environment'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['irrigation_water_environment'],
            'checks': water_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['irrigation_water_environment'],
            'score': compliance_rate * self.check_weights['irrigation_water_environment']
        }
    
    def check_air_quality(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查空气质量 - 权重15%
        """
        print("正在检查空气质量...")
        
        # 模拟空气质量检查项目
        air_checks = [
            {
                'id': '4.1.1',
                'name': '二氧化硫(SO₂)检测',
                'required': True,
                'passed': field_info.get('air_so2', 0.3) <= 0.5,
                'standard': 'GB 3095-2012',
                'measured_value': field_info.get('air_so2', 0.3),
                'unit': 'mg/m³'
            },
            {
                'id': '4.1.2',
                'name': '二氧化氮(NO₂)检测',
                'required': True,
                'passed': field_info.get('air_no2', 0.15) <= 0.24,
                'standard': 'GB 3095-2012',
                'measured_value': field_info.get('air_no2', 0.15),
                'unit': 'mg/m³'
            },
            {
                'id': '4.1.3',
                'name': '可吸入颗粒物(PM10)检测',
                'required': True,
                'passed': field_info.get('air_pm10', 0.1) <= 0.15,
                'standard': 'GB 3095-2012',
                'measured_value': field_info.get('air_pm10', 0.1),
                'unit': 'mg/m³'
            },
            {
                'id': '4.1.4',
                'name': '防护距离检查',
                'required': True,
                'passed': field_info.get('protection_distance', 600) >= 500,
                'standard': 'NY/T 391-2013',
                'measured_value': field_info.get('protection_distance', 600),
                'unit': 'm'
            },
            {
                'id': '4.2.1',
                'name': '日均光照检测',
                'required': True,
                'passed': field_info.get('daily_light_hours', 7) >= 6,
                'standard': 'NY/T 3442-2019',
                'measured_value': field_info.get('daily_light_hours', 7),
                'unit': '小时'
            },
            {
                'id': '4.3.1',
                'name': '排水系统功能',
                'required': True,
                'passed': field_info.get('drainage_function', True),
                'standard': 'GB/T 5084-2021'
            },
            {
                'id': '4.4.1',
                'name': '土壤田间持水量',
                'required': True,
                'passed': field_info.get('soil_field_capacity', 65) >= 60,
                'standard': 'NY/T 391-2013',
                'measured_value': field_info.get('soil_field_capacity', 65),
                'unit': '%'
            }
        ]
        
        total_checks = len(air_checks)
        passed_checks = sum(1 for check in air_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'air_quality',
            'name': '空气质量',
            'weight': self.check_weights['air_quality'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['air_quality'],
            'checks': air_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['air_quality'],
            'score': compliance_rate * self.check_weights['air_quality']
        }
    
    def check_pest_control(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查病虫害防控 - 权重15%
        """
        print("正在检查病虫害防控...")
        
        # 模拟病虫害防控检查项目
        pest_checks = [
            {
                'id': '5.1.1',
                'name': '农业防治措施',
                'required': True,
                'passed': field_info.get('agricultural_control', True),
                'standard': 'NY/T 1654-2008'
            },
            {
                'id': '5.1.2',
                'name': '物理防治措施',
                'required': True,
                'passed': field_info.get('physical_control', True),
                'standard': 'NY/T 1654-2008'
            },
            {
                'id': '5.1.3',
                'name': '生物防治措施',
                'required': True,
                'passed': field_info.get('biological_control', True),
                'standard': 'NY/T 1654-2008'
            },
            {
                'id': '5.1.4',
                'name': '化学防治规范',
                'required': True,
                'passed': field_info.get('chemical_control_compliant', True),
                'standard': 'NY/T 1654-2008'
            },
            {
                'id': '5.2.1',
                'name': '检疫性病虫害监测',
                'required': True,
                'passed': field_info.get('quarantine_monitoring', True),
                'standard': 'NY/T 1851-2010'
            },
            {
                'id': '5.2.2',
                'name': '外来入侵物种防控',
                'required': True,
                'passed': field_info.get('invasive_species_control', True),
                'standard': 'NY/T 1851-2010'
            },
            {
                'id': '5.4.1',
                'name': '病虫害监测体系',
                'required': True,
                'passed': field_info.get('pest_monitoring_system', True),
                'standard': 'NY/T 1654-2008'
            }
        ]
        
        total_checks = len(pest_checks)
        passed_checks = sum(1 for check in pest_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'pest_control',
            'name': '病虫害防控',
            'weight': self.check_weights['pest_control'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['pest_control'],
            'checks': pest_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['pest_control'],
            'score': compliance_rate * self.check_weights['pest_control']
        }
    
    def check_input_risk(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查投入品风险 - 权重15%
        """
        print("正在检查投入品风险...")
        
        # 模拟投入品风险检查项目
        input_checks = [
            {
                'id': '6.1.1',
                'name': '农药选择规范',
                'required': True,
                'passed': field_info.get('pesticide_selection_compliant', True),
                'standard': 'NY/T 393-2013'
            },
            {
                'id': '6.1.2',
                'name': '农药使用剂量',
                'required': True,
                'passed': field_info.get('pesticide_dose_compliant', True),
                'standard': 'NY/T 393-2013'
            },
            {
                'id': '6.1.3',
                'name': '农药使用时期',
                'required': True,
                'passed': field_info.get('pesticide_timing_compliant', True),
                'standard': 'NY/T 393-2013'
            },
            {
                'id': '6.1.4',
                'name': '安全间隔期遵守',
                'required': True,
                'passed': field_info.get('safety_interval_compliant', True),
                'standard': 'NY/T 393-2013'
            },
            {
                'id': '6.1.5',
                'name': '农药残留检测',
                'required': True,
                'passed': field_info.get('pesticide_residue_compliant', True),
                'standard': 'NY/T 393-2013'
            },
            {
                'id': '6.2.1',
                'name': '测土配方施肥',
                'required': True,
                'passed': field_info.get('soil_testing_fertilization', True),
                'standard': 'NY/T 393-2013'
            },
            {
                'id': '6.2.2',
                'name': '化肥用量控制',
                'required': True,
                'passed': field_info.get('fertilizer_usage_control', True),
                'standard': 'NY/T 393-2013'
            },
            {
                'id': '6.3.1',
                'name': '可降解农膜使用',
                'required': True,
                'passed': field_info.get('biodegradable_mulch_film', True),
                'standard': 'NY/T 393-2013'
            },
            {
                'id': '6.3.2',
                'name': '残膜回收率',
                'required': True,
                'passed': field_info.get('plastic_film_recycling_rate', 95) >= 90,
                'standard': 'NY/T 393-2013',
                'measured_value': field_info.get('plastic_film_recycling_rate', 95),
                'unit': '%'
            },
            {
                'id': '6.4.1',
                'name': '投入品台账管理',
                'required': True,
                'passed': field_info.get('input_record_management', True),
                'standard': 'NY/T 393-2013'
            }
        ]
        
        total_checks = len(input_checks)
        passed_checks = sum(1 for check in input_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'input_risk',
            'name': '投入品风险',
            'weight': self.check_weights['input_risk'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['input_risk'],
            'checks': input_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['input_risk'],
            'score': compliance_rate * self.check_weights['input_risk']
        }
    
    def check_weather_disaster(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查气象灾害防控 - 权重10%
        """
        print("正在检查气象灾害防控...")
        
        # 模拟气象灾害防控检查项目
        weather_checks = [
            {
                'id': '7.1.1',
                'name': '排水系统建设',
                'required': True,
                'passed': field_info.get('drainage_system', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.1.2',
                'name': '生态沟渠建设',
                'required': True,
                'passed': field_info.get('ecological_ditch', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.1.3',
                'name': '应急预案制定',
                'required': True,
                'passed': field_info.get('emergency_plan', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.2.1',
                'name': '节水灌溉设施',
                'required': True,
                'passed': field_info.get('water_saving_irrigation', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.2.2',
                'name': '抗旱品种选择',
                'required': True,
                'passed': field_info.get('drought_resistant_varieties', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.2.3',
                'name': '保墒措施实施',
                'required': True,
                'passed': field_info.get('soil_moisture_conservation', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.3.1',
                'name': '果树加固措施',
                'required': True,
                'passed': field_info.get('fruit_tree_reinforcement', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.3.2',
                'name': '防风林建设',
                'required': True,
                'passed': field_info.get('windbreak_forest', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.4.1',
                'name': '覆盖保温措施',
                'required': True,
                'passed': field_info.get('covering_insulation', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.4.2',
                'name': '防冻剂使用',
                'required': True,
                'passed': field_info.get('antifreeze_agent', True),
                'standard': 'GB/T 38757-2020'
            },
            {
                'id': '7.4.3',
                'name': '抗冻品种选择',
                'required': True,
                'passed': field_info.get('frost_resistant_varieties', True),
                'standard': 'GB/T 38757-2020'
            }
        ]
        
        total_checks = len(weather_checks)
        passed_checks = sum(1 for check in weather_checks if check['passed'])
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'category': 'weather_disaster',
            'name': '气象灾害防控',
            'weight': self.check_weights['weather_disaster'],
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_rate': compliance_rate,
            'threshold': self.compliance_thresholds['weather_disaster'],
            'checks': weather_checks,
            'compliant': compliance_rate >= self.compliance_thresholds['weather_disaster'],
            'score': compliance_rate * self.check_weights['weather_disaster']
        }
    
    def generate_field_info_template(self) -> Dict[str, Any]:
        """
        生成田块信息模板，用于指导用户提供必要的检查数据
        """
        return {
            "field_id": "田块唯一标识符",
            "field_name": "田块名称",
            "crop_type": "种植作物类型 (如: rice, wheat, vegetable)",
            "location": "田块位置",
            "area": "田块面积(亩)",
            "planting_date": "种植日期",
            # 土壤环境相关
            "soil_cadmium": 0.2,  # 土壤镉含量 (mg/kg)
            "soil_mercury": 0.3,  # 土壤汞含量 (mg/kg)
            "soil_arsenic": 10,  # 土壤砷含量 (mg/kg)
            "soil_ph": 6.5,  # 土壤pH值
            "soil_organic_matter": 18,  # 土壤有机质 (g/kg)
            "soil_porosity": 55,  # 土壤孔隙度 (%)
            "soil_bulk_density": 1.2,  # 土壤容重 (g/cm³)
            "soil_salinity": 0.8,  # 土壤全盐量 (g/kg)
            # 灌溉水环境相关
            "water_ph": 7.0,  # 灌溉水pH值
            "water_salinity": 800,  # 灌溉水全盐量 (mg/L)
            "water_cod": 45,  # 灌溉水化学需氧量 (mg/L)
            "water_cadmium": 0.005,  # 灌溉水镉含量 (mg/L)
            "water_lead": 0.05,  # 灌溉水铅含量 (mg/L)
            "water_mercury": 0.0005,  # 灌溉水汞含量 (mg/L)
            "water_arsenic": 0.03,  # 灌溉水砷含量 (mg/L)
            "water_suspended_solids_dryland": 80,  # 旱作悬浮物 (mg/L)
            "water_suspended_solids_paddy": 60,  # 水作悬浮物 (mg/L)
            "water_fecal_coliform": 5000,  # 粪大肠菌群 (个/L)
            # 空气环境相关
            "air_so2": 0.3,  # 二氧化硫 (mg/m³)
            "air_no2": 0.15,  # 二氧化氮 (mg/m³)
            "air_pm10": 0.1,  # 可吸入颗粒物 (mg/m³)
            "protection_distance": 600,  # 防护距离 (m)
            "daily_light_hours": 7,  # 日均光照小时数
            "drainage_function": True,  # 排水系统功能
            "soil_field_capacity": 65,  # 土壤田间持水量 (%)
            # 病虫害防控相关
            "agricultural_control": True,  # 农业防治措施
            "physical_control": True,  # 物理防治措施
            "biological_control": True,  # 生物防治措施
            "chemical_control_compliant": True,  # 化学防治规范
            "quarantine_monitoring": True,  # 检疫性病虫害监测
            "invasive_species_control": True,  # 外来入侵物种防控
            "pest_monitoring_system": True,  # 病虫害监测体系
            # 投入品风险相关
            "pesticide_selection_compliant": True,  # 农药选择规范
            "pesticide_dose_compliant": True,  # 农药使用剂量
            "pesticide_timing_compliant": True,  # 农药使用时期
            "safety_interval_compliant": True,  # 安全间隔期遵守
            "pesticide_residue_compliant": True,  # 农药残留检测
            "soil_testing_fertilization": True,  # 测土配方施肥
            "fertilizer_usage_control": True,  # 化肥用量控制
            "biodegradable_mulch_film": True,  # 可降解农膜使用
            "plastic_film_recycling_rate": 95,  # 残膜回收率 (%)
            "input_record_management": True,  # 投入品台账管理
            # 气象灾害防控相关
            "drainage_system": True,  # 排水系统建设
            "ecological_ditch": True,  # 生态沟渠建设
            "emergency_plan": True,  # 应急预案制定
            "water_saving_irrigation": True,  # 节水灌溉设施
            "drought_resistant_varieties": True,  # 抗旱品种选择
            "soil_moisture_conservation": True,  # 保墒措施实施
            "fruit_tree_reinforcement": True,  # 果树加固措施
            "windbreak_forest": True,  # 防风林建设
            "covering_insulation": True,  # 覆盖保温措施
            "antifreeze_agent": True,  # 防冻剂使用
            "frost_resistant_varieties": True  # 抗冻品种选择
        }
    
    def run_comprehensive_check(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行综合检查
        """
        print(f"开始对田块 {field_info.get('field_id', 'Unknown')} 进行标准化检查...")
        print("="*60)
        
        # 执行各项检查
        soil_result = self.check_soil_environment(field_info)
        water_result = self.check_irrigation_water_environment(field_info)
        air_result = self.check_air_quality(field_info)
        pest_result = self.check_pest_control(field_info)
        input_result = self.check_input_risk(field_info)
        weather_result = self.check_weather_disaster(field_info)
        
        # 计算总体合规性
        all_results = [soil_result, water_result, air_result, pest_result, input_result, weather_result]
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
                'type': field_info.get('crop_type', 'Unknown Type'),
                'location': field_info.get('location', 'Unknown'),
                'area': field_info.get('area', 'Unknown'),
                'planting_date': field_info.get('planting_date', 'Unknown')
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
                'soil_environment': soil_result,
                'irrigation_water_environment': water_result,
                'air_quality': air_result,
                'pest_control': pest_result,
                'input_risk': input_result,
                'weather_disaster': weather_result
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
            recommendations.append("- 建议优先解决土壤环境和投入品风险方面的问题，这两项是强制性要求")
            recommendations.append("- 对于灌溉水环境，建议加强水质净化和监测措施")
            recommendations.append("- 考虑加强病虫害绿色防控技术的应用")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """
        打印格式化的检查报告
        """
        print("\n" + "="*60)
        print("农业室外种植环境标准化检查报告")
        print("="*60)
        
        field = report['field_info']
        print(f"田块信息: {field['name']} ({field['type']}) - {field['location']}")
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
            filename = f"agricultural_environment_check_report_{field_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存至: {filename}")


def main():
    """
    主函数，处理命令行参数并执行检查
    """
    parser = argparse.ArgumentParser(description='农业室外种植环境标准化快速检查工具')
    parser.add_argument('--field-info', type=str, help='田块信息JSON文件路径')
    parser.add_argument('--template', action='store_true', help='生成田块信息模板')
    parser.add_argument('--output', type=str, help='输出报告文件路径')
    parser.add_argument('--generate-sample', action='store_true', help='生成示例田块信息')
    
    args = parser.parse_args()
    
    checker = AgriculturalOutdoorEnvironmentStandardChecker()
    
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
            "field_id": "FIELD-2025-001",
            "field_name": "示范田块",
            "crop_type": "rice",
            "location": "XX省XX市XX县XX村",
            "area": 10.5,
            "planting_date": "2025-03-15",
            # 随机生成一些值用于演示
            "soil_ph": round(random.uniform(5.5, 7.5), 2),
            "soil_organic_matter": round(random.uniform(15, 25), 1),
            "air_so2": round(random.uniform(0.1, 0.4), 2),
            "water_salinity": random.randint(700, 900),
            "plastic_film_recycling_rate": random.randint(90, 98)
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
            "field_id": "FIELD-2025-001",
            "field_name": "示范田块",
            "crop_type": "rice",
            "location": "XX省XX市XX县XX村",
            "area": 10.5,
            "planting_date": "2025-03-15"
        })
    
    # 执行检查
    report = checker.run_comprehensive_check(field_info)
    
    # 打印报告
    checker.print_report(report)
    
    # 保存报告
    checker.save_report(report, args.output)


if __name__ == "__main__":
    main()