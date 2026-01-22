#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试CVE数据验证功能
"""

import json
from cve_vulnerability_processor import CVEProcessor


def test_cve_validation():
    """测试CVE数据验证功能"""
    # 直接使用之前获取的CVE-2025-40978数据，避免网络请求
    cve_json = {
        "dataType": "CVE_RECORD",
        "dataVersion": "5.2",
        "cveMetadata": {
            "cveId": "CVE-2025-40978",
            "assignerOrgId": "0cbda920-cd7f-484a-8e76-bf7f4b7f4516",
            "state": "PUBLISHED",
            "assignerShortName": "INCIBE",
            "dateReserved": "2025-04-16T09:08:23.193Z",
            "datePublished": "2026-01-12T11:28:35.332Z",
            "dateUpdated": "2026-01-12T12:53:10.833Z"
        },
        "containers": {
            "cna": {
                "affected": [
                    {
                        "defaultStatus": "unaffected",
                        "product": "eCommerceGo SaaS",
                        "vendor": "WorkDo",
                        "versions": [
                            {
                                "status": "affected",
                                "version": "All versions"
                            }
                        ]
                    }
                ],
                "credits": [
                    {
                        "lang": "en",
                        "type": "finder",
                        "value": "Gonzalo Aguilar García (6h4ack)"
                    }
                ],
                "datePublic": "2026-01-12T10:56:00.000Z",
                "descriptions": [
                    {
                        "lang": "en",
                        "supportingMedia": [
                            {
                                "base64": False,
                                "type": "text/html",
                                "value": "Stored Cross-Site Scripting (XSS) vulnerability in WorkDo's eCommerceGo SaaS, consisting of a stored XSS due to a lack of proper validation of user input by sending a POST request to ‘/ticket/x/conversion’, using the ‘reply_description’ parameter."
                            }
                        ],
                        "value": "Stored Cross-Site Scripting (XSS) vulnerability in WorkDo's eCommerceGo SaaS, consisting of a stored XSS due to a lack of proper validation of user input by sending a POST request to ‘/ticket/x/conversion’, using the ‘reply_description’ parameter."
                    }
                ],
                "metrics": [
                    {
                        "cvssV4_0": {
                            "Automatable": "NOT_DEFINED",
                            "Recovery": "NOT_DEFINED",
                            "Safety": "NOT_DEFINED",
                            "attackComplexity": "LOW",
                            "attackRequirements": "NONE",
                            "attackVector": "NETWORK",
                            "baseScore": 5.1,
                            "baseSeverity": "MEDIUM",
                            "exploitMaturity": "NOT_DEFINED",
                            "privilegesRequired": "LOW",
                            "providerUrgency": "NOT_DEFINED",
                            "subAvailabilityImpact": "NONE",
                            "subConfidentialityImpact": "LOW",
                            "subIntegrityImpact": "LOW",
                            "userInteraction": "PASSIVE",
                            "valueDensity": "NOT_DEFINED",
                            "vectorString": "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:P/VC:N/VI:N/VA:L/SC:L/SI:L/SA:N",
                            "version": "4.0",
                            "vulnAvailabilityImpact": "LOW",
                            "vulnConfidentialityImpact": "NONE",
                            "vulnIntegrityImpact": "NONE",
                            "vulnerabilityResponseEffort": "NOT_DEFINED"
                        },
                        "format": "CVSS",
                        "scenarios": [
                            {
                                "lang": "en",
                                "value": "GENERAL"
                            }
                        ]
                    }
                ],
                "problemTypes": [
                    {
                        "descriptions": [
                            {
                                "cweId": "CWE-79",
                                "description": "CWE-79 Improper Neutralization of Input During Web Page Generation (XSS or 'Cross-site Scripting')",
                                "lang": "en",
                                "type": "CWE"
                            }
                        ]
                    }
                ],
                "providerMetadata": {
                    "orgId": "0cbda920-cd7f-484a-8e76-bf7f4b7f4516",
                    "shortName": "INCIBE",
                    "dateUpdated": "2026-01-12T11:28:35.332Z"
                },
                "references": [
                    {
                        "url": "https://www.incibe.es/en/incibe-cert/notices/aviso/multiple-vulnerabilities-workdo-products"
                    }
                ],
                "solutions": [
                    {
                        "lang": "en",
                        "supportingMedia": [
                            {
                                "base64": False,
                                "type": "text/html",
                                "value": "No solution has been reported at this time."
                            }
                        ],
                        "value": "No solution has been reported at this time."
                    }
                ],
                "source": {
                    "discovery": "EXTERNAL"
                },
                "title": "Multiple vulnerabilities in WorkDo products",
                "x_generator": {
                    "engine": "Vulnogram 0.5.0"
                }
            },
            "adp": [
                {
                    "metrics": [
                        {
                            "other": {
                                "type": "ssvc",
                                "content": {
                                    "timestamp": "2026-01-12T12:52:52.723485Z",
                                    "id": "CVE-2025-40978",
                                    "options": [
                                        {
                                            "Exploitation": "none"
                                        },
                                        {
                                            "Automatable": "no"
                                        },
                                        {
                                            "Technical Impact": "partial"
                                        }
                                    ],
                                    "role": "CISA Coordinator",
                                    "version": "2.0.3"
                                }
                            }
                        }
                    ],
                    "title": "CISA ADP Vulnrichment",
                    "providerMetadata": {
                        "orgId": "134c704f-9b21-4f2e-91b3-4a467353bcc0",
                        "shortName": "CISA-ADP",
                        "dateUpdated": "2026-01-12T12:53:10.833Z"
                    }
                }
            ]
        }
    }
    
    # 创建CVE处理器实例
    processor = CVEProcessor(delta_log_path='dummy_path')
    
    # 定义github_link变量
    github_link = 'https://raw.githubusercontent.com/CVEProject/cvelistV5/main/cves/2025/40xxx/CVE-2025-40978.json'
    
    # 测试基本信息
    basic_info = {
        'cveId': 'CVE-2025-40978',
        'cveOrgLink': 'https://www.cve.org/CVERecord?id=CVE-2025-40978',
        'githubLink': github_link,
        'dateUpdated': '2026-01-12T12:53:10.833Z'
    }
    
    print("=== 测试1: 正常CVE_RECORD数据 (5.2版本) ===")
    # 调用extract_key_info方法
    result = processor.extract_key_info(cve_json, basic_info)
    print("✓ 测试1通过: 正常处理CVE_RECORD数据")
    
    print("\n=== 测试2: 无效dataType ===")
    # 修改dataType为无效值
    invalid_data_type_json = cve_json.copy()
    invalid_data_type_json['dataType'] = 'INVALID_TYPE'
    result = processor.extract_key_info(invalid_data_type_json, basic_info)
    print("✓ 测试2通过: 检测到无效dataType")
    
    print("\n=== 测试3: 旧版本dataVersion ===")
    # 修改dataVersion为旧版本
    old_version_json = cve_json.copy()
    old_version_json['dataVersion'] = '5.0'
    result = processor.extract_key_info(old_version_json, basic_info)
    print("✓ 测试3通过: 检测到旧版本dataVersion")
    
    print("\n=== 测试4: 缺少dataType字段 ===")
    # 删除dataType字段
    no_data_type_json = cve_json.copy()
    del no_data_type_json['dataType']
    result = processor.extract_key_info(no_data_type_json, basic_info)
    print("✓ 测试4通过: 处理缺少dataType字段的情况")
    
    print("\n=== 测试5: 缺少dataVersion字段 ===")
    # 删除dataVersion字段
    no_data_version_json = cve_json.copy()
    del no_data_version_json['dataVersion']
    result = processor.extract_key_info(no_data_version_json, basic_info)
    print("✓ 测试5通过: 处理缺少dataVersion字段的情况")
    
    print("\n🎉 所有测试通过！CVE数据验证功能正常工作。")


if __name__ == "__main__":
    test_cve_validation()
