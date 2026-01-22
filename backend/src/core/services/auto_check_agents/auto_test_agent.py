"""自动测试智能体
自动化运行单元测试、集成测试等，确保代码的正确性和稳定性
"""

from typing import Dict, List, Any
import logging
import subprocess
import os
import tempfile
import json
from datetime import datetime
import xml.etree.ElementTree as ET

# 配置日志
logger = logging.getLogger(__name__)


class AutoTestAgent:
    """自动测试智能体
    
    功能：
    1. 运行单元测试
    2. 运行集成测试
    3. 生成测试报告
    4. 支持多种测试框架
    5. 支持测试结果分析
    """
    
    def __init__(self):
        """初始化自动测试智能体"""
        self.name = "auto_test_agent"
        self.type = "code"
        self.capabilities = ["unit_test", "integration_test", "test_report", "test_analysis"]
        self.supported_frameworks = ["unittest", "pytest"]
        logger.info(f"{self.name} 初始化完成，支持的测试框架: {self.supported_frameworks}")
    
    def run_unit_tests(self, test_path: str, framework: str = "pytest", 
                     test_pattern: str = "test_*.py", report_format: str = "json") -> Dict[str, Any]:
        """运行单元测试
        
        Args:
            test_path: 测试文件或目录路径
            framework: 测试框架（unittest或pytest）
            test_pattern: 测试文件匹配模式
            report_format: 报告格式（json或xml）
            
        Returns:
            测试结果
        """
        logger.info(f"开始运行单元测试，路径: {test_path}，框架: {framework}，报告格式: {report_format}")
        
        # 检查测试框架是否支持
        if framework not in self.supported_frameworks:
            return {
                "status": "failed",
                "error": f"不支持的测试框架: {framework}，支持的框架: {self.supported_frameworks}"
            }
        
        # 检查测试路径是否存在
        if not os.path.exists(test_path):
            return {
                "status": "failed",
                "error": f"测试路径不存在: {test_path}"
            }
        
        # 根据测试框架运行测试
        if framework == "pytest":
            return self._run_pytest(test_path, test_pattern, report_format)
        elif framework == "unittest":
            return self._run_unittest(test_path, test_pattern, report_format)
    
    def _run_pytest(self, test_path: str, test_pattern: str = "test_*.py", 
                   report_format: str = "json") -> Dict[str, Any]:
        """使用pytest运行测试
        
        Args:
            test_path: 测试文件或目录路径
            test_pattern: 测试文件匹配模式
            report_format: 报告格式（json或xml）
            
        Returns:
            测试结果
        """
        try:
            # 检查pytest是否安装
            subprocess.run(
                ["pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
        except FileNotFoundError:
            return {
                "status": "failed",
                "error": "pytest未安装，请先安装pytest: pip install pytest"
            }
        
        try:
            # 创建临时报告文件
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{report_format}', delete=False) as f:
                report_file = f.name
            
            # 构建pytest命令
            cmd = ["pytest"]
            
            # 添加测试文件匹配模式
            if os.path.isdir(test_path):
                cmd.append(f"{test_path}/{test_pattern}")
            else:
                cmd.append(test_path)
            
            # 添加报告选项
            if report_format == "json":
                cmd.extend(["--json-report", f"--json-report-file={report_file}"])
            elif report_format == "xml":
                cmd.extend(["--junit-xml", report_file])
            
            # 添加详细输出
            cmd.extend(["-v"])
            
            logger.debug(f"运行pytest命令: {' '.join(cmd)}")
            
            # 运行测试
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            # 解析测试结果
            test_result = {
                "status": "success" if result.returncode == 0 else "failed",
                "framework": "pytest",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "report_path": report_file
            }
            
            # 解析报告文件
            if os.path.exists(report_file):
                if report_format == "json":
                    with open(report_file, "r", encoding="utf-8") as f:
                        report_data = json.load(f)
                    test_result["report"] = report_data
                    test_result["summary"] = self._parse_pytest_json_report(report_data)
                elif report_format == "xml":
                    tree = ET.parse(report_file)
                    root = tree.getroot()
                    test_result["report"] = ET.tostring(root, encoding="unicode")
                    test_result["summary"] = self._parse_pytest_xml_report(root)
            
            logger.info(f"pytest测试完成，状态: {test_result['status']}，退出码: {result.returncode}")
            return test_result
            
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": "测试运行超时",
                "framework": "pytest"
            }
        except Exception as e:
            logger.error(f"运行pytest测试失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "framework": "pytest"
            }
    
    def _parse_pytest_json_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析pytest JSON报告
        
        Args:
            report_data: pytest JSON报告数据
            
        Returns:
            报告摘要
        """
        summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0.0
        }
        
        # 从pytest-json-report中提取信息
        if "summary" in report_data:
            summary["total_tests"] = report_data["summary"].get("total", 0)
            summary["passed"] = report_data["summary"].get("passed", 0)
            summary["failed"] = report_data["summary"].get("failed", 0)
            summary["skipped"] = report_data["summary"].get("skipped", 0)
            summary["duration"] = report_data["summary"].get("duration", 0.0)
        
        return summary
    
    def _parse_pytest_xml_report(self, root: ET.Element) -> Dict[str, Any]:
        """解析pytest XML报告
        
        Args:
            root: XML根元素
            
        Returns:
            报告摘要
        """
        summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0.0
        }
        
        # 从JUnit XML报告中提取信息
        summary["total_tests"] = int(root.attrib.get("tests", 0))
        summary["failed"] = int(root.attrib.get("failures", 0))
        summary["skipped"] = int(root.attrib.get("skipped", 0))
        summary["passed"] = summary["total_tests"] - summary["failed"] - summary["skipped"]
        summary["duration"] = float(root.attrib.get("time", 0.0))
        
        return summary
    
    def _run_unittest(self, test_path: str, test_pattern: str = "test_*.py", 
                     report_format: str = "json") -> Dict[str, Any]:
        """使用unittest运行测试
        
        Args:
            test_path: 测试文件或目录路径
            test_pattern: 测试文件匹配模式
            report_format: 报告格式（json或xml）
            
        Returns:
            测试结果
        """
        try:
            # 创建临时报告文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                report_file = f.name
            
            # 构建unittest命令
            cmd = ["python", "-m", "xmlrunner", "discover"]
            
            if os.path.isdir(test_path):
                cmd.extend(["-s", test_path, "-p", test_pattern])
            else:
                cmd.extend(["-s", os.path.dirname(test_path), "-p", os.path.basename(test_path)])
            
            cmd.extend(["-o", report_file])
            
            logger.debug(f"运行unittest命令: {' '.join(cmd)}")
            
            # 运行测试
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            # 解析测试结果
            test_result = {
                "status": "success" if result.returncode == 0 else "failed",
                "framework": "unittest",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "report_path": report_file
            }
            
            # 解析XML报告
            if os.path.exists(report_file):
                tree = ET.parse(report_file)
                root = tree.getroot()
                test_result["report"] = ET.tostring(root, encoding="unicode")
                test_result["summary"] = self._parse_unittest_xml_report(root)
            
            logger.info(f"unittest测试完成，状态: {test_result['status']}，退出码: {result.returncode}")
            return test_result
            
        except FileNotFoundError:
            return {
                "status": "failed",
                "error": "xmlrunner未安装，请先安装xmlrunner: pip install unittest-xml-reporting",
                "framework": "unittest"
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": "测试运行超时",
                "framework": "unittest"
            }
        except Exception as e:
            logger.error(f"运行unittest测试失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "framework": "unittest"
            }
    
    def _parse_unittest_xml_report(self, root: ET.Element) -> Dict[str, Any]:
        """解析unittest XML报告
        
        Args:
            root: XML根元素
            
        Returns:
            报告摘要
        """
        summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0.0
        }
        
        # 从JUnit XML报告中提取信息
        summary["total_tests"] = int(root.attrib.get("tests", 0))
        summary["failed"] = int(root.attrib.get("failures", 0))
        summary["skipped"] = int(root.attrib.get("skipped", 0))
        summary["passed"] = summary["total_tests"] - summary["failed"] - summary["skipped"]
        summary["duration"] = float(root.attrib.get("time", 0.0))
        
        return summary
    
    def run_integration_tests(self, test_path: str, framework: str = "pytest", 
                            report_format: str = "json") -> Dict[str, Any]:
        """运行集成测试
        
        Args:
            test_path: 测试文件或目录路径
            framework: 测试框架（unittest或pytest）
            report_format: 报告格式（json或xml）
            
        Returns:
            测试结果
        """
        logger.info(f"开始运行集成测试，路径: {test_path}，框架: {framework}，报告格式: {report_format}")
        
        # 集成测试与单元测试的主要区别在于测试范围和测试环境
        # 这里我们可以复用单元测试的逻辑，但添加特定的集成测试标记或配置
        result = self.run_unit_tests(test_path, framework, "test_*.py", report_format)
        result["test_type"] = "integration"
        
        return result
    
    def run_all_tests(self, test_dir: str, framework: str = "pytest", 
                    report_format: str = "json") -> Dict[str, Any]:
        """运行所有测试（单元测试和集成测试）
        
        Args:
            test_dir: 测试目录
            framework: 测试框架（unittest或pytest）
            report_format: 报告格式（json或xml）
            
        Returns:
            测试结果
        """
        logger.info(f"开始运行所有测试，目录: {test_dir}，框架: {framework}，报告格式: {report_format}")
        
        # 运行单元测试
        unit_test_result = self.run_unit_tests(
            os.path.join(test_dir, "unit"), 
            framework, 
            "test_*.py", 
            report_format
        )
        
        # 运行集成测试
        integration_test_result = self.run_integration_tests(
            os.path.join(test_dir, "integration"), 
            framework, 
            report_format
        )
        
        # 合并结果
        combined_summary = {
            "total_tests": unit_test_result["summary"]["total_tests"] + integration_test_result["summary"]["total_tests"],
            "passed": unit_test_result["summary"]["passed"] + integration_test_result["summary"]["passed"],
            "failed": unit_test_result["summary"]["failed"] + integration_test_result["summary"]["failed"],
            "skipped": unit_test_result["summary"]["skipped"] + integration_test_result["summary"]["skipped"],
            "duration": unit_test_result["summary"]["duration"] + integration_test_result["summary"]["duration"]
        }
        
        return {
            "status": "success" if unit_test_result["status"] == "success" and integration_test_result["status"] == "success" else "failed",
            "framework": framework,
            "report_format": report_format,
            "unit_tests": unit_test_result,
            "integration_tests": integration_test_result,
            "summary": combined_summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_test_report(self, test_results: Dict[str, Any], format: str = "json") -> Dict[str, Any]:
        """生成测试报告
        
        Args:
            test_results: 测试结果
            format: 报告格式（json或html）
            
        Returns:
            测试报告
        """
        logger.info(f"生成测试报告，格式: {format}")
        
        # 生成测试报告
        report = {
            "report_type": "test_report",
            "timestamp": datetime.now().isoformat(),
            "framework": test_results.get("framework", "unknown"),
            "summary": test_results.get("summary", {}),
            "results": test_results
        }
        
        # 根据格式生成不同的报告
        if format == "json":
            return {
                "status": "success",
                "report": report,
                "format": format,
                "content": json.dumps(report, indent=2, ensure_ascii=False)
            }
        elif format == "html":
            html_content = self._generate_html_report(report)
            return {
                "status": "success",
                "report": report,
                "format": format,
                "content": html_content
            }
        else:
            return {
                "status": "failed",
                "error": f"不支持的报告格式: {format}"
            }
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """生成HTML格式的测试报告
        
        Args:
            report: 测试报告数据
            
        Returns:
            HTML格式的测试报告
        """
        # 生成简单的HTML报告
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>测试报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .summary-item {{ margin: 5px 0; }}
                .status {{ font-weight: bold; }}
                .success {{ color: green; }}
                .failed {{ color: red; }}
                .test-result {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>自动测试报告</h1>
            <div class="summary">
                <h2>测试摘要</h2>
                <div class="summary-item">框架: {report['framework']}</div>
                <div class="summary-item">生成时间: {report['timestamp']}</div>
                <div class="summary-item">总测试数: {report['summary']['total_tests']}</div>
                <div class="summary-item">通过: <span class="status success">{report['summary']['passed']}</span></div>
                <div class="summary-item">失败: <span class="status failed">{report['summary']['failed']}</span></div>
                <div class="summary-item">跳过: {report['summary']['skipped']}</div>
                <div class="summary-item">总耗时: {report['summary']['duration']:.2f}秒</div>
            </div>
        </body>
        </html>
        """
        return html_template
    
    def analyze_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析测试结果
        
        Args:
            test_results: 测试结果
            
        Returns:
            分析结果
        """
        logger.info("开始分析测试结果")
        
        summary = test_results.get("summary", {})
        
        # 计算通过率
        total_tests = summary.get("total_tests", 0)
        passed = summary.get("passed", 0)
        pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        # 识别失败原因
        failure_causes = {
            "total_failures": summary.get("failed", 0),
            "common_errors": []
        }
        
        # 简单的失败原因分析（根据stderr）
        if "stderr" in test_results:
            stderr = test_results["stderr"]
            if "AssertionError" in stderr:
                failure_causes["common_errors"].append("断言失败")
            if "ImportError" in stderr:
                failure_causes["common_errors"].append("导入错误")
            if "AttributeError" in stderr:
                failure_causes["common_errors"].append("属性错误")
            if "TypeError" in stderr:
                failure_causes["common_errors"].append("类型错误")
            if "KeyError" in stderr:
                failure_causes["common_errors"].append("键错误")
        
        # 生成分析报告
        analysis = {
            "pass_rate": round(pass_rate, 2),
            "total_tests": total_tests,
            "failure_causes": failure_causes,
            "recommendations": self._generate_test_recommendations(pass_rate, failure_causes)
        }
        
        logger.info(f"测试结果分析完成，通过率: {analysis['pass_rate']}%")
        return {
            "status": "success",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_test_recommendations(self, pass_rate: float, failure_causes: Dict[str, Any]) -> List[str]:
        """生成测试建议
        
        Args:
            pass_rate: 通过率
            failure_causes: 失败原因
            
        Returns:
            建议列表
        """
        recommendations = []
        
        if pass_rate < 70:
            recommendations.append("测试通过率较低，建议优先修复失败的测试用例")
        elif pass_rate < 90:
            recommendations.append("测试通过率中等，建议继续优化测试用例")
        else:
            recommendations.append("测试通过率较高，建议保持当前测试质量")
        
        # 根据失败原因生成建议
        common_errors = failure_causes.get("common_errors", [])
        if "断言失败" in common_errors:
            recommendations.append("检查断言条件是否正确，确保测试用例与预期一致")
        if "导入错误" in common_errors:
            recommendations.append("检查导入路径和依赖是否正确")
        if "属性错误" in common_errors:
            recommendations.append("检查对象属性是否存在，可能是API变更导致")
        if "类型错误" in common_errors:
            recommendations.append("检查参数类型是否正确，确保类型匹配")
        if "键错误" in common_errors:
            recommendations.append("检查字典键是否存在，可能是数据结构变更导致")
        
        return recommendations
    
    def discover_test_files(self, search_dir: str, test_pattern: str = "test_*.py") -> List[str]:
        """发现测试文件
        
        Args:
            search_dir: 搜索目录
            test_pattern: 测试文件匹配模式
            
        Returns:
            测试文件列表
        """
        test_files = []
        
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.startswith("."):
                    continue
                if file.endswith(".py") and (test_pattern in file or file.startswith("test_")):
                    test_files.append(os.path.join(root, file))
        
        logger.info(f"发现 {len(test_files)} 个测试文件")
        return test_files


# 单例模式
auto_test_agent = AutoTestAgent()
