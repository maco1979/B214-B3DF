"""代码静态分析智能体
集成PyLint、Flake8等工具，用于检查代码中的潜在问题和风格问题
"""

from typing import Dict, List, Any
import logging
import subprocess
import os
import tempfile

# 配置日志
logger = logging.getLogger(__name__)


class StaticAnalysisAgent:
    """代码静态分析智能体
    
    功能：
    1. 使用PyLint检查代码质量
    2. 使用Flake8检查代码风格
    3. 生成综合分析报告
    """
    
    def __init__(self):
        """初始化代码静态分析智能体"""
        self.name = "static_analysis_agent"
        self.type = "code"
        self.capabilities = ["pylint", "flake8", "code_analysis"]
        logger.info(f"{self.name} 初始化完成")
    
    def analyze_code(self, code: str, file_name: str = "temp.py", file_type: str = "python") -> Dict[str, Any]:
        """分析代码
        
        Args:
            code: 要分析的代码内容
            file_name: 文件名（用于识别文件类型）
            file_type: 文件类型（python, javascript, etc.）
            
        Returns:
            分析结果
        """
        logger.info(f"开始分析代码: {file_name} ({file_type})")
        
        # 根据文件类型选择分析工具
        if file_type == "python":
            return self._analyze_python_code(code, file_name)
        elif file_type == "javascript":
            return self._analyze_javascript_code(code, file_name)
        else:
            return {
                "error": f"不支持的文件类型: {file_type}",
                "status": "failed"
            }
    
    def _analyze_python_code(self, code: str, file_name: str) -> Dict[str, Any]:
        """分析Python代码
        
        Args:
            code: Python代码内容
            file_name: 文件名
            
        Returns:
            分析结果
        """
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file_path = f.name
            
            # 运行PyLint
            pylint_result = self._run_pylint(temp_file_path)
            
            # 运行Flake8
            flake8_result = self._run_flake8(temp_file_path)
            
            # 清理临时文件
            os.unlink(temp_file_path)
            
            # 生成综合报告
            report = {
                "status": "success",
                "file_name": file_name,
                "file_type": "python",
                "analysis_tools": ["pylint", "flake8"],
                "pylint": pylint_result,
                "flake8": flake8_result,
                "summary": {
                    "pylint_score": pylint_result.get("score", 0),
                    "pylint_errors": len(pylint_result.get("errors", [])),
                    "pylint_warnings": len(pylint_result.get("warnings", [])),
                    "flake8_errors": len(flake8_result.get("errors", [])),
                    "total_issues": len(pylint_result.get("errors", [])) + len(pylint_result.get("warnings", [])) + len(flake8_result.get("errors", []))
                }
            }
            
            logger.info(f"Python代码分析完成: {file_name}")
            return report
            
        except Exception as e:
            logger.error(f"Python代码分析失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _run_pylint(self, file_path: str) -> Dict[str, Any]:
        """运行PyLint分析
        
        Args:
            file_path: 文件路径
            
        Returns:
            PyLint分析结果
        """
        try:
            # 运行PyLint命令
            result = subprocess.run(
                ["pylint", "--output-format=json", file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 or result.returncode == 1:
                # PyLint返回0表示没有错误，1表示有错误或警告
                import json
                pylint_output = json.loads(result.stdout)
                
                # 解析结果
                errors = []
                warnings = []
                
                for msg in pylint_output:
                    msg_dict = {
                        "line": msg.get("line", 0),
                        "column": msg.get("column", 0),
                        "message": msg.get("message", ""),
                        "symbol": msg.get("symbol", ""),
                        "type": msg.get("type", "error")
                    }
                    
                    if msg.get("type") in ["error", "fatal", "critical"]:
                        errors.append(msg_dict)
                    else:
                        warnings.append(msg_dict)
                
                # 获取总分
                score = 10.0  # 默认满分
                if pylint_output:
                    score = pylint_output[0].get("score", 10.0)
                
                return {
                    "status": "success",
                    "score": score,
                    "errors": errors,
                    "warnings": warnings,
                    "raw_output": result.stdout
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr
                }
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": "PyLint分析超时"
            }
        except Exception as e:
            logger.error(f"PyLint运行失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _run_flake8(self, file_path: str) -> Dict[str, Any]:
        """运行Flake8分析
        
        Args:
            file_path: 文件路径
            
        Returns:
            Flake8分析结果
        """
        try:
            # 运行Flake8命令
            result = subprocess.run(
                ["flake8", file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            errors = []
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        # 解析Flake8输出格式: file:line:column: code message
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            error = {
                                "line": int(parts[1]),
                                "column": int(parts[2]),
                                "code": parts[3].split()[0],
                                "message": parts[3][len(parts[3].split()[0])+1:].strip()
                            }
                            errors.append(error)
            
            return {
                "status": "success",
                "errors": errors,
                "raw_output": result.stdout.strip()
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": "Flake8分析超时"
            }
        except Exception as e:
            logger.error(f"Flake8运行失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _analyze_javascript_code(self, code: str, file_name: str) -> Dict[str, Any]:
        """分析JavaScript代码
        
        Args:
            code: JavaScript代码内容
            file_name: 文件名
            
        Returns:
            分析结果
        """
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file_path = f.name
            
            # 运行ESLint（如果可用）
            eslint_result = self._run_eslint(temp_file_path)
            
            # 清理临时文件
            os.unlink(temp_file_path)
            
            report = {
                "status": "success",
                "file_name": file_name,
                "file_type": "javascript",
                "analysis_tools": ["eslint"],
                "eslint": eslint_result,
                "summary": {
                    "eslint_errors": len(eslint_result.get("errors", []))
                }
            }
            
            logger.info(f"JavaScript代码分析完成: {file_name}")
            return report
            
        except Exception as e:
            logger.error(f"JavaScript代码分析失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _run_eslint(self, file_path: str) -> Dict[str, Any]:
        """运行ESLint分析
        
        Args:
            file_path: 文件路径
            
        Returns:
            ESLint分析结果
        """
        try:
            # 检查ESLint是否可用
            subprocess.run(
                ["eslint", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # 运行ESLint命令
            result = subprocess.run(
                ["eslint", "--format", "json", file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 or result.returncode == 1 or result.returncode == 2:
                # ESLint返回0表示没有错误，1表示有警告，2表示有错误
                import json
                eslint_output = json.loads(result.stdout)
                
                errors = []
                if eslint_output and isinstance(eslint_output, list) and len(eslint_output) > 0:
                    for msg in eslint_output[0].get("messages", []):
                        error = {
                            "line": msg.get("line", 0),
                            "column": msg.get("column", 0),
                            "message": msg.get("message", ""),
                            "ruleId": msg.get("ruleId", ""),
                            "severity": msg.get("severity", 1)
                        }
                        errors.append(error)
                
                return {
                    "status": "success",
                    "errors": errors,
                    "raw_output": result.stdout
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr
                }
        except FileNotFoundError:
            return {
                "status": "skipped",
                "error": "ESLint未安装"
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": "ESLint分析超时"
            }
        except Exception as e:
            logger.error(f"ESLint运行失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def analyze_directory(self, directory_path: str, file_types: List[str] = ["python"]) -> Dict[str, Any]:
        """分析目录中的所有代码文件
        
        Args:
            directory_path: 目录路径
            file_types: 要分析的文件类型列表
            
        Returns:
            分析结果
        """
        try:
            logger.info(f"开始分析目录: {directory_path}，文件类型: {file_types}")
            
            # 收集要分析的文件
            files_to_analyze = []
            for root, _, files in os.walk(directory_path):
                for file in files:
                    # 跳过临时文件和隐藏文件
                    if file.startswith('.') or file.endswith('~'):
                        continue
                    
                    # 根据文件扩展名确定文件类型
                    file_type = self._get_file_type(file)
                    if file_type in file_types:
                        file_path = os.path.join(root, file)
                        files_to_analyze.append((file_path, file, file_type))
            
            # 分析每个文件
            results = {}
            for file_path, file_name, file_type in files_to_analyze:
                logger.info(f"分析文件: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                result = self.analyze_code(code, file_name, file_type)
                results[file_path] = result
            
            # 生成目录分析报告
            report = {
                "status": "success",
                "directory": directory_path,
                "file_types": file_types,
                "files_analyzed": len(files_to_analyze),
                "results": results,
                "summary": self._generate_directory_summary(results)
            }
            
            logger.info(f"目录分析完成: {directory_path}")
            return report
            
        except Exception as e:
            logger.error(f"目录分析失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_file_type(self, file_name: str) -> str:
        """根据文件名获取文件类型
        
        Args:
            file_name: 文件名
            
        Returns:
            文件类型
        """
        ext = os.path.splitext(file_name)[1].lower()
        if ext in ['.py', '.pyw']:
            return 'python'
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            return 'javascript'
        elif ext in ['.html', '.css', '.scss', '.less']:
            return 'web'
        else:
            return 'unknown'
    
    def _generate_directory_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成目录分析摘要
        
        Args:
            results: 所有文件的分析结果
            
        Returns:
            目录分析摘要
        """
        summary = {
            "total_files": len(results),
            "success_files": 0,
            "failed_files": 0,
            "warning_files": 0,
            "total_issues": 0,
            "pylint_issues": 0,
            "flake8_issues": 0,
            "eslint_issues": 0
        }
        
        for file_path, result in results.items():
            if result["status"] == "success":
                summary["success_files"] += 1
                
                # 统计问题数量
                if "summary" in result:
                    total_issues = result["summary"].get("total_issues", 0)
                    summary["total_issues"] += total_issues
                    
                    # 统计各工具的问题数量
                    if "pylint_score" in result["summary"]:
                        summary["pylint_issues"] += result["summary"].get("pylint_errors", 0) + result["summary"].get("pylint_warnings", 0)
                    if "flake8_errors" in result["summary"]:
                        summary["flake8_issues"] += result["summary"]["flake8_errors"]
                    if "eslint_errors" in result["summary"]:
                        summary["eslint_issues"] += result["summary"]["eslint_errors"]
            else:
                summary["failed_files"] += 1
        
        # 计算警告文件数量（有问题但分析成功的文件）
        summary["warning_files"] = summary["success_files"] - (summary["total_files"] - summary["failed_files"] - summary["warning_files"])
        
        return summary


# 单例模式
static_analysis_agent = StaticAnalysisAgent()
