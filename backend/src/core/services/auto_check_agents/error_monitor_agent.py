"""运行时错误监控智能体
监控和分析运行时错误日志，识别潜在问题
"""

from typing import Dict, List, Any, Optional
import logging
import os
import re
import json
from datetime import datetime, timedelta
import threading
import time

# 配置日志
logger = logging.getLogger(__name__)


class ErrorMonitorAgent:
    """运行时错误监控智能体
    
    功能：
    1. 监控日志文件中的错误信息
    2. 分析错误模式和趋势
    3. 生成错误报告
    4. 提供错误修复建议
    """
    
    def __init__(self):
        """初始化运行时错误监控智能体"""
        self.name = "error_monitor_agent"
        self.type = "analysis"
        self.capabilities = ["error_monitoring", "log_analysis", "error_detection"]
        self.monitored_logs = []  # 监控的日志文件列表
        self.error_patterns = self._load_error_patterns()  # 错误模式列表
        self.error_history = []  # 错误历史记录
        self.is_monitoring = False  # 监控状态
        self.monitor_thread = None  # 监控线程
        logger.info(f"{self.name} 初始化完成")
    
    def _load_error_patterns(self) -> List[Dict[str, Any]]:
        """加载错误模式
        
        Returns:
            错误模式列表
        """
        # 定义常见的错误模式
        patterns = [
            {
                "name": "python_exception",
                "pattern": r"Traceback \(most recent call last\):[\s\S]*?\w+Error:[\s\S]*?$",
                "severity": "high",
                "language": "python"
            },
            {
            "name": "javascript_error",
            "pattern": r'Error: [^"]*',
            "severity": "high",
            "language": "javascript"
        },
            {
                "name": "file_not_found",
                "pattern": r"FileNotFoundError:|[Ff]ile [Nn]ot [Ff]ound|[Cc]annot [Oo]pen [Ff]ile",
                "severity": "medium"
            },
            {
                "name": "permission_denied",
                "pattern": r"PermissionError:|[Pp]ermission [Dd]enied|[Aa]ccess [Dd]enied",
                "severity": "medium"
            },
            {
                "name": "connection_error",
                "pattern": r"ConnectionError:|[Cc]onnection [Rr]efused|[Tt]imeout",
                "severity": "medium"
            },
            {
                "name": "database_error",
                "pattern": r"DatabaseError:|[Ss]QL [Ee]rror|[Dd]atabase [Cc]onnection",
                "severity": "high"
            },
            {
                "name": "out_of_memory",
                "pattern": r"MemoryError:|[Oo]ut [Oo]f [Mm]emory",
                "severity": "critical"
            }
        ]
        return patterns
    
    def monitor_log_file(self, log_path: str, log_type: str = "general") -> bool:
        """监控日志文件
        
        Args:
            log_path: 日志文件路径
            log_type: 日志类型
            
        Returns:
            监控是否成功
        """
        if not os.path.exists(log_path):
            logger.error(f"日志文件不存在: {log_path}")
            return False
        
        # 检查是否已经在监控
        for monitored_log in self.monitored_logs:
            if monitored_log["path"] == log_path:
                logger.warning(f"日志文件已在监控中: {log_path}")
                return True
        
        # 添加到监控列表
        self.monitored_logs.append({
            "path": log_path,
            "type": log_type,
            "last_position": os.path.getsize(log_path),  # 从文件末尾开始监控
            "last_check": datetime.now().isoformat()
        })
        
        logger.info(f"开始监控日志文件: {log_path}，类型: {log_type}")
        return True
    
    def stop_monitoring(self, log_path: str) -> bool:
        """停止监控指定的日志文件
        
        Args:
            log_path: 日志文件路径
            
        Returns:
            停止是否成功
        """
        for i, monitored_log in enumerate(self.monitored_logs):
            if monitored_log["path"] == log_path:
                del self.monitored_logs[i]
                logger.info(f"停止监控日志文件: {log_path}")
                return True
        
        logger.warning(f"日志文件不在监控列表中: {log_path}")
        return False
    
    def start_monitoring(self, interval: int = 5) -> bool:
        """开始监控所有日志文件
        
        Args:
            interval: 检查间隔（秒）
            
        Returns:
            启动是否成功
        """
        if self.is_monitoring:
            logger.warning("监控已经在运行中")
            return False
        
        self.is_monitoring = True
        
        # 创建监控线程
        def monitor_loop():
            while self.is_monitoring:
                self._check_all_logs()
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"启动监控，检查间隔: {interval}秒")
        return True
    
    def stop_all_monitoring(self) -> bool:
        """停止所有监控
        
        Returns:
            停止是否成功
        """
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("停止所有监控")
        return True
    
    def _check_all_logs(self) -> None:
        """检查所有监控的日志文件
        """
        for monitored_log in self.monitored_logs:
            self._check_log_file(monitored_log)
    
    def _check_log_file(self, monitored_log: Dict[str, Any]) -> None:
        """检查单个日志文件
        
        Args:
            monitored_log: 监控的日志文件信息
        """
        log_path = monitored_log["path"]
        try:
            # 检查文件是否存在
            if not os.path.exists(log_path):
                logger.error(f"日志文件不存在: {log_path}")
                return
            
            current_size = os.path.getsize(log_path)
            last_position = monitored_log["last_position"]
            
            # 如果文件被截断（可能是日志轮转），从开头开始读取
            if current_size < last_position:
                last_position = 0
            
            # 读取新内容
            if current_size > last_position:
                with open(log_path, "r", encoding="utf-8") as f:
                    f.seek(last_position)
                    new_content = f.read()
                
                # 分析新内容
                errors = self.analyze_log_content(new_content, log_path, monitored_log["type"])
                
                # 更新最后检查位置
                monitored_log["last_position"] = current_size
                monitored_log["last_check"] = datetime.now().isoformat()
                
                # 保存错误信息
                if errors:
                    self.error_history.extend(errors)
        except Exception as e:
            logger.error(f"检查日志文件失败: {log_path}，错误: {str(e)}")
    
    def analyze_log_content(self, content: str, log_path: str, log_type: str = "general") -> List[Dict[str, Any]]:
        """分析日志内容
        
        Args:
            content: 日志内容
            log_path: 日志文件路径
            log_type: 日志类型
            
        Returns:
            识别到的错误列表
        """
        errors = []
        
        # 按行分析日志
        lines = content.splitlines()
        for i, line in enumerate(lines):
            # 检查每行是否包含错误模式
            for pattern in self.error_patterns:
                match = re.search(pattern["pattern"], line)
                if match:
                    error = {
                        "timestamp": datetime.now().isoformat(),
                        "log_path": log_path,
                        "log_type": log_type,
                        "error_type": pattern["name"],
                        "severity": pattern["severity"],
                        "line": line.strip(),
                        "line_number": i + 1,
                        "match_text": match.group(0),
                        "context": self._get_context(lines, i, 3)  # 获取上下文
                    }
                    errors.append(error)
                    logger.warning(f"发现错误: {pattern['name']} - {line.strip()}")
        
        return errors
    
    def _get_context(self, lines: List[str], line_index: int, context_size: int = 3) -> List[str]:
        """获取上下文
        
        Args:
            lines: 所有行
            line_index: 当前行索引
            context_size: 上下文大小
            
        Returns:
            上下文列表
        """
        start = max(0, line_index - context_size)
        end = min(len(lines), line_index + context_size + 1)
        return lines[start:end]
    
    def analyze_log_file(self, log_path: str, log_type: str = "general", time_range_hours: int = 24) -> Dict[str, Any]:
        """分析日志文件
        
        Args:
            log_path: 日志文件路径
            log_type: 日志类型
            time_range_hours: 分析的时间范围（小时）
            
        Returns:
            分析结果
        """
        try:
            if not os.path.exists(log_path):
                return {
                    "status": "failed",
                    "error": f"日志文件不存在: {log_path}"
                }
            
            logger.info(f"开始分析日志文件: {log_path}，时间范围: {time_range_hours}小时")
            
            # 读取日志文件
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 分析日志内容
            errors = self.analyze_log_content(content, log_path, log_type)
            
            # 过滤时间范围内的错误
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            recent_errors = [
                error for error in errors
                if datetime.fromisoformat(error["timestamp"]) >= cutoff_time
            ]
            
            # 生成分析报告
            report = {
                "status": "success",
                "log_path": log_path,
                "log_type": log_type,
                "time_range_hours": time_range_hours,
                "total_errors": len(recent_errors),
                "errors": recent_errors,
                "summary": self._generate_error_summary(recent_errors)
            }
            
            logger.info(f"日志文件分析完成: {log_path}，发现 {len(recent_errors)} 个错误")
            return report
        except Exception as e:
            logger.error(f"分析日志文件失败: {log_path}，错误: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _generate_error_summary(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成错误摘要
        
        Args:
            errors: 错误列表
            
        Returns:
            错误摘要
        """
        summary = {
            "total_errors": len(errors),
            "error_types": {},
            "severity_counts": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "top_errors": []
        }
        
        if not errors:
            return summary
        
        # 统计错误类型和严重程度
        for error in errors:
            error_type = error["error_type"]
            severity = error["severity"]
            
            # 统计错误类型
            if error_type not in summary["error_types"]:
                summary["error_types"][error_type] = 0
            summary["error_types"][error_type] += 1
            
            # 统计严重程度
            if severity in summary["severity_counts"]:
                summary["severity_counts"][severity] += 1
        
        # 找出最常见的错误类型
        summary["top_errors"] = sorted(
            summary["error_types"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # 只显示前5个
        
        return summary
    
    def get_error_history(self, limit: int = 100, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取错误历史
        
        Args:
            limit: 返回的错误数量限制
            severity: 按严重程度过滤
            
        Returns:
            错误历史列表
        """
        if severity:
            filtered_errors = [
                error for error in self.error_history
                if error["severity"] == severity
            ]
        else:
            filtered_errors = self.error_history
        
        # 按时间倒序排序，返回最近的错误
        return sorted(
            filtered_errors,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def generate_error_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """生成错误报告
        
        Args:
            time_range_hours: 报告的时间范围（小时）
            
        Returns:
            错误报告
        """
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error["timestamp"]) >= cutoff_time
        ]
        
        report = {
            "report_time": datetime.now().isoformat(),
            "time_range_hours": time_range_hours,
            "total_errors": len(recent_errors),
            "summary": self._generate_error_summary(recent_errors),
            "errors": recent_errors,
            "monitored_logs": [
                {
                    "path": log["path"],
                    "type": log["type"],
                    "last_check": log["last_check"]
                }
                for log in self.monitored_logs
            ]
        }
        
        logger.info(f"生成错误报告，时间范围: {time_range_hours}小时，错误数量: {len(recent_errors)}")
        return report
    
    def clear_error_history(self) -> bool:
        """清除错误历史
        
        Returns:
            清除是否成功
        """
        self.error_history = []
        logger.info("错误历史已清除")
        return True
    
    def detect_error_trends(self, time_range_hours: int = 24, interval_hours: int = 1) -> Dict[str, Any]:
        """检测错误趋势
        
        Args:
            time_range_hours: 时间范围（小时）
            interval_hours: 统计间隔（小时）
            
        Returns:
            错误趋势数据
        """
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error["timestamp"]) >= cutoff_time
        ]
        
        # 按时间间隔分组
        intervals = []
        current_time = cutoff_time
        while current_time < datetime.now():
            next_time = current_time + timedelta(hours=interval_hours)
            interval_errors = [
                error for error in recent_errors
                if current_time <= datetime.fromisoformat(error["timestamp"]) < next_time
            ]
            
            interval_data = {
                "start_time": current_time.isoformat(),
                "end_time": next_time.isoformat(),
                "error_count": len(interval_errors),
                "error_types": {}
            }
            
            # 统计错误类型
            for error in interval_errors:
                error_type = error["error_type"]
                if error_type not in interval_data["error_types"]:
                    interval_data["error_types"][error_type] = 0
                interval_data["error_types"][error_type] += 1
            
            intervals.append(interval_data)
            current_time = next_time
        
        return {
            "time_range_hours": time_range_hours,
            "interval_hours": interval_hours,
            "trends": intervals
        }


# 单例模式
error_monitor_agent = ErrorMonitorAgent()
