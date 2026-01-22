"""
AI决策系统性能优化器 - 确保系统实现秒级实时响应能力
"""

import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import psutil
import gc
from ...config.performance_config import PerformanceConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time: float  # 响应时间（秒）
    throughput: float     # 吞吐量（决策/秒）
    cpu_usage: float      # CPU使用率
    memory_usage: float   # 内存使用率
    error_rate: float     # 错误率
    timestamp: datetime


class PerformanceOptimizer:
    """AI决策系统性能优化器"""
    
    def __init__(self, environment: str = "production"):
        # 加载性能配置
        self.config = PerformanceConfig.get_optimized_config(environment)
        
        # 性能目标（从配置中获取）
        perf_thresholds = self.config["monitoring"]["performance_thresholds"]
        self.performance_targets = {
            "max_response_time": perf_thresholds["max_response_time"] / 1000.0,  # 转换为秒
            "min_throughput": perf_thresholds["throughput_threshold"],
            "max_cpu_usage": 0.8,
            "max_memory_usage": 0.7,
            "max_error_rate": perf_thresholds["error_rate_threshold"]
        }
        
        # 性能监控数据
        self.performance_history: List[PerformanceMetrics] = []
        
        # 优化配置（从配置中获取）
        decision_config = self.config["decision_engine"]
        real_time_config = self.config["real_time_optimization"]
        
        self.optimization_config = {
            "batch_processing_threshold": decision_config["batch_size"],
            "cache_ttl": decision_config["cache_ttl"],
            "precompute_interval": 60,
            "parallel_processing_limit": real_time_config["thread_pool_size"],
            "memory_cleanup_threshold": 0.8
        }
        
        # 缓存机制
        self.decision_cache: Dict[str, Dict[str, Any]] = {}
        self.feature_cache: Dict[str, Any] = {}
        
        # 预计算任务
        self.precomputation_tasks: Dict[str, Any] = {}
        
        # 记录配置环境
        self.environment = environment
        logger.info(f"性能优化器初始化完成 - 环境: {environment}")
    
    async def optimize_decision_making(self, 
                                      module_name: str,
                                      state_data: Dict[str, Any],
                                      objective: str) -> Dict[str, Any]:
        """
        优化决策过程
        
        Args:
            module_name: 决策模块名称
            state_data: 状态数据
            objective: 决策目标
            
        Returns:
            优化后的决策结果
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(module_name, state_data, objective)
            cached_result = self._get_cached_decision(cache_key)
            
            if cached_result:
                logger.info(f"使用缓存决策: {cache_key}")
                return cached_result
            
            # 并行处理检查
            if self._should_use_parallel_processing(state_data):
                result = await self._parallel_decision_making(module_name, state_data, objective)
            else:
                result = await self._sequential_decision_making(module_name, state_data, objective)
            
            # 缓存结果
            self._cache_decision(cache_key, result)
            
            # 记录性能指标
            await self._record_performance_metrics(start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"决策优化错误: {str(e)}")
            
            # 记录错误性能指标
            await self._record_error_metrics(start_time)
            
            # 返回降级决策
            return self._get_fallback_decision(module_name, state_data, objective)
    
    def _generate_cache_key(self, 
                           module_name: str, 
                           state_data: Dict[str, Any], 
                           objective: str) -> str:
        """生成缓存键"""
        # 简化状态数据生成缓存键
        simplified_state = {}
        for key, value in state_data.items():
            if isinstance(value, (int, float, str)):
                simplified_state[key] = value
            elif isinstance(value, dict):
                # 只取前3个键值对
                simplified_state[key] = {k: v for i, (k, v) in enumerate(value.items()) if i < 3}
        
        import hashlib
        state_str = str(sorted(simplified_state.items()))
        key_str = f"{module_name}:{objective}:{state_str}"
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_decision(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存的决策"""
        if cache_key in self.decision_cache:
            cached_data = self.decision_cache[cache_key]
            
            # 检查缓存是否过期
            if time.time() - cached_data["timestamp"] < self.optimization_config["cache_ttl"]:
                return cached_data["result"]
            else:
                # 删除过期缓存
                del self.decision_cache[cache_key]
        
        return None
    
    def _cache_decision(self, cache_key: str, result: Dict[str, Any]):
        """缓存决策结果"""
        # 限制缓存大小
        if len(self.decision_cache) > 1000:
            # 删除最旧的缓存
            oldest_key = min(self.decision_cache.keys(), 
                           key=lambda k: self.decision_cache[k]["timestamp"])
            del self.decision_cache[oldest_key]
        
        self.decision_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def _should_use_parallel_processing(self, state_data: Dict[str, Any]) -> bool:
        """判断是否使用并行处理"""
        # 基于状态复杂度和系统负载判断
        state_complexity = len(str(state_data)) / 1000  # 简化复杂度计算
        current_load = psutil.cpu_percent() / 100.0
        
        return (state_complexity > 0.5 and 
                current_load < self.performance_targets["max_cpu_usage"] * 0.8)
    
    async def _parallel_decision_making(self, 
                                       module_name: str,
                                       state_data: Dict[str, Any],
                                       objective: str) -> Dict[str, Any]:
        """并行决策处理"""
        # 创建并行任务
        tasks = []
        
        # 特征提取任务
        feature_task = asyncio.create_task(
            self._extract_features_parallel(module_name, state_data)
        )
        tasks.append(feature_task)
        
        # 风险评估任务
        risk_task = asyncio.create_task(
            self._assess_risk_parallel(module_name, state_data)
        )
        tasks.append(risk_task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        features = results[0] if not isinstance(results[0], Exception) else {}
        risk_assessment = results[1] if not isinstance(results[1], Exception) else {}
        
        # 执行决策
        decision_result = await self._execute_decision_engine(
            module_name, features, risk_assessment, objective
        )
        
        return decision_result
    
    async def _sequential_decision_making(self, 
                                        module_name: str,
                                        state_data: Dict[str, Any],
                                        objective: str) -> Dict[str, Any]:
        """顺序决策处理"""
        # 顺序执行各步骤
        features = await self._extract_features_sequential(module_name, state_data)
        risk_assessment = await self._assess_risk_sequential(module_name, state_data)
        
        # 执行决策
        decision_result = await self._execute_decision_engine(
            module_name, features, risk_assessment, objective
        )
        
        return decision_result
    
    async def _extract_features_parallel(self, 
                                       module_name: str,
                                       state_data: Dict[str, Any]) -> Dict[str, Any]:
        """并行特征提取"""
        # 模拟并行特征提取
        await asyncio.sleep(0.01)  # 模拟计算时间
        
        # 简化特征提取
        features = {}
        for key, value in state_data.items():
            if isinstance(value, (int, float)):
                features[f"{key}_feature"] = float(value)
            elif isinstance(value, dict):
                features[f"{key}_count"] = len(value)
        
        return features
    
    async def _extract_features_sequential(self, 
                                         module_name: str,
                                         state_data: Dict[str, Any]) -> Dict[str, Any]:
        """顺序特征提取"""
        # 简化特征提取
        features = {}
        for key, value in state_data.items():
            if isinstance(value, (int, float)):
                features[f"{key}_feature"] = float(value)
            elif isinstance(value, dict):
                features[f"{key}_count"] = len(value)
        
        return features
    
    async def _assess_risk_parallel(self, 
                                  module_name: str,
                                  state_data: Dict[str, Any]) -> Dict[str, Any]:
        """并行风险评估"""
        # 模拟并行风险评估
        await asyncio.sleep(0.005)  # 模拟计算时间
        
        # 简化风险评估
        risk_scores = {}
        
        # 基于数值的风险评估
        numeric_values = [v for v in state_data.values() if isinstance(v, (int, float))]
        if numeric_values:
            avg_value = sum(numeric_values) / len(numeric_values)
            risk_scores["numeric_risk"] = min(1.0, avg_value / 100.0)
        
        # 基于复杂度的风险评估
        complexity = len(str(state_data)) / 1000
        risk_scores["complexity_risk"] = min(1.0, complexity)
        
        return risk_scores
    
    async def _assess_risk_sequential(self, 
                                    module_name: str,
                                    state_data: Dict[str, Any]) -> Dict[str, Any]:
        """顺序风险评估"""
        # 简化风险评估
        risk_scores = {}
        
        # 基于数值的风险评估
        numeric_values = [v for v in state_data.values() if isinstance(v, (int, float))]
        if numeric_values:
            avg_value = sum(numeric_values) / len(numeric_values)
            risk_scores["numeric_risk"] = min(1.0, avg_value / 100.0)
        
        # 基于复杂度的风险评估
        complexity = len(str(state_data)) / 1000
        risk_scores["complexity_risk"] = min(1.0, complexity)
        
        return risk_scores
    
    async def _execute_decision_engine(self, 
                                     module_name: str,
                                     features: Dict[str, Any],
                                     risk_assessment: Dict[str, Any],
                                     objective: str) -> Dict[str, Any]:
        """执行决策引擎"""
        # 模拟决策引擎执行
        await asyncio.sleep(0.02)  # 模拟决策计算时间
        
        # 基于特征和风险生成决策
        decision_result = {
            "action": f"optimize_{module_name}",
            "parameters": {
                "feature_based_param": sum(features.values()) / len(features) if features else 0.5,
                "risk_adjusted_param": 1.0 - risk_assessment.get("numeric_risk", 0.5),
                "objective_factor": 1.0 if objective == "maximize" else 0.8
            },
            "confidence": max(0.7, 1.0 - risk_assessment.get("complexity_risk", 0.3)),
            "execution_time": 0.05,
            "optimized": True
        }
        
        return decision_result
    
    def _get_fallback_decision(self, 
                             module_name: str,
                             state_data: Dict[str, Any],
                             objective: str) -> Dict[str, Any]:
        """获取降级决策"""
        # 简化降级决策逻辑
        return {
            "action": f"fallback_{module_name}",
            "parameters": {"fallback_mode": True, "safety_factor": 0.5},
            "confidence": 0.5,
            "execution_time": 0.01,
            "fallback": True,
            "message": "使用降级决策模式"
        }
    
    async def _record_performance_metrics(self, start_time: float):
        """记录性能指标"""
        response_time = time.time() - start_time
        
        metrics = PerformanceMetrics(
            response_time=response_time,
            throughput=1.0 / response_time if response_time > 0 else 0.0,
            cpu_usage=psutil.cpu_percent() / 100.0,
            memory_usage=psutil.virtual_memory().percent / 100.0,
            error_rate=0.0,
            timestamp=datetime.now()
        )
        
        self.performance_history.append(metrics)
        
        # 保持历史记录长度
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
    
    async def _record_error_metrics(self, start_time: float):
        """记录错误性能指标"""
        response_time = time.time() - start_time
        
        metrics = PerformanceMetrics(
            response_time=response_time,
            throughput=0.0,
            cpu_usage=psutil.cpu_percent() / 100.0,
            memory_usage=psutil.virtual_memory().percent / 100.0,
            error_rate=1.0,
            timestamp=datetime.now()
        )
        
        self.performance_history.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {"message": "暂无性能数据"}
        
        # 计算统计指标
        recent_metrics = self.performance_history[-100:]  # 最近100次
        
        response_times = [m.response_time for m in recent_metrics]
        throughputs = [m.throughput for m in recent_metrics if m.throughput > 0]
        cpu_usages = [m.cpu_usage for m in recent_metrics]
        memory_usages = [m.memory_usage for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]
        
        summary = {
            "total_decisions": len(self.performance_history),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
            "max_response_time": max(response_times) if response_times else 0.0,
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0.0,
            "avg_cpu_usage": sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0,
            "avg_memory_usage": sum(memory_usages) / len(memory_usages) if memory_usages else 0.0,
            "avg_error_rate": sum(error_rates) / len(error_rates) if error_rates else 0.0,
            "performance_targets": self.performance_targets,
            "meets_targets": self._check_performance_targets(recent_metrics)
        }
        
        return summary
    
    def _check_performance_targets(self, metrics: List[PerformanceMetrics]) -> Dict[str, bool]:
        """检查性能目标达成情况"""
        if not metrics:
            return {}
        
        response_times = [m.response_time for m in metrics]
        throughputs = [m.throughput for m in metrics if m.throughput > 0]
        cpu_usages = [m.cpu_usage for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]
        error_rates = [m.error_rate for m in metrics]
        
        meets_targets = {
            "response_time": (sum(response_times) / len(response_times)) <= self.performance_targets["max_response_time"],
            "throughput": (sum(throughputs) / len(throughputs)) >= self.performance_targets["min_throughput"] if throughputs else False,
            "cpu_usage": (sum(cpu_usages) / len(cpu_usages)) <= self.performance_targets["max_cpu_usage"],
            "memory_usage": (sum(memory_usages) / len(memory_usages)) <= self.performance_targets["max_memory_usage"],
            "error_rate": (sum(error_rates) / len(error_rates)) <= self.performance_targets["max_error_rate"]
        }
        
        return meets_targets
    
    async def optimize_memory_usage(self):
        """优化内存使用"""
        current_memory = psutil.virtual_memory().percent / 100.0
        
        if current_memory > self.optimization_config["memory_cleanup_threshold"]:
            logger.info("执行内存优化")
            
            # 清理缓存
            self._cleanup_caches()
            
            # 执行垃圾回收
            gc.collect()
            
            # 清理性能历史记录
            if len(self.performance_history) > 500:
                self.performance_history = self.performance_history[-250:]
    
    def _cleanup_caches(self):
        """清理缓存"""
        current_time = time.time()
        
        # 清理过期缓存
        expired_keys = [
            key for key, data in self.decision_cache.items()
            if current_time - data["timestamp"] > self.optimization_config["cache_ttl"]
        ]
        
        for key in expired_keys:
            del self.decision_cache[key]
        
        # 限制缓存大小
        if len(self.decision_cache) > 500:
            # 删除最旧的缓存
            keys_to_remove = sorted(
                self.decision_cache.keys(),
                key=lambda k: self.decision_cache[k]["timestamp"]
            )[:100]
            
            for key in keys_to_remove:
                del self.decision_cache[key]
    
    async def precompute_common_decisions(self):
        """预计算常见决策"""
        logger.info("开始预计算常见决策")
        
        # 常见状态模式
        common_states = [
            {"load": 0.3, "risk": 0.1, "objective": "efficiency"},
            {"load": 0.6, "risk": 0.3, "objective": "balance"},
            {"load": 0.8, "risk": 0.5, "objective": "performance"}
        ]
        
        for state in common_states:
            for module in ["agriculture", "blockchain", "model_training", "resource_allocation"]:
                cache_key = self._generate_cache_key(module, state, state["objective"])
                
                # 如果缓存中不存在，预计算并缓存
                if cache_key not in self.decision_cache:
                    try:
                        result = await self._execute_decision_engine(
                            module, 
                            {"load_feature": state["load"]}, 
                            {"risk_score": state["risk"]}, 
                            state["objective"]
                        )
                        
                        self._cache_decision(cache_key, result)
                        
                    except Exception as e:
                        logger.warning(f"预计算失败: {module} - {str(e)}")
    
    async def start_performance_monitoring(self):
        """启动性能监控"""
        logger.info("启动性能监控")
        
        while True:
            try:
                # 检查性能
                summary = self.get_performance_summary()
                
                # 检查是否满足性能目标
                if summary.get("meets_targets"):
                    meets_all = all(summary["meets_targets"].values())
                    if not meets_all:
                        logger.warning("性能目标未完全达成")
                        
                        # 执行优化
                        await self.optimize_memory_usage()
                
                # 定期预计算
                await self.precompute_common_decisions()
                
                # 等待下一次检查
                await asyncio.sleep(self.optimization_config["precompute_interval"])
                
            except Exception as e:
                logger.error(f"性能监控错误: {str(e)}")
                await asyncio.sleep(10)  # 错误后等待10秒


# 全局性能优化器实例
performance_optimizer = PerformanceOptimizer()