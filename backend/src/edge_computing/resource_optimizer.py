"""
边缘计算资源优化器

负责边缘计算环境的资源优化调度，包括：
- 资源分配策略
- 负载均衡优化
- 能效优化
- 成本优化
"""

import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import heapq


class OptimizationObjective(Enum):
    """优化目标"""
    PERFORMANCE = "performance"  # 性能优化
    COST = "cost"  # 成本优化
    ENERGY_EFFICIENCY = "energy_efficiency"  # 能效优化
    RELIABILITY = "reliability"  # 可靠性优化
    BALANCED = "balanced"  # 平衡优化


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"  # 计算资源
    MEMORY = "memory"  # 内存资源
    STORAGE = "storage"  # 存储资源
    NETWORK = "network"  # 网络资源
    GPU = "gpu"  # GPU资源


class AllocationStrategy(Enum):
    """分配策略"""
    STATIC = "static"  # 静态分配
    DYNAMIC = "dynamic"  # 动态分配
    PREDICTIVE = "predictive"  # 预测性分配
    ADAPTIVE = "adaptive"  # 自适应分配


@dataclass
class ResourceAllocation:
    """资源分配"""
    allocation_id: str
    resource_type: ResourceType
    allocated_amount: float
    max_capacity: float
    utilization: float
    cost_per_unit: float
    allocation_time: datetime
    duration: timedelta


@dataclass
class OptimizationResult:
    """优化结果"""
    optimization_id: str
    objective: OptimizationObjective
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement: Dict[str, float]
    cost_savings: float
    energy_savings: float
    performance_gain: float
    recommendations: List[str]


class EdgeResourceOptimizer:
    """边缘计算资源优化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 资源池管理
        self.resource_pool: Dict[str, Dict[ResourceType, float]] = {}
        
        # 分配记录
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # 优化历史
        self.optimization_history: Dict[str, OptimizationResult] = {}
        
        # 性能监控数据
        self.performance_metrics: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # 预测模型（简化实现）
        self.prediction_models = self._initialize_prediction_models()
        
        # 自动优化任务
        self.auto_optimization_task: Optional[asyncio.Task] = None
        self.is_optimizing = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "optimization_objective": OptimizationObjective.BALANCED,
            "allocation_strategy": AllocationStrategy.ADAPTIVE,
            "optimization_interval": 300,  # 优化间隔(秒)
            "max_resource_utilization": 0.85,  # 最大资源利用率
            "min_resource_utilization": 0.3,  # 最小资源利用率
            "cost_weight": 0.4,  # 成本权重
            "performance_weight": 0.3,  # 性能权重
            "energy_weight": 0.2,  # 能耗权重
            "reliability_weight": 0.1,  # 可靠性权重
            "prediction_horizon": 3600,  # 预测时长(秒)
            "auto_optimization_enabled": True  # 是否启用自动优化
        }
    
    def register_resource_pool(self, 
                             pool_id: str,
                             resources: Dict[ResourceType, float]) -> bool:
        """注册资源池"""
        try:
            if pool_id in self.resource_pool:
                self.logger.warning(f"资源池 {pool_id} 已存在")
                return False
            
            self.resource_pool[pool_id] = resources
            self.performance_metrics[pool_id] = []
            
            self.logger.info(f"资源池注册成功: {pool_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"注册资源池失败: {e}")
            return False
    
    async def allocate_resources(self,
                               pool_id: str,
                               requirements: Dict[ResourceType, float],
                               duration: timedelta = timedelta(hours=1)) -> Optional[Dict[ResourceType, ResourceAllocation]]:
        """分配资源"""
        try:
            if pool_id not in self.resource_pool:
                self.logger.error(f"资源池 {pool_id} 不存在")
                return None
            
            pool_resources = self.resource_pool[pool_id]
            allocations = {}
            
            for resource_type, required_amount in requirements.items():
                # 检查资源可用性
                available = self._get_available_resources(pool_id, resource_type)
                
                if available < required_amount:
                    self.logger.warning(f"资源不足: {resource_type.value}, "
                                      f"需求: {required_amount}, 可用: {available}")
                    
                    # 尝试优化资源分配
                    if await self._optimize_resource_allocation(pool_id, resource_type, required_amount):
                        available = self._get_available_resources(pool_id, resource_type)
                    
                    if available < required_amount:
                        self.logger.error(f"资源分配失败: {resource_type.value}")
                        return None
                
                # 创建资源分配
                allocation_id = f"alloc_{pool_id}_{resource_type.value}_{int(datetime.now().timestamp())}"
                allocation = ResourceAllocation(
                    allocation_id=allocation_id,
                    resource_type=resource_type,
                    allocated_amount=required_amount,
                    max_capacity=pool_resources.get(resource_type, 0.0),
                    utilization=required_amount / pool_resources.get(resource_type, 1.0),
                    cost_per_unit=self._get_resource_cost(resource_type),
                    allocation_time=datetime.now(),
                    duration=duration
                )
                
                self.allocations[allocation_id] = allocation
                allocations[resource_type] = allocation
                
                # 更新资源池
                self._update_resource_pool(pool_id, resource_type, -required_amount)
            
            self.logger.info(f"资源分配成功: {pool_id}, 分配资源: {len(requirements)}种")
            return allocations
            
        except Exception as e:
            self.logger.error(f"资源分配失败: {e}")
            return None
    
    async def optimize_resources(self,
                               pool_id: str,
                               objective: Optional[OptimizationObjective] = None) -> Optional[OptimizationResult]:
        """优化资源分配"""
        try:
            if pool_id not in self.resource_pool:
                self.logger.error(f"资源池 {pool_id} 不存在")
                return None
            
            optimization_objective = objective or self.config["optimization_objective"]
            
            # 收集当前性能指标
            before_metrics = self._collect_current_metrics(pool_id)
            
            # 执行优化
            optimization_result = await self._execute_optimization(pool_id, optimization_objective)
            
            if optimization_result:
                # 收集优化后指标
                after_metrics = self._collect_current_metrics(pool_id)
                
                # 计算改进效果
                improvement = self._calculate_improvement(before_metrics, after_metrics)
                
                # 创建优化结果
                optimization_id = f"opt_{pool_id}_{int(datetime.now().timestamp())}"
                result = OptimizationResult(
                    optimization_id=optimization_id,
                    objective=optimization_objective,
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    improvement=improvement,
                    cost_savings=improvement.get("cost_reduction", 0.0),
                    energy_savings=improvement.get("energy_reduction", 0.0),
                    performance_gain=improvement.get("performance_improvement", 0.0),
                    recommendations=self._generate_recommendations(improvement)
                )
                
                self.optimization_history[optimization_id] = result
                
                self.logger.info(f"资源优化完成: {pool_id}, "
                               f"成本节约: {result.cost_savings:.2f}%, "
                               f"性能提升: {result.performance_gain:.2f}%")
                
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"资源优化失败: {e}")
            return None
    
    async def start_auto_optimization(self) -> bool:
        """启动自动优化"""
        try:
            if self.auto_optimization_task and not self.auto_optimization_task.done():
                self.logger.warning("自动优化任务已在运行")
                return False
            
            if not self.config["auto_optimization_enabled"]:
                self.logger.warning("自动优化未启用")
                return False
            
            self.is_optimizing = True
            self.auto_optimization_task = asyncio.create_task(self._auto_optimization_loop())
            
            self.logger.info("自动资源优化已启动")
            return True
            
        except Exception as e:
            self.logger.error(f"启动自动优化失败: {e}")
            self.is_optimizing = False
            return False
    
    async def stop_auto_optimization(self) -> bool:
        """停止自动优化"""
        try:
            if not self.is_optimizing:
                return True
            
            self.is_optimizing = False
            
            if self.auto_optimization_task:
                self.auto_optimization_task.cancel()
                try:
                    await self.auto_optimization_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("自动资源优化已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止自动优化失败: {e}")
            return False
    
    def get_resource_utilization(self, pool_id: str) -> Dict[ResourceType, float]:
        """获取资源利用率"""
        if pool_id not in self.resource_pool:
            return {}
        
        utilization = {}
        pool_resources = self.resource_pool[pool_id]
        
        for resource_type, total_capacity in pool_resources.items():
            allocated = self._get_allocated_resources(pool_id, resource_type)
            utilization[resource_type] = allocated / total_capacity if total_capacity > 0 else 0.0
        
        return utilization
    
    def predict_resource_demand(self, 
                              pool_id: str,
                              resource_type: ResourceType,
                              horizon: Optional[int] = None) -> List[float]:
        """预测资源需求"""
        try:
            if pool_id not in self.performance_metrics:
                return []
            
            prediction_horizon = horizon or self.config["prediction_horizon"]
            
            # 获取历史数据
            historical_data = self.performance_metrics[pool_id]
            
            if not historical_data:
                return []
            
            # 简化预测实现（实际应用中应使用时间序列预测模型）
            # 这里使用简单的移动平均
            recent_data = [metric for _, metric in historical_data[-10:]]  # 最近10个数据点
            
            if not recent_data:
                return []
            
            # 计算移动平均
            moving_avg = sum(recent_data) / len(recent_data)
            
            # 生成预测序列
            predictions = [moving_avg] * (prediction_horizon // 300)  # 每5分钟一个预测点
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"资源需求预测失败: {e}")
            return []
    
    async def _auto_optimization_loop(self):
        """自动优化循环"""
        while self.is_optimizing:
            try:
                # 对所有资源池进行优化
                for pool_id in self.resource_pool.keys():
                    await self.optimize_resources(pool_id)
                
                # 等待下一个优化周期
                await asyncio.sleep(self.config["optimization_interval"])
                
            except Exception as e:
                self.logger.error(f"自动优化循环异常: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟
    
    async def _execute_optimization(self, 
                                  pool_id: str,
                                  objective: OptimizationObjective) -> bool:
        """执行优化"""
        try:
            strategy = self.config["allocation_strategy"]
            
            if strategy == AllocationStrategy.STATIC:
                return await self._static_optimization(pool_id, objective)
            elif strategy == AllocationStrategy.DYNAMIC:
                return await self._dynamic_optimization(pool_id, objective)
            elif strategy == AllocationStrategy.PREDICTIVE:
                return await self._predictive_optimization(pool_id, objective)
            else:  # ADAPTIVE
                return await self._adaptive_optimization(pool_id, objective)
                
        except Exception as e:
            self.logger.error(f"执行优化失败: {e}")
            return False
    
    async def _static_optimization(self, pool_id: str, objective: OptimizationObjective) -> bool:
        """静态优化"""
        # 基于固定规则的优化
        utilization = self.get_resource_utilization(pool_id)
        
        for resource_type, util in utilization.items():
            if util > self.config["max_resource_utilization"]:
                # 资源过载，需要重新分配
                await self._rebalance_resources(pool_id, resource_type)
            elif util < self.config["min_resource_utilization"]:
                # 资源闲置，可以释放
                await self._release_idle_resources(pool_id, resource_type)
        
        return True
    
    async def _dynamic_optimization(self, pool_id: str, objective: OptimizationObjective) -> bool:
        """动态优化"""
        # 基于实时负载的优化
        current_load = self._get_current_load(pool_id)
        
        # 根据负载调整资源分配
        for resource_type, load in current_load.items():
            target_utilization = self._calculate_target_utilization(resource_type, objective)
            current_utilization = self.get_resource_utilization(pool_id).get(resource_type, 0.0)
            
            if abs(current_utilization - target_utilization) > 0.1:  # 10%偏差
                adjustment = target_utilization - current_utilization
                await self._adjust_resource_allocation(pool_id, resource_type, adjustment)
        
        return True
    
    async def _predictive_optimization(self, pool_id: str, objective: OptimizationObjective) -> bool:
        """预测性优化"""
        # 基于预测的优化
        predictions = {}
        
        for resource_type in ResourceType:
            predictions[resource_type] = self.predict_resource_demand(pool_id, resource_type)
        
        # 根据预测结果提前调整资源
        for resource_type, pred_sequence in predictions.items():
            if pred_sequence:
                avg_demand = sum(pred_sequence) / len(pred_sequence)
                current_utilization = self.get_resource_utilization(pool_id).get(resource_type, 0.0)
                
                # 如果预测需求与当前利用率有显著差异，进行调整
                if abs(avg_demand - current_utilization) > 0.15:  # 15%差异
                    adjustment = avg_demand - current_utilization
                    await self._adjust_resource_allocation(pool_id, resource_type, adjustment)
        
        return True
    
    async def _adaptive_optimization(self, pool_id: str, objective: OptimizationObjective) -> bool:
        """自适应优化"""
        # 结合多种策略的自适应优化
        
        # 1. 基于当前状态优化
        await self._dynamic_optimization(pool_id, objective)
        
        # 2. 基于预测优化
        await self._predictive_optimization(pool_id, objective)
        
        # 3. 特殊场景处理
        await self._handle_special_scenarios(pool_id, objective)
        
        return True
    
    async def _optimize_resource_allocation(self, 
                                          pool_id: str,
                                          resource_type: ResourceType,
                                          required_amount: float) -> bool:
        """优化资源分配以满足需求"""
        try:
            # 检查是否可以重新分配现有资源
            reclaimable = self._get_reclaimable_resources(pool_id, resource_type)
            
            if reclaimable >= required_amount:
                # 重新分配资源
                await self._reallocate_resources(pool_id, resource_type, required_amount)
                return True
            
            # 检查是否可以扩展资源池
            if await self._can_expand_pool(pool_id, resource_type):
                # 扩展资源池
                expansion_amount = required_amount - reclaimable
                await self._expand_resource_pool(pool_id, resource_type, expansion_amount)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"优化资源分配失败: {e}")
            return False
    
    def _get_available_resources(self, pool_id: str, resource_type: ResourceType) -> float:
        """获取可用资源量"""
        if pool_id not in self.resource_pool:
            return 0.0
        
        total_capacity = self.resource_pool[pool_id].get(resource_type, 0.0)
        allocated = self._get_allocated_resources(pool_id, resource_type)
        
        return max(0.0, total_capacity - allocated)
    
    def _get_allocated_resources(self, pool_id: str, resource_type: ResourceType) -> float:
        """获取已分配资源量"""
        allocated = 0.0
        
        for allocation in self.allocations.values():
            if (allocation.resource_type == resource_type and 
                allocation.allocation_id.startswith(f"alloc_{pool_id}")):
                allocated += allocation.allocated_amount
        
        return allocated
    
    def _get_resource_cost(self, resource_type: ResourceType) -> float:
        """获取资源成本"""
        # 简化成本模型
        cost_map = {
            ResourceType.CPU: 0.05,      # $0.05 per CPU-hour
            ResourceType.MEMORY: 0.01,   # $0.01 per GB-hour
            ResourceType.STORAGE: 0.001, # $0.001 per GB-hour
            ResourceType.NETWORK: 0.02,  # $0.02 per GB
            ResourceType.GPU: 0.50       # $0.50 per GPU-hour
        }
        
        return cost_map.get(resource_type, 0.01)
    
    def _update_resource_pool(self, pool_id: str, resource_type: ResourceType, delta: float):
        """更新资源池"""
        if pool_id in self.resource_pool:
            current = self.resource_pool[pool_id].get(resource_type, 0.0)
            self.resource_pool[pool_id][resource_type] = max(0.0, current + delta)
    
    def _collect_current_metrics(self, pool_id: str) -> Dict[str, float]:
        """收集当前性能指标"""
        utilization = self.get_resource_utilization(pool_id)
        
        return {
            "cpu_utilization": utilization.get(ResourceType.CPU, 0.0),
            "memory_utilization": utilization.get(ResourceType.MEMORY, 0.0),
            "storage_utilization": utilization.get(ResourceType.STORAGE, 0.0),
            "network_utilization": utilization.get(ResourceType.NETWORK, 0.0),
            "gpu_utilization": utilization.get(ResourceType.GPU, 0.0),
            "total_cost": self._calculate_total_cost(pool_id),
            "energy_consumption": self._estimate_energy_consumption(pool_id)
        }
    
    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """计算改进效果"""
        improvement = {}
        
        for metric, before_value in before.items():
            after_value = after.get(metric, before_value)
            
            if before_value > 0:
                if "cost" in metric or "consumption" in metric:
                    # 成本或能耗指标，值越小越好
                    improvement[f"{metric}_reduction"] = ((before_value - after_value) / before_value) * 100
                else:
                    # 性能指标，值越大越好
                    improvement[f"{metric}_improvement"] = ((after_value - before_value) / before_value) * 100
        
        return improvement
    
    def _generate_recommendations(self, improvement: Dict[str, float]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        for metric, improvement_value in improvement.items():
            if "reduction" in metric and improvement_value > 5.0:
                recommendations.append(f"{metric.replace('_reduction', '')} 降低了 {improvement_value:.1f}%")
            elif "improvement" in metric and improvement_value > 5.0:
                recommendations.append(f"{metric.replace('_improvement', '')} 提升了 {improvement_value:.1f}%")
        
        if not recommendations:
            recommendations.append("当前资源分配已接近最优状态")
        
        return recommendations
    
    def _initialize_prediction_models(self) -> Dict[str, Any]:
        """初始化预测模型"""
        # 简化实现（实际应用中应使用机器学习模型）
        return {}
    
    # 其他辅助方法（简化实现）
    async def _rebalance_resources(self, pool_id: str, resource_type: ResourceType):
        """重新平衡资源"""
        try:
            self.logger.info(f"开始重新平衡资源池 {pool_id} 的 {resource_type.value} 资源")
            
            # 简化实现：实际应用中需要复杂的资源平衡算法
            # 这里只记录日志并模拟平衡过程
            
            if pool_id in self.resource_pool and resource_type in self.resource_pool[pool_id]:
                # 获取当前资源状态
                current_allocation = self.resource_pool[pool_id][resource_type]
                self.logger.info(f"资源池 {pool_id} 的 {resource_type.value} 当前分配: {current_allocation}")
                
                # 模拟资源平衡：保持当前状态不变
                self.logger.info(f"资源池 {pool_id} 的 {resource_type.value} 资源平衡完成")
            else:
                self.logger.warning(f"资源池 {pool_id} 或资源类型 {resource_type.value} 不存在")
                
        except Exception as e:
            self.logger.error(f"资源平衡失败: {e}")
    
    async def _release_idle_resources(self, pool_id: str, resource_type: ResourceType):
        """释放闲置资源"""
        try:
            self.logger.info(f"开始释放资源池 {pool_id} 的 {resource_type.value} 闲置资源")
            
            # 简化实现：实际应用中需要检测闲置资源的算法
            # 这里只记录日志并模拟释放过程
            
            if pool_id in self.resource_pool and resource_type in self.resource_pool[pool_id]:
                # 模拟释放10%的闲置资源
                current_allocation = self.resource_pool[pool_id][resource_type]
                release_amount = current_allocation * 0.1
                self.resource_pool[pool_id][resource_type] = max(0, current_allocation - release_amount)
                
                self.logger.info(f"已释放资源池 {pool_id} 的 {resource_type.value} 闲置资源: {release_amount:.2f}")
            else:
                self.logger.warning(f"资源池 {pool_id} 或资源类型 {resource_type.value} 不存在")
                
        except Exception as e:
            self.logger.error(f"释放闲置资源失败: {e}")
    
    def _get_current_load(self, pool_id: str) -> Dict[ResourceType, float]:
        """获取当前负载"""
        try:
            # 简化实现：基于资源利用率估算负载
            utilization = self.get_resource_utilization(pool_id)
            
            # 返回与利用率相同的值作为负载（简化实现）
            return utilization
        except Exception as e:
            self.logger.error(f"获取当前负载失败: {e}")
            return {}
    
    def _calculate_target_utilization(self, resource_type: ResourceType, objective: OptimizationObjective) -> float:
        """计算目标利用率"""
        try:
            # 不同资源类型的默认目标利用率
            base_utilization = {
                ResourceType.CPU: 0.8,      # CPU可以有较高利用率
                ResourceType.MEMORY: 0.7,    # 内存需要留有空间防止OOM
                ResourceType.STORAGE: 0.85,  # 存储可以有较高利用率
                ResourceType.NETWORK: 0.65,  # 网络需要留有缓冲
                ResourceType.GPU: 0.85       # GPU可以有较高利用率
            }
            
            # 根据优化目标调整利用率
            objective_adjustment = {
                OptimizationObjective.PERFORMANCE: 0.1,      # 性能优先，提高利用率
                OptimizationObjective.COST: -0.15,           # 成本优先，降低利用率
                OptimizationObjective.ENERGY_EFFICIENCY: 0,   # 能效优先，保持基准
                OptimizationObjective.RELIABILITY: -0.1,     # 可靠性优先，降低利用率
                OptimizationObjective.BALANCED: -0.05        # 平衡，略微降低
            }
            
            # 计算目标利用率并确保在合理范围内
            target = base_utilization.get(resource_type, 0.7) + objective_adjustment.get(objective, 0)
            return max(0.3, min(0.95, target))  # 确保在0.3到0.95之间
        except Exception as e:
            self.logger.error(f"计算目标利用率失败: {e}")
            return 0.7  # 默认目标利用率
    
    async def _adjust_resource_allocation(self, pool_id: str, resource_type: ResourceType, adjustment: float):
        """调整资源分配"""
        try:
            self.logger.info(f"调整资源池 {pool_id} 的 {resource_type.value} 资源分配: {adjustment:+.2f}")
            
            # 初始化资源池（如果不存在）
            if pool_id not in self.resource_pool:
                self.resource_pool[pool_id] = {}
                self.logger.info(f"创建资源池 {pool_id}")
            
            # 初始化资源类型（如果不存在）
            if resource_type not in self.resource_pool[pool_id]:
                self.resource_pool[pool_id][resource_type] = 0.0
            
            # 调整资源分配
            new_allocation = max(0, self.resource_pool[pool_id][resource_type] + adjustment)
            self.resource_pool[pool_id][resource_type] = new_allocation
            
            self.logger.info(f"资源池 {pool_id} 的 {resource_type.value} 资源分配已更新为: {new_allocation:.2f}")
            
        except Exception as e:
            self.logger.error(f"调整资源分配失败: {e}")
    
    async def _handle_special_scenarios(self, pool_id: str, objective: OptimizationObjective):
        """处理特殊场景"""
        try:
            self.logger.info(f"处理资源池 {pool_id} 针对 {objective.value} 目标的特殊场景")
            
            # 简化实现：实际应用中需要处理各种特殊情况
            # 这里只记录日志并返回
            
            self.logger.info(f"资源池 {pool_id} 的特殊场景处理完成")
            
        except Exception as e:
            self.logger.error(f"处理特殊场景失败: {e}")
    
    def _get_reclaimable_resources(self, pool_id: str, resource_type: ResourceType) -> float:
        """获取可回收资源"""
        try:
            if pool_id not in self.resource_pool or resource_type not in self.resource_pool[pool_id]:
                return 0.0
            
            # 获取当前资源池中的资源总量
            total_allocated = self.resource_pool[pool_id][resource_type]
            
            # 获取实际分配给任务的资源量
            actually_used = self._get_allocated_resources(pool_id, resource_type)
            
            # 计算可回收资源：当前分配的资源总量减去实际使用的资源量
            # 但需要保留一定的缓冲空间
            reclaimable = max(0, total_allocated - actually_used)
            
            # 保留20%的缓冲空间，确保系统稳定
            buffer = total_allocated * 0.2
            reclaimable = max(0, reclaimable - buffer)
            
            self.logger.info(f"资源池 {pool_id} 的 {resource_type.value} 可回收资源: {reclaimable:.2f}")
            return reclaimable
        except Exception as e:
            self.logger.error(f"获取可回收资源失败: {e}")
            return 0.0
    
    async def _reallocate_resources(self, pool_id: str, resource_type: ResourceType, amount: float):
        """重新分配资源"""
        try:
            self.logger.info(f"重新分配资源池 {pool_id} 的 {resource_type.value} 资源: {amount:.2f}")
            
            # 初始化资源池（如果不存在）
            if pool_id not in self.resource_pool:
                self.resource_pool[pool_id] = {}
                self.logger.info(f"创建资源池 {pool_id}")
            
            # 重新分配资源
            self.resource_pool[pool_id][resource_type] = max(0, amount)
            
            self.logger.info(f"资源池 {pool_id} 的 {resource_type.value} 资源已重新分配为: {amount:.2f}")
            
        except Exception as e:
            self.logger.error(f"重新分配资源失败: {e}")
    
    async def _can_expand_pool(self, pool_id: str, resource_type: ResourceType) -> bool:
        """检查是否可以扩展资源池"""
        try:
            if pool_id not in self.resource_pool:
                return True  # 资源池不存在，可以创建
            
            # 检查是否配置了资源池最大容量
            max_capacity = self.config.get(f"max_{resource_type.value}_capacity")
            
            if max_capacity:
                current_capacity = self.resource_pool[pool_id].get(resource_type, 0.0)
                if current_capacity >= max_capacity:
                    self.logger.info(f"资源池 {pool_id} 的 {resource_type.value} 已达到最大容量: {current_capacity}/{max_capacity}")
                    return False
            
            # 根据资源类型判断扩展可能性
            expandable_resources = {
                ResourceType.CPU: False,       # CPU通常难以动态扩展
                ResourceType.MEMORY: False,     # 内存通常难以动态扩展
                ResourceType.STORAGE: True,     # 存储通常可以动态扩展
                ResourceType.NETWORK: True,     # 网络可以通过增加带宽等方式扩展
                ResourceType.GPU: False         # GPU通常难以动态扩展
            }
            
            is_expandable = expandable_resources.get(resource_type, False)
            self.logger.info(f"资源池 {pool_id} 的 {resource_type.value} 是否可扩展: {is_expandable}")
            
            return is_expandable
        except Exception as e:
            self.logger.error(f"检查资源池扩展可能性失败: {e}")
            return False
    
    async def _expand_resource_pool(self, pool_id: str, resource_type: ResourceType, amount: float):
        """扩展资源池"""
        try:
            self.logger.info(f"扩展资源池 {pool_id} 的 {resource_type.value} 资源: +{amount:.2f}")
            
            # 初始化资源池（如果不存在）
            if pool_id not in self.resource_pool:
                self.resource_pool[pool_id] = {}
                self.logger.info(f"创建资源池 {pool_id}")
            
            # 初始化资源类型（如果不存在）
            if resource_type not in self.resource_pool[pool_id]:
                self.resource_pool[pool_id][resource_type] = 0.0
            
            # 扩展资源池
            new_allocation = self.resource_pool[pool_id][resource_type] + amount
            self.resource_pool[pool_id][resource_type] = new_allocation
            
            self.logger.info(f"资源池 {pool_id} 的 {resource_type.value} 资源已扩展为: {new_allocation:.2f}")
            
        except Exception as e:
            self.logger.error(f"扩展资源池失败: {e}")
    
    def _calculate_total_cost(self, pool_id: str) -> float:
        """计算总成本"""
        try:
            total_cost = 0.0
            
            # 遍历所有分配记录
            for allocation in self.allocations.values():
                # 检查是否属于指定资源池
                if allocation.allocation_id.startswith(f"alloc_{pool_id}"):
                    # 计算资源分配的成本：资源量 * 单位成本 * 持续时间（小时）
                    duration_hours = allocation.duration.total_seconds() / 3600
                    allocation_cost = allocation.allocated_amount * allocation.cost_per_unit * duration_hours
                    
                    total_cost += allocation_cost
                    self.logger.debug(f"资源分配 {allocation.allocation_id} 的成本: {allocation_cost:.2f}")
            
            self.logger.info(f"资源池 {pool_id} 的总成本: {total_cost:.2f}")
            return total_cost
        except Exception as e:
            self.logger.error(f"计算总成本失败: {e}")
            return 0.0
    
    def _estimate_energy_consumption(self, pool_id: str) -> float:
        """估算能耗"""
        try:
            total_energy = 0.0
            
            # 不同资源类型的基础能耗（kWh/单位/小时）
            base_energy_per_unit = {
                ResourceType.CPU: 0.05,      # 每CPU核心每小时消耗0.05 kWh
                ResourceType.MEMORY: 0.005,   # 每GB内存每小时消耗0.005 kWh
                ResourceType.STORAGE: 0.001,  # 每GB存储每小时消耗0.001 kWh
                ResourceType.NETWORK: 0.002,  # 每Gbps网络每小时消耗0.002 kWh
                ResourceType.GPU: 0.25        # 每GPU每小时消耗0.25 kWh
            }
            
            # 遍历所有分配记录
            for allocation in self.allocations.values():
                # 检查是否属于指定资源池
                if allocation.allocation_id.startswith(f"alloc_{pool_id}"):
                    # 获取资源类型的基础能耗
                    base_energy = base_energy_per_unit.get(allocation.resource_type, 0.001)
                    
                    # 计算能耗：基础能耗 * 资源量 * 利用率 * 持续时间（小时）
                    duration_hours = allocation.duration.total_seconds() / 3600
                    energy = base_energy * allocation.allocated_amount * allocation.utilization * duration_hours
                    
                    total_energy += energy
                    self.logger.debug(f"资源分配 {allocation.allocation_id} 的能耗: {energy:.2f} kWh")
            
            self.logger.info(f"资源池 {pool_id} 的总能耗: {total_energy:.2f} kWh")
            return total_energy
        except Exception as e:
            self.logger.error(f"估算能耗失败: {e}")
            return 0.0