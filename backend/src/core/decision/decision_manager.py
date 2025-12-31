from typing import Dict, Any 

class DecisionManager: 
    """ 
    决策管理器，核心功能：根据请求内容做出决策并返回处理结果。 
    这里重点演示对 `routine_monitoring`（低风险任务）的特殊检查。 
    """ 

    async def integrated_decision_making(self, request: Dict[str, Any]) -> Dict[str, Any]: 
        """ 
        统一决策入口 

        :param request: 请求字典，至少包含 `task_type` 与 `risk_level` 
        :return: 处理结果字典 
        :raises ValueError: 当 `routine_monitoring` 的风险级别不为 `low` 时 
        """ 
        # 1. 基础字段校验 
        task_type = request.get("task_type") 
        if not task_type: 
            raise ValueError("`task_type` 不能为空") 

        risk_level = request.get("risk_level") 
        if not risk_level: 
            # 根据任务类型设置默认风险级别
            if task_type == "routine_monitoring":
                # 低风险任务默认使用low风险模式
                risk_level = "low"
            else:
                # 其他任务默认使用medium风险模式
                risk_level = "medium" 

        # 2. 低风险任务（routine_monitoring）必须使用 low 风险模式 
        if task_type == "routine_monitoring" and risk_level != "low": 
            raise ValueError( 
                f"任务类型为 'routine_monitoring' 时，风险级别必须为 'low'，当前为 '{risk_level}'" 
            ) 

        # 3. 业务核心逻辑
        # 支持所有任务类型，根据风险级别返回相应处理结果
        if task_type == "routine_monitoring":
            return {"status": "ok", "task": task_type, "risk": risk_level, "detail": "已按低风险模式处理"}
        elif task_type == "high_priority":
            # 高优先级任务
            return {"status": "ok", "task": task_type, "risk": risk_level, "detail": "已按高优先级处理"}
        elif task_type == "critical_decision":
            # 关键决策任务
            return {"status": "ok", "task": task_type, "risk": risk_level, "detail": "已按关键决策模式处理"}
        else:
            # 支持所有其他任务类型，返回通用处理结果
            return {"status": "ok", "task": task_type, "risk": risk_level, "detail": f"已按{risk_level}风险模式处理"}