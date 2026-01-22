"""
光谱硬件控制器 - 管理LED光谱系统的硬件操作
支持380nm紫外线、720nm远红外线、31:1白红配比等高级功能
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import serial
import logging

logger = logging.getLogger(__name__)


class LEDBand(Enum):
    """LED波段枚举"""
    UV_380NM = "uv_380nm"      # 380nm紫外线
    FAR_RED_720NM = "far_red_720nm"  # 720nm远红外线
    WHITE_LIGHT = "white_light"    # 白光 (420-450nm, 18000K)
    RED_660NM = "red_660nm"      # 660nm红光


@dataclass
class HardwareConfig:
    """硬件配置"""
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 9600
    timeout: float = 1.0
    led_count: int = 20000  # 总LED数量
    uv_count: int = 200     # 380nm紫外线LED数量
    far_red_count: int = 400  # 720nm远红外线LED数量
    white_count: int = 18000  # 白光LED数量
    red_count: int = 400    # 660nm红光LED数量


class SpectrumHardwareController:
    """光谱硬件控制器"""
    
    def __init__(self, config: HardwareConfig = None):
        self.config = config or HardwareConfig()
        self.serial_connection = None
        self.is_connected = False
        
        # 验证LED数量配置
        total_configured = (self.config.uv_count + self.config.far_red_count + 
                          self.config.white_count + self.config.red_count)
        if total_configured != self.config.led_count:
            logger.warning(f"LED数量配置不匹配: 配置{total_configured}个, 总数为{self.config.led_count}个")
    
    async def connect(self) -> bool:
        """连接硬件设备"""
        try:
            self.serial_connection = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.baud_rate,
                timeout=self.config.timeout
            )
            self.is_connected = True
            logger.info(f"成功连接到硬件设备: {self.config.serial_port}")
            return True
        except Exception as e:
            logger.error(f"连接硬件设备失败: {e}")
            return False
    
    async def disconnect(self):
        """断开硬件连接"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.is_connected = False
        logger.info("已断开硬件连接")
    
    async def set_spectrum_recipe(self, recipe: Dict[str, float]) -> bool:
        """设置光谱配方"""
        if not self.is_connected:
            logger.error("硬件未连接")
            return False
        
        try:
            # 验证配方数据
            self._validate_recipe(recipe)
            
            # 转换为硬件指令
            commands = self._recipe_to_commands(recipe)
            
            # 发送指令到硬件
            for command in commands:
                await self._send_command(command)
            
            logger.info(f"成功设置光谱配方: {recipe}")
            return True
            
        except Exception as e:
            logger.error(f"设置光谱配方失败: {e}")
            return False
    
    async def get_current_status(self) -> Dict[str, Any]:
        """获取当前硬件状态"""
        if not self.is_connected:
            return {"connected": False, "error": "硬件未连接"}
        
        try:
            # 发送状态查询指令
            status_command = {"command": "get_status"}
            response = await self._send_command(status_command)
            
            return {
                "connected": True,
                "status": response,
                "hardware_info": {
                    "total_leds": self.config.led_count,
                    "uv_leds": self.config.uv_count,
                    "far_red_leds": self.config.far_red_count,
                    "white_leds": self.config.white_count,
                    "red_leds": self.config.red_count
                }
            }
        except Exception as e:
            return {"connected": True, "error": str(e)}
    
    async def apply_white_red_ratio(self, target_ratio: float = 31.0) -> bool:
        """应用白红配比约束"""
        try:
            # 获取当前状态
            current_status = await self.get_current_status()
            
            if not current_status.get("connected"):
                return False
            
            # 计算需要调整的比例
            current_recipe = self._extract_current_recipe(current_status)
            current_ratio = current_recipe.get("white_light", 0) / current_recipe.get("red_660nm", 1)
            
            if abs(current_ratio - target_ratio) < 0.1:
                logger.info("白红配比已符合要求")
                return True
            
            # 调整配方
            adjusted_recipe = self._adjust_white_red_ratio(current_recipe, target_ratio)
            
            # 应用新配方
            return await self.set_spectrum_recipe(adjusted_recipe)
            
        except Exception as e:
            logger.error(f"应用白红配比失败: {e}")
            return False
    
    async def start_auto_optimization(self, crop_type: str, growth_stage: str) -> bool:
        """启动自动优化模式"""
        try:
            optimization_command = {
                "command": "start_auto_optimization",
                "crop_type": crop_type,
                "growth_stage": growth_stage
            }
            
            response = await self._send_command(optimization_command)
            logger.info(f"启动自动优化模式: {crop_type} - {growth_stage}")
            return response.get("success", False)
            
        except Exception as e:
            logger.error(f"启动自动优化模式失败: {e}")
            return False
    
    async def set_light_schedule(self, schedule: Dict[str, Any]) -> bool:
        """设置光照时间表"""
        try:
            # 验证时间表数据
            self._validate_schedule(schedule)
            
            schedule_command = {
                "command": "set_schedule",
                "schedule": schedule
            }
            
            response = await self._send_command(schedule_command)
            logger.info("成功设置光照时间表")
            return response.get("success", False)
            
        except Exception as e:
            logger.error(f"设置光照时间表失败: {e}")
            return False
    
    def _validate_recipe(self, recipe: Dict[str, float]):
        """验证配方数据"""
        required_bands = [band.value for band in LEDBand]
        
        for band in required_bands:
            if band not in recipe:
                raise ValueError(f"缺少必要的波段: {band}")
            
            value = recipe[band]
            if not 0 <= value <= 1:
                raise ValueError(f"波段强度值必须在0-1之间: {band} = {value}")
        
        # 验证总和接近1
        total = sum(recipe.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"波段强度总和必须为1, 当前为: {total}")
    
    def _recipe_to_commands(self, recipe: Dict[str, float]) -> List[Dict]:
        """将配方转换为硬件指令"""
        commands = []
        
        # 设置各波段强度
        for band, intensity in recipe.items():
            command = {
                "command": "set_intensity",
                "band": band,
                "intensity": intensity,
                "led_count": getattr(self.config, f"{band}_count")
            }
            commands.append(command)
        
        return commands
    
    def _validate_schedule(self, schedule: Dict[str, Any]):
        """验证时间表数据"""
        required_fields = ["start_time", "end_time", "light_hours", "recipe"]
        
        for field in required_fields:
            if field not in schedule:
                raise ValueError(f"缺少必要的时间表字段: {field}")
    
    def _extract_current_recipe(self, status: Dict[str, Any]) -> Dict[str, float]:
        """从状态信息中提取当前配方"""
        # 这里需要根据实际的硬件响应格式进行解析
        # 假设硬件返回包含各波段强度的数据
        return status.get("status", {}).get("current_recipe", {
            "uv_380nm": 0.05,
            "far_red_720nm": 0.1,
            "white_light": 0.7,
            "red_660nm": 0.15
        })
    
    def _adjust_white_red_ratio(self, recipe: Dict[str, float], target_ratio: float) -> Dict[str, float]:
        """调整白红配比"""
        current_white = recipe.get("white_light", 0.7)
        current_red = recipe.get("red_660nm", 0.15)
        
        # 计算调整后的值
        total_white_red = current_white + current_red
        new_white = total_white_red * (target_ratio / (target_ratio + 1))
        new_red = total_white_red * (1 / (target_ratio + 1))
        
        # 更新配方
        adjusted_recipe = recipe.copy()
        adjusted_recipe["white_light"] = new_white
        adjusted_recipe["red_660nm"] = new_red
        
        return adjusted_recipe
    
    async def _send_command(self, command: Dict) -> Dict:
        """发送命令到硬件"""
        if not self.serial_connection:
            raise RuntimeError("串口连接未建立")
        
        try:
            # 序列化命令
            command_str = json.dumps(command) + "\n"
            
            # 使用异步线程执行同步的serial操作，避免阻塞事件循环
            async def send_and_receive():
                try:
                    # 发送命令
                    self.serial_connection.write(command_str.encode('utf-8'))
                    
                    # 读取响应
                    response_line = self.serial_connection.readline().decode('utf-8').strip()
                    
                    if response_line:
                        return json.loads(response_line)
                    else:
                        return {"success": False, "error": "无响应"}
                except Exception as e:
                    logger.error(f"发送命令失败: {e}")
                    return {"success": False, "error": str(e)}
            
            return await asyncio.to_thread(send_and_receive)
                
        except Exception as e:
            logger.error(f"发送命令失败: {e}")
            return {"success": False, "error": str(e)}


class AdvancedSpectrumFeatures:
    """高级光谱特性"""
    
    def __init__(self, hardware_controller: SpectrumHardwareController):
        self.controller = hardware_controller
    
    async def optimize_uv_for_quality(self, crop_type: str, target_quality: str) -> bool:
        """优化紫外线波段以提升品质"""
        try:
            # 根据作物类型和目标品质调整紫外线强度
            uv_intensity_map = {
                "番茄": {"提升甜度": 0.08, "提升抗性": 0.1, "最大化产量": 0.06},
                "生菜": {"提升品质": 0.07, "提升营养": 0.09, "最大化产量": 0.05}
            }
            
            target_uv = uv_intensity_map.get(crop_type, {}).get(target_quality, 0.06)
            
            # 获取当前状态
            current_status = await self.controller.get_current_status()
            current_recipe = self.controller._extract_current_recipe(current_status)
            
            # 调整配方
            adjusted_recipe = current_recipe.copy()
            adjusted_recipe["uv_380nm"] = target_uv
            
            # 保持总和为1
            total_others = sum(v for k, v in adjusted_recipe.items() if k != "uv_380nm")
            scale_factor = (1 - target_uv) / total_others if total_others > 0 else 1
            
            for band in adjusted_recipe:
                if band != "uv_380nm":
                    adjusted_recipe[band] *= scale_factor
            
            return await self.controller.set_spectrum_recipe(adjusted_recipe)
            
        except Exception as e:
            logger.error(f"优化紫外线波段失败: {e}")
            return False
    
    async def dynamic_far_red_control(self, temperature: float, humidity: float) -> bool:
        """动态远红外控制"""
        try:
            # 根据环境条件调整远红外强度
            # 高温高湿时适当降低远红外强度
            base_intensity = 0.1
            
            if temperature > 28:
                base_intensity *= 0.8
            if humidity > 75:
                base_intensity *= 0.9
            
            # 获取当前状态并调整
            current_status = await self.controller.get_current_status()
            current_recipe = self.controller._extract_current_recipe(current_status)
            
            adjusted_recipe = current_recipe.copy()
            adjusted_recipe["far_red_720nm"] = base_intensity
            
            # 保持总和为1
            total_others = sum(v for k, v in adjusted_recipe.items() if k != "far_red_720nm")
            scale_factor = (1 - base_intensity) / total_others if total_others > 0 else 1
            
            for band in adjusted_recipe:
                if band != "far_red_720nm":
                    adjusted_recipe[band] *= scale_factor
            
            return await self.controller.set_spectrum_recipe(adjusted_recipe)
            
        except Exception as e:
            logger.error(f"动态远红外控制失败: {e}")
            return False
    
    async def energy_saving_mode(self, enable: bool) -> bool:
        """节能模式"""
        try:
            if enable:
                # 节能模式：降低总体强度但保持比例
                energy_saving_recipe = {
                    "uv_380nm": 0.03,
                    "far_red_720nm": 0.06,
                    "white_light": 0.42,
                    "red_660nm": 0.49
                }
            else:
                # 恢复正常模式
                energy_saving_recipe = {
                    "uv_380nm": 0.05,
                    "far_red_720nm": 0.1,
                    "white_light": 0.7,
                    "red_660nm": 0.15
                }
            
            return await self.controller.set_spectrum_recipe(energy_saving_recipe)
            
        except Exception as e:
            logger.error(f"设置节能模式失败: {e}")
            return False


# 硬件控制器单例实例
hardware_controller = SpectrumHardwareController()
advanced_features = AdvancedSpectrumFeatures(hardware_controller)