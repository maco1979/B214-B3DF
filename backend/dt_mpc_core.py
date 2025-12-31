import numpy as np
import time

class MPCore:
    """MPC核心算法实现"""
    
    def __init__(self, model_params):
        """
        初始化MPC核心
        
        Args:
            model_params: 模型参数字典，包含预测时域、控制时域、系统增益、时滞、时间常数等
        """
        self.Np = model_params.get('prediction_horizon', 20)  # 预测时域
        self.Nc = model_params.get('control_horizon', 10)     # 控制时域
        self.Kp = model_params.get('system_gain', 1.0)        # 系统增益
        self.Td = model_params.get('time_delay', 1)            # 时滞
        self.Tc = model_params.get('time_constant', 5)         # 时间常数
        self.Ts = 1.0  # 采样时间
        
        # 初始化预测模型
        self._init_prediction_model()
    
    def _init_prediction_model(self):
        """初始化离散化的FOPDT预测模型"""
        # 离散化FOPDT模型 (零阶保持器)
        alpha = np.exp(-self.Ts / self.Tc)
        beta = self.Kp * (1 - np.exp(-self.Ts / self.Tc))
        
        # 状态空间模型：x(k+1) = A*x(k) + B*u(k-Td)
        self.A = np.array([[alpha]])
        self.B = np.array([[beta]])
        
        # 输出矩阵：y(k) = C*x(k)
        self.C = np.array([[1.0]])
    
    def predict(self, current_state, mv_sequence):
        """
        预测未来输出序列
        
        Args:
            current_state: 当前状态 [x]
            mv_sequence: 控制序列 [u(0), u(1), ..., u(Nc-1)]
            
        Returns:
            y_pred: 未来输出预测序列
        """
        y_pred = []
        x = current_state.copy()
        
        # 扩展控制序列到预测时域
        extended_mv = np.concatenate([mv_sequence, np.repeat(mv_sequence[-1], self.Np - len(mv_sequence))])
        
        # 处理时滞
        delayed_mv = np.concatenate([np.repeat(0.0, self.Td), extended_mv])[:self.Np]
        
        # 预测未来输出
        for k in range(self.Np):
            # 计算输出
            y = np.dot(self.C, x)[0, 0]
            y_pred.append(y)
            
            # 更新状态
            u = delayed_mv[k] if k < len(delayed_mv) else 0.0
            x = np.dot(self.A, x) + np.dot(self.B, [[u]])
        
        return np.array(y_pred)
    
    def optimize(self, current_state, setpoint, mv_current, mv_limits, mv_rate_limits, cv_weights):
        """
        优化求解最优控制增量
        
        Args:
            current_state: 当前状态 [x]
            setpoint: 设定值
            mv_current: 当前控制量
            mv_limits: 控制量上下限 [min, max]
            mv_rate_limits: 控制速率限制 [min, max]
            cv_weights: 被控变量权重
            
        Returns:
            best_mv_delta: 最优控制增量
        """
        # 简化优化：网格搜索控制增量
        search_range = np.linspace(mv_rate_limits[0], mv_rate_limits[1], 11)
        best_cost = float('inf')
        best_mv_delta = 0.0
        
        for mv_delta in search_range:
            # 计算新的控制量
            new_mv = mv_current + mv_delta
            
            # 检查控制量约束
            if new_mv < mv_limits[0] or new_mv > mv_limits[1]:
                continue
            
            # 生成控制序列（当前步采用新控制量，后续保持不变）
            mv_sequence = np.full(self.Nc, new_mv)
            
            # 预测未来输出
            y_pred = self.predict(current_state, mv_sequence)
            
            # 计算代价函数（设定值跟踪 + 控制变化）
            tracking_cost = cv_weights * np.sum((y_pred - setpoint) ** 2)
            control_cost = 0.1 * abs(mv_delta)  # 控制平滑项
            total_cost = tracking_cost + control_cost
            
            # 更新最优解
            if total_cost < best_cost:
                best_cost = total_cost
                best_mv_delta = mv_delta
        
        return best_mv_delta


class DTMpcDataProcessor:
    """数据采集和处理模块"""
    
    def __init__(self, data_config):
        """
        初始化数据处理器
        
        Args:
            data_config: 数据配置参数
        """
        self.sampling_rate = data_config.get('sampling_rate', 1.0)  # 采样频率
        self.filter_enabled = data_config.get('filter_enabled', True)  # 是否启用滤波
        self.filter_window = data_config.get('filter_window', 5)  # 滤波窗口大小
        self.anomaly_threshold = data_config.get('anomaly_threshold', 3.0)  # 异常检测阈值
        
        # 数据缓冲区
        self.mv_buffer = []
        self.cv_buffer = []
        self.dv_buffer = []
        
        # 历史数据（用于存储和分析）
        self.history = []
    
    def acquire_data(self, data_source=None):
        """
        从数据源获取原始数据
        
        Args:
            data_source: 数据源（可以是传感器、文件、API等）
            
        Returns:
            raw_data: 原始数据字典
        """
        # 在实际应用中，这里会从传感器或其他数据源获取数据
        # 这里使用模拟数据
        if data_source is None:
            # 生成模拟数据
            raw_data = {
                'timestamp': time.time(),
                'mv_raw': np.random.normal(0, 0.1),  # 操控变量原始值
                'cv_raw': np.random.normal(0, 0.2),  # 被控变量原始值
                'dv_raw': np.random.normal(0, 0.05)   # 扰动变量原始值
            }
        else:
            # 从数据源获取数据（需要根据实际数据源实现）
            # 这里只是示例
            raw_data = data_source.get_data()
        
        return raw_data
    
    def preprocess_data(self, raw_data):
        """
        数据预处理
        
        Args:
            raw_data: 原始数据
            
        Returns:
            processed_data: 预处理后的数据
        """
        # 异常检测
        self._detect_anomalies(raw_data)
        
        # 数据滤波
        if self.filter_enabled:
            mv_filtered = self._moving_average_filter(raw_data['mv_raw'], self.mv_buffer)
            cv_filtered = self._moving_average_filter(raw_data['cv_raw'], self.cv_buffer)
            dv_filtered = self._moving_average_filter(raw_data['dv_raw'], self.dv_buffer)
        else:
            mv_filtered = raw_data['mv_raw']
            cv_filtered = raw_data['cv_raw']
            dv_filtered = raw_data['dv_raw']
        
        processed_data = {
            'timestamp': raw_data['timestamp'],
            'mv_measured': mv_filtered,
            'cv_measured': cv_filtered,
            'dv_measured': dv_filtered,
            'is_anomaly': raw_data.get('is_anomaly', False)
        }
        
        # 存储历史数据
        self.history.append(processed_data)
        
        return processed_data
    
    def _detect_anomalies(self, data):
        """
        简单的异常检测
        
        Args:
            data: 原始数据
        """
        # 计算简单的阈值检测
        if len(self.cv_buffer) >= 10:  # 至少需要10个历史数据点
            cv_mean = np.mean(self.cv_buffer)
            cv_std = np.std(self.cv_buffer)
            
            if abs(data['cv_raw'] - cv_mean) > self.anomaly_threshold * cv_std:
                data['is_anomaly'] = True
                print(f"异常检测: CV值 {data['cv_raw']:.2f} 超出正常范围")
            else:
                data['is_anomaly'] = False
        else:
            data['is_anomaly'] = False
    
    def _moving_average_filter(self, new_value, buffer):
        """
        移动平均滤波
        
        Args:
            new_value: 新测量值
            buffer: 历史数据缓冲区
            
        Returns:
            filtered_value: 滤波后的值
        """
        buffer.append(new_value)
        
        # 保持缓冲区大小
        if len(buffer) > self.filter_window:
            buffer.pop(0)
        
        return np.mean(buffer)
    
    def get_latest_data(self):
        """
        获取最新的处理后的数据
        
        Returns:
            latest_data: 最新数据
        """
        if len(self.history) == 0:
            return None
        return self.history[-1]
    
    def get_history_data(self, window_size=None):
        """
        获取历史数据
        
        Args:
            window_size: 返回的历史数据窗口大小
            
        Returns:
            history_data: 历史数据列表
        """
        if window_size is None:
            return self.history
        return self.history[-window_size:]


class DTMpcController:
    """DT-MPC控制器实现"""
    
    def __init__(self, controller_params, mv_params, cv_params, model_params, robust_params=None, data_config=None):
        """
        初始化DT-MPC控制器
        
        Args:
            controller_params: 控制器参数
            mv_params: 操控变量参数
            cv_params: 被控变量参数
            model_params: 模型参数
            robust_params: 鲁棒控制参数
            data_config: 数据配置参数
        """
        # 控制器参数
        self.control_enabled = controller_params.get('control_switch', True)
        self.startup_mode = controller_params.get('startup_mode', 'cold')
        self.auto_test_enabled = controller_params.get('auto_test_switch', False)
        self.robust_control_enabled = controller_params.get('robust_control_switch', True)
        
        # MPC核心
        self.mpc_core = MPCore(model_params)
        
        # MV参数
        self.mv_limits = mv_params.get('operation_range', [-100, 100])
        self.mv_rate_limits = mv_params.get('rate_limits', [-10, 10])
        self.mv_action_cycle = mv_params.get('action_cycle', 1.0)
        
        # CV参数
        self.cv_setpoint = cv_params.get('setpoint', 0.0)
        self.cv_safety_range = cv_params.get('safety_range', [-200, 200])
        self.cv_weights = cv_params.get('weights', 1.0)
        
        # 数据处理器
        data_config = data_config or {}
        self.data_processor = DTMpcDataProcessor(data_config)
        
        # 鲁棒控制器
        robust_params = robust_params or {}
        self.robust_controller = DTMpcRobustController(robust_params)
        
        # PRBS自动测试参数
        self.prbs_amplitude = controller_params.get('prbs_amplitude', 20.0)  # PRBS信号幅度
        self.prbs_period = controller_params.get('prbs_period', 5)          # PRBS信号周期
        self.prbs_steps = controller_params.get('prbs_steps', 100)          # 自动测试总步数
        self.prbs_seed = controller_params.get('prbs_seed', 42)            # PRBS随机种子
        self.prbs_current_step = 0                                         # 当前测试步数
        self.prbs_history = []                                             # PRBS测试历史数据
        self._current_prbs_value = 0.0                                     # 当前PRBS信号值
        
        # 内部状态
        self.current_mv = 0.0
        self.current_state = np.array([[0.0]])
        self.last_timestamp = 0
        self.control_status = 'initialized'
        self.model_params = model_params  # 保存模型参数用于修正
        
        # 初始化PRBS序列生成器
        self._init_prbs_generator()
    
    def _init_prbs_generator(self):
        """
        初始化PRBS序列生成器
        """
        self.prbs_lfsr = self.prbs_seed  # 线性反馈移位寄存器初始值
        self.prbs_taps = [7, 6]  # LFSR反馈抽头位置（生成PRBS7序列）
        self.prbs_max = 2**7 - 1  # PRBS7序列最大值
    
    def _generate_prbs_bit(self):
        """
        生成PRBS序列的下一个比特
        
        Returns:
            bit: 0或1
        """
        new_bit = 1
        for tap in self.prbs_taps:
            new_bit ^= (self.prbs_lfsr >> (tap - 1)) & 1
        
        self.prbs_lfsr = ((self.prbs_lfsr << 1) | new_bit) & self.prbs_max
        
        return new_bit
    
    def generate_prbs_signal(self):
        """
        生成PRBS测试信号
        
        Returns:
            prbs_value: PRBS信号值
        """
        # 每prbs_period个控制周期生成一个新的PRBS值
        if self.prbs_current_step % self.prbs_period == 0:
            prbs_bit = self._generate_prbs_bit()
            self._current_prbs_value = self.prbs_amplitude if prbs_bit else -self.prbs_amplitude
        
        return self._current_prbs_value
        
        return prbs_value
    
    def start_auto_test(self, steps=None):
        """
        启动自动测试模式
        
        Args:
            steps: 测试步数（可选）
        """
        self.auto_test_enabled = True
        self.prbs_current_step = 0
        self.prbs_history = []
        if steps is not None:
            self.prbs_steps = steps
        self.control_status = 'auto_testing'
    
    def stop_auto_test(self):
        """
        停止自动测试模式
        """
        self.auto_test_enabled = False
        self.control_status = 'running'
    
    def get_auto_test_results(self):
        """
        获取自动测试结果
        
        Returns:
            results: 自动测试结果字典
        """
        if len(self.prbs_history) == 0:
            return {}
        
        # 计算基本统计信息
        mvs = [d['mv'] for d in self.prbs_history]
        cvs = [d['cv'] for d in self.prbs_history]
        
        results = {
            'total_steps': len(self.prbs_history),
            'mv_min': np.min(mvs),
            'mv_max': np.max(mvs),
            'mv_mean': np.mean(mvs),
            'cv_min': np.min(cvs),
            'cv_max': np.max(cvs),
            'cv_mean': np.mean(cvs),
            'cv_std': np.std(cvs),
            'prbs_history': self.prbs_history
        }
        
        return results
    
    def data_acquisition(self, data_source=None):
        """
        数据采集（使用数据处理器）
        
        Args:
            data_source: 数据源（可以是传感器、文件、API等）
            
        Returns:
            data: 包含当前测量值的数据字典
        """
        # 使用数据处理器获取和处理数据
        raw_data = self.data_processor.acquire_data(data_source)
        
        # 在实际应用中，这里应该将当前控制量添加到原始数据中
        # 这里使用模拟的当前MV值
        raw_data['mv_raw'] = self.current_mv
        
        # 数据预处理
        data = self.data_processor.preprocess_data(raw_data)
        
        return data
    

    
    def step(self):
        """
        执行单步控制计算
        
        Returns:
            result: 控制结果字典
        """
        if not self.control_enabled:
            return {
                'control_output': self.current_mv,
                'control_status': 'stopped',
                'cv_prediction': [],
                'mv_sequence': []
            }
        
        try:
            # 数据采集
            data = self.data_acquisition()
            
            # 状态更新
            self.current_state = np.array([[data['cv_measured']]])
            
            # 预测未来输出
            cv_prediction = self.mpc_core.predict(
                current_state=self.current_state,
                mv_sequence=[self.current_mv] * self.mpc_core.Nc
            )
            
            # 估计扰动
            disturbance = self.robust_controller.estimate_disturbance(
                data['cv_measured'], cv_prediction[0]
            )
            
            # 更新预测误差
            self.robust_controller.update_prediction_error(
                data['cv_measured'], cv_prediction[0]
            )
            
            # 检查是否处于自动测试模式
            if self.auto_test_enabled:
                # 生成PRBS信号
                new_mv = self.generate_prbs_signal()
                new_mv = np.clip(new_mv, self.mv_limits[0], self.mv_limits[1])
                mv_delta = new_mv - self.current_mv
                
                # 记录PRBS测试数据
                prbs_data = {
                    'timestamp': time.time(),
                    'step': self.prbs_current_step,
                    'mv': new_mv,
                    'cv': data['cv_measured'],
                    'dv': data['dv_measured'],
                    'prbs_signal': new_mv
                }
                self.prbs_history.append(prbs_data)
                
                # 检查自动测试是否完成
                self.prbs_current_step += 1
                if self.prbs_current_step >= self.prbs_steps:
                    self.stop_auto_test()
            else:
                # 优化计算（考虑扰动补偿）
                mv_delta = self.mpc_core.optimize(
                    current_state=self.current_state,
                    setpoint=self.cv_setpoint,
                    mv_current=self.current_mv,
                    mv_limits=self.mv_limits,
                    mv_rate_limits=self.mv_rate_limits,
                    cv_weights=self.cv_weights
                )
                
                # 更新控制量
                new_mv = self.current_mv + mv_delta
                new_mv = np.clip(new_mv, self.mv_limits[0], self.mv_limits[1])
            
            # 重新预测未来输出（使用新的控制量）
            cv_prediction = self.mpc_core.predict(
                current_state=self.current_state,
                mv_sequence=[new_mv] * self.mpc_core.Nc
            )
            
            # 扰动补偿
            cv_prediction = cv_prediction - disturbance
            
            # 鲁棒调整
            cv_prediction = self.robust_controller.robust_adjustment(
                cv_prediction, self.cv_setpoint, self.cv_safety_range
            )
            
            # 模型修正
            corrected_params = self.robust_controller.model_correction(self.model_params)
            if corrected_params != self.model_params:
                # 如果模型参数有变化，更新MPC核心
                self.model_params = corrected_params
                self.mpc_core = MPCore(corrected_params)
            
            # 更新内部状态
            self.current_mv = new_mv
            self.control_status = 'running'
            
            return {
                'control_output': self.current_mv,
                'control_status': self.control_status,
                'cv_prediction': cv_prediction.tolist(),
                'mv_sequence': [new_mv] * self.mpc_core.Nc,
                'mv_delta': mv_delta,
                'cv_measured': data['cv_measured'],
                'disturbance_estimate': disturbance
            }
            
        except Exception as e:
            self.control_status = f'error: {str(e)}'
            return {
                'control_output': self.current_mv,
                'control_status': self.control_status,
                'cv_prediction': [],
                'mv_sequence': [],
                'error': str(e)
            }
    
    def set_setpoint(self, new_setpoint):
        """
        设置新的设定值
        
        Args:
            new_setpoint: 新的设定值
        """
        # 检查设定值是否在安全范围内
        if self.cv_safety_range[0] <= new_setpoint <= self.cv_safety_range[1]:
            self.cv_setpoint = new_setpoint
            return True
        return False
    
    def enable_control(self):
        """启用控制"""
        self.control_enabled = True
    
    def disable_control(self):
        """禁用控制"""
        self.control_enabled = False


class DTMpcRobustController:
    """鲁棒控制和扰动抑制模块"""
    
    def __init__(self, robust_params):
        """
        初始化鲁棒控制器
        
        Args:
            robust_params: 鲁棒控制参数字典
        """
        self.robust_enabled = robust_params.get('robust_control_switch', True)
        self.uncertainty_range = robust_params.get('uncertainty_range', 0.2)  # 模型不确定性范围
        self.gain_adjustment = robust_params.get('gain_adjustment', 0.1)  # 增益调整系数
        self.lag_adjustment = robust_params.get('lag_adjustment', 0.1)  # 滞后调整系数
        self.disturbance_estimation_enabled = robust_params.get('disturbance_estimation', True)  # 是否启用扰动估计
        
        # 扰动估计状态
        self.disturbance_estimate = 0.0
        self.disturbance_gain = robust_params.get('disturbance_gain', 0.1)  # 扰动估计增益
        
        # 历史预测误差（用于模型修正）
        self.prediction_errors = []
        self.error_window = robust_params.get('error_window', 10)  # 误差窗口大小
    
    def estimate_disturbance(self, current_output, predicted_output):
        """
        估计外部扰动
        
        Args:
            current_output: 当前实际输出
            predicted_output: 预测输出
            
        Returns:
            disturbance: 扰动估计值
        """
        if not self.disturbance_estimation_enabled:
            return 0.0
        
        # 计算预测误差
        prediction_error = current_output - predicted_output
        
        # 更新扰动估计（简单的一阶低通滤波）
        self.disturbance_estimate = (1 - self.disturbance_gain) * self.disturbance_estimate + \
                                   self.disturbance_gain * prediction_error
        
        return self.disturbance_estimate
    
    def model_correction(self, model_params):
        """
        模型修正（基于历史预测误差）
        
        Args:
            model_params: 当前模型参数
            
        Returns:
            corrected_params: 修正后的模型参数
        """
        if not self.robust_enabled or len(self.prediction_errors) < self.error_window:
            return model_params
        
        corrected_params = model_params.copy()
        
        # 计算平均预测误差
        avg_error = np.mean(self.prediction_errors)
        
        # 根据误差调整模型增益
        if avg_error > 0:
            # 实际输出大于预测输出，增加模型增益
            corrected_params['system_gain'] *= (1 + self.gain_adjustment)
        else:
            # 实际输出小于预测输出，减小模型增益
            corrected_params['system_gain'] *= (1 - self.gain_adjustment)
        
        # 限制增益调整范围
        corrected_params['system_gain'] = max(0.1, min(corrected_params['system_gain'], 10.0))
        
        return corrected_params
    
    def robust_adjustment(self, prediction, cv_setpoint, cv_safety_range):
        """
        鲁棒控制调整
        
        Args:
            prediction: 原始预测结果
            cv_setpoint: 设定值
            cv_safety_range: 安全范围
            
        Returns:
            adjusted_prediction: 鲁棒调整后的预测结果
        """
        if not self.robust_enabled:
            return prediction
        
        adjusted_prediction = np.copy(prediction)
        
        # 计算不确定性边界
        uncertainty_margin = np.maximum(0.05 * cv_setpoint, 1.0)  # 至少1.0的安全裕度
        
        # 检查并调整预测值使其在安全范围内
        for i in range(len(adjusted_prediction)):
            # 计算带不确定性的预测值范围
            lower_bound = cv_safety_range[0] + uncertainty_margin
            upper_bound = cv_safety_range[1] - uncertainty_margin
            
            # 调整预测值
            if adjusted_prediction[i] > upper_bound:
                adjusted_prediction[i] = upper_bound
            elif adjusted_prediction[i] < lower_bound:
                adjusted_prediction[i] = lower_bound
        
        return adjusted_prediction
    
    def update_prediction_error(self, actual, predicted):
        """
        更新预测误差历史
        
        Args:
            actual: 实际输出
            predicted: 预测输出
        """
        error = actual - predicted
        self.prediction_errors.append(error)
        
        # 保持误差窗口大小
        if len(self.prediction_errors) > self.error_window:
            self.prediction_errors.pop(0)
    
    def get_uncertainty_bounds(self, prediction):
        """
        获取带不确定性边界的预测
        
        Args:
            prediction: 原始预测结果
            
        Returns:
            lower_bound: 下界预测
            upper_bound: 上界预测
        """
        if not self.robust_enabled:
            return prediction, prediction
        
        # 计算不确定性边界
        uncertainty = self.uncertainty_range * np.abs(prediction)
        
        return prediction - uncertainty, prediction + uncertainty


class DTMpcParameterManager:
    """DT-MPC参数配置管理器"""
    
    def __init__(self):
        """初始化参数管理器"""
        self.default_controller_params = {
            'control_switch': True,
            'startup_mode': 'cold',
            'auto_test_switch': False,
            'robust_control_switch': True
        }
        
        self.default_robust_params = {
            'robust_control_switch': True,
            'uncertainty_range': 0.2,
            'gain_adjustment': 0.1,
            'lag_adjustment': 0.1,
            'disturbance_estimation': True,
            'disturbance_gain': 0.1,
            'error_window': 10
        }
        
        self.default_mv_params = {
            'operation_range': [-100, 100],
            'rate_limits': [-10, 10],
            'action_cycle': 1.0
        }
        
        self.default_cv_params = {
            'setpoint': 0.0,
            'safety_range': [-200, 200],
            'weights': 1.0
        }
        
        self.default_model_params = {
            'prediction_horizon': 20,
            'control_horizon': 10,
            'system_gain': 1.0,
            'time_delay': 1,
            'time_constant': 5
        }
    
    def validate_parameters(self, params):
        """
        验证参数有效性
        
        Args:
            params: 包含所有参数的字典
            
        Returns:
            is_valid: 参数是否有效
            errors: 错误信息列表
        """
        errors = []
        
        # 验证控制器参数
        if 'controller_params' in params:
            controller_params = params['controller_params']
            if 'control_switch' in controller_params and not isinstance(controller_params['control_switch'], bool):
                errors.append('control_switch must be a boolean')
        
        # 验证鲁棒控制参数
        if 'robust_params' in params:
            robust_params = params['robust_params']
            if 'uncertainty_range' in robust_params:
                if not isinstance(robust_params['uncertainty_range'], (int, float)) or robust_params['uncertainty_range'] < 0:
                    errors.append('uncertainty_range must be a non-negative number')
        
        # 验证MV参数
        mv_params = params.get('mv_params', {})
        if 'operation_range' in mv_params:
            if len(mv_params['operation_range']) != 2:
                errors.append('mv operation_range must have exactly 2 elements')
            elif mv_params['operation_range'][0] >= mv_params['operation_range'][1]:
                errors.append('mv operation_range min must be less than max')
        
        # 验证CV参数
        cv_params = params.get('cv_params', {})
        if 'safety_range' in cv_params:
            if len(cv_params['safety_range']) != 2:
                errors.append('cv safety_range must have exactly 2 elements')
            elif cv_params['safety_range'][0] >= cv_params['safety_range'][1]:
                errors.append('cv safety_range min must be less than max')
        
        # 验证模型参数
        model_params = params.get('model_params', {})
        if 'prediction_horizon' in model_params:
            if not isinstance(model_params['prediction_horizon'], int) or model_params['prediction_horizon'] <= 0:
                errors.append('prediction_horizon must be a positive integer')
        
        return len(errors) == 0, errors
    
    def get_complete_params(self, user_params):
        """
        获取完整的参数配置（合并默认参数和用户参数）
        
        Args:
            user_params: 用户提供的参数
            
        Returns:
            params: 完整的参数配置
        """
        params = {
            'controller_params': self.default_controller_params.copy(),
            'robust_params': self.default_robust_params.copy(),
            'mv_params': self.default_mv_params.copy(),
            'cv_params': self.default_cv_params.copy(),
            'model_params': self.default_model_params.copy()
        }
        
        # 更新用户提供的参数
        for param_type, default_params in params.items():
            if param_type in user_params:
                default_params.update(user_params[param_type])
        
        return params


# 示例用法
if __name__ == "__main__":
    # 初始化参数管理器
    param_manager = DTMpcParameterManager()
    
    # 配置参数
    user_params = {
        'controller_params': {
            'control_switch': True,
            'robust_control_switch': True
        },
        'cv_params': {
            'setpoint': 10.0,
            'safety_range': [-50, 50]
        },
        'model_params': {
            'prediction_horizon': 20,
            'system_gain': 2.0,
            'time_constant': 10
        },
        'robust_params': {
            'uncertainty_range': 0.1,
            'disturbance_gain': 0.05,
            'error_window': 5
        }
    }
    
    # 验证并获取完整参数
    is_valid, errors = param_manager.validate_parameters(user_params)
    if not is_valid:
        print("参数错误:", errors)
        exit(1)
    
    params = param_manager.get_complete_params(user_params)
    
    # 创建DT-MPC控制器
    controller = DTMpcController(
        params['controller_params'],
        params['mv_params'],
        params['cv_params'],
        params['model_params'],
        params['robust_params']
    )
    
    # 运行控制循环
    print("开始DT-MPC控制模拟...")
    for i in range(50):
        result = controller.step()
        print(f"\n控制周期 {i+1}:")
        print(f"  控制状态: {result['control_status']}")
        print(f"  控制输出: {result['control_output']:.2f}")
        print(f"  CV测量值: {result['cv_measured']:.2f}")
        print(f"  CV预测值: [{', '.join([f'{v:.2f}' for v in result['cv_prediction'][:5]])}, ...]")
        print(f"  控制增量: {result['mv_delta']:.2f}")
        
        # 模拟系统响应
        # 在实际应用中，这里会将控制输出发送到执行器，并等待下一个采样周期
        import time
        time.sleep(0.1)