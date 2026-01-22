import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))
from dt_mpc_core import DTMpcController, DTMpcParameterManager, MPCore
from jepa_core import JEPA, JEPATrainer


class JepaDtmpcController(DTMpcController):
    """
    JEPA增强的DT-MPC控制器实现
    
    结合JEPA的抽象嵌入预测能力，增强DT-MPC的预测准确性和鲁棒性
    """
    
    def __init__(self, controller_params, mv_params, cv_params, model_params, robust_params=None, data_config=None, jepa_params=None):
        """
        初始化JEPA-DT-MPC控制器
        
        Args:
            controller_params: DT-MPC控制器参数
            mv_params: 操控变量参数
            cv_params: 被控变量参数
            model_params: 模型参数
            robust_params: 鲁棒控制参数
            data_config: 数据配置参数
            jepa_params: JEPA模型参数
        """
        # 初始化DT-MPC控制器
        super().__init__(controller_params, mv_params, cv_params, model_params, robust_params, data_config)
        
        # JEPA参数
        jepa_params = jepa_params or {}
        self.jepa_enabled = jepa_params.get('enabled', True)
        self.jepa_embedding_dim = jepa_params.get('embedding_dim', 10)
        self.jepa_input_dim = jepa_params.get('input_dim', 3)  # MV, CV, DV作为输入
        self.jepa_output_dim = jepa_params.get('output_dim', 1)  # CV作为输出
        self.jepa_prediction_horizon = jepa_params.get('prediction_horizon', self.mpc_core.Np)
        
        # 初始化JEPA模型
        self._init_jepa_model()
        
        # JEPA训练状态
        self.jepa_trained = jepa_params.get('pretrained', False)
        self.jepa_training_steps = jepa_params.get('training_steps', 100)
        self.jepa_training_data = []
        self.jepa_current_training_step = 0
        
        # JEPA预测缓存
        self.jepa_last_prediction = None
        self.jepa_last_energy = None
    
    def _init_jepa_model(self):
        """
        初始化JEPA模型和训练器
        """
        if not self.jepa_enabled:
            return
        
        # 创建JEPA模型
        self.jepa_model = JEPA(
            input_dim=self.jepa_input_dim,
            embedding_dim=self.jepa_embedding_dim,
            output_dim=self.jepa_output_dim,
            prediction_horizon=self.jepa_prediction_horizon
        )
        
        # 创建训练器
        self.jepa_trainer = JEPATrainer(self.jepa_model, learning_rate=0.001)
    
    def _prepare_jepa_input(self, data):
        """
        准备JEPA输入数据
        
        Args:
            data: DT-MPC数据字典
            
        Returns:
            jepa_input: JEPA输入向量
        """
        # 使用当前MV、CV、DV作为JEPA输入
        jepa_input = np.array([
            data['mv_measured'],
            data['cv_measured'],
            data['dv_measured']
        ])
        
        return jepa_input
    
    def _train_jepa_step(self):
        """
        执行JEPA单步训练
        
        Returns:
            loss: 训练损失
        """
        if not self.jepa_enabled or self.jepa_trained:
            return None
        
        # 检查是否有足够的训练数据
        if len(self.jepa_training_data) < self.jepa_prediction_horizon + 1:
            return None
        
        # 准备训练数据
        batch_size = len(self.jepa_training_data) - self.jepa_prediction_horizon
        if batch_size < 1:
            return None
        
        # 随机选择一个训练样本
        idx = np.random.randint(0, batch_size)
        current_input = self.jepa_training_data[idx]
        
        # 获取未来输入数据
        future_inputs = np.array(self.jepa_training_data[idx+1:idx+1+self.jepa_prediction_horizon])
        future_inputs = future_inputs.reshape(1, self.jepa_prediction_horizon, self.jepa_input_dim)
        
        # 执行训练步骤
        loss = self.jepa_trainer.train_step(
            np.array([current_input]),  # 当前输入 (batch_size=1)
            future_inputs               # 未来输入 (batch_size=1)
        )
        
        # 更新训练状态
        self.jepa_current_training_step += 1
        if self.jepa_current_training_step >= self.jepa_training_steps:
            self.jepa_trained = True
            print("JEPA模型训练完成！")
        
        return loss
    
    def _jepa_predict(self, current_input):
        """
        使用JEPA进行预测
        
        Args:
            current_input: 当前输入向量
            
        Returns:
            jepa_prediction: JEPA预测结果
            energy: 能量值
        """
        if not self.jepa_enabled:
            return None, None
        
        # 使用JEPA进行预测
        with torch.no_grad():
            # 编码当前输入
            current_embedding = self.jepa_model.encode(torch.tensor(current_input.reshape(1, -1), dtype=torch.float32))
            
            # 预测未来嵌入
            predicted_embeddings = self.jepa_model.predict_embedding(current_embedding)
            
            # 解码为输出预测
            jepa_prediction = self.jepa_model.decode(predicted_embeddings)
            jepa_prediction = jepa_prediction.cpu().numpy().squeeze()
            
            # 计算能量值（使用预测嵌入和当前嵌入的扩展）
            # 由于没有未来的实际嵌入，这里使用当前嵌入的扩展作为参考
            reference_embeddings = current_embedding.repeat(1, self.jepa_prediction_horizon, 1)
            energy = self.jepa_model.compute_energy(predicted_embeddings, reference_embeddings)
            energy = energy.cpu().numpy().mean()
        
        return jepa_prediction, energy
    
    def data_acquisition(self, data_source=None):
        """
        数据采集（扩展父类方法以支持JEPA训练）
        
        Args:
            data_source: 数据源
            
        Returns:
            data: 处理后的数据
        """
        # 调用父类的数据采集方法
        data = super().data_acquisition(data_source)
        
        # 为JEPA准备训练数据
        if self.jepa_enabled and not self.jepa_trained:
            jepa_input = self._prepare_jepa_input(data)
            self.jepa_training_data.append(jepa_input)
            
            # 保持训练数据窗口大小
            max_training_data = self.jepa_training_steps * 2
            if len(self.jepa_training_data) > max_training_data:
                self.jepa_training_data.pop(0)
        
        return data
    
    def step(self):
        """
        执行单步控制计算（扩展父类方法以集成JEPA）
        
        Returns:
            result: 控制结果字典
        """
        if not self.control_enabled:
            return super().step()
        
        try:
            # 数据采集
            data = self.data_acquisition()
            
            # 更新状态
            self.current_state = np.array([[data['cv_measured']]])
            
            # 使用JEPA进行预测
            jepa_prediction = None
            jepa_energy = None
            if self.jepa_enabled and self.jepa_trained:
                jepa_input = self._prepare_jepa_input(data)
                jepa_prediction, jepa_energy = self._jepa_predict(jepa_input)
                
                # 缓存JEPA预测结果
                self.jepa_last_prediction = jepa_prediction
                self.jepa_last_energy = jepa_energy
            
            # 执行JEPA训练（如果需要）
            if self.jepa_enabled and not self.jepa_trained:
                loss = self._train_jepa_step()
                if loss is not None and self.jepa_current_training_step % 10 == 0:
                    print(f"JEPA训练进度: {self.jepa_current_training_step}/{self.jepa_training_steps}, 损失: {loss:.4f}")
            
            # 使用DT-MPC进行预测
            cv_prediction = self.mpc_core.predict(
                current_state=self.current_state,
                mv_sequence=[self.current_mv] * self.mpc_core.Nc
            )
            
            # 融合JEPA预测和DT-MPC预测
            if jepa_prediction is not None:
                # 使用JEPA能量值作为权重（能量越低，预测越可靠）
                jepa_weight = np.exp(-jepa_energy) if jepa_energy is not None else 0.5
                
                # 调整JEPA预测到与DT-MPC相同的长度
                if len(jepa_prediction) > len(cv_prediction):
                    jepa_prediction = jepa_prediction[:len(cv_prediction)]
                elif len(jepa_prediction) < len(cv_prediction):
                    # 重复最后一个预测值
                    jepa_prediction = np.pad(jepa_prediction, (0, len(cv_prediction) - len(jepa_prediction)), 'edge')
                
                # 融合预测
                cv_prediction = (1 - jepa_weight) * cv_prediction + jepa_weight * jepa_prediction
            
            # 估计扰动（使用融合后的预测）
            disturbance = self.robust_controller.estimate_disturbance(
                data['cv_measured'], cv_prediction[0]
            )
            
            # 更新预测误差（使用融合后的预测）
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
                    'timestamp': data['timestamp'],
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
            
            # 使用JEPA再次融合预测
            if jepa_prediction is not None:
                jepa_weight = np.exp(-jepa_energy) if jepa_energy is not None else 0.5
                cv_prediction = (1 - jepa_weight) * cv_prediction + jepa_weight * jepa_prediction
            
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
            
            # 构建结果
            result = {
                'control_output': self.current_mv,
                'control_status': self.control_status,
                'cv_prediction': cv_prediction.tolist(),
                'mv_sequence': [new_mv] * self.mpc_core.Nc,
                'mv_delta': mv_delta,
                'cv_measured': data['cv_measured'],
                'disturbance_estimate': disturbance
            }
            
            # 添加JEPA相关结果
            if jepa_prediction is not None:
                result['jepa_prediction'] = jepa_prediction.tolist()
                result['jepa_energy'] = jepa_energy
                result['jepa_weight'] = np.exp(-jepa_energy) if jepa_energy is not None else 0.5
            
            return result
            
        except Exception as e:
            self.control_status = f'error: {str(e)}'
            return {
                'control_output': self.current_mv,
                'control_status': self.control_status,
                'cv_prediction': [],
                'mv_sequence': [],
                'error': str(e)
            }
    
    def get_jepa_status(self):
        """
        获取JEPA模型状态
        
        Returns:
            status: JEPA状态字典
        """
        if not self.jepa_enabled:
            return {
                'enabled': False,
                'status': 'disabled'
            }
        
        return {
            'enabled': True,
            'status': 'trained' if self.jepa_trained else 'training',
            'training_step': self.jepa_current_training_step,
            'training_steps': self.jepa_training_steps,
            'embedding_dim': self.jepa_embedding_dim,
            'prediction_horizon': self.jepa_prediction_horizon
        }


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
    
    # JEPA参数
    jepa_params = {
        'enabled': True,
        'embedding_dim': 10,
        'input_dim': 3,
        'output_dim': 1,
        'prediction_horizon': 20,
        'training_steps': 50
    }
    
    # 创建JEPA-DT-MPC控制器
    controller = JepaDtmpcController(
        params['controller_params'],
        params['mv_params'],
        params['cv_params'],
        params['model_params'],
        params['robust_params'],
        jepa_params=jepa_params
    )
    
    # 运行控制循环
    print("开始JEPA-DT-MPC控制模拟...")
    for i in range(100):
        result = controller.step()
        print(f"\n控制周期 {i+1}:")
        print(f"  控制状态: {result['control_status']}")
        print(f"  控制输出: {result['control_output']:.2f}")
        print(f"  CV测量值: {result['cv_measured']:.2f}")
        print(f"  CV预测值: [{', '.join([f'{v:.2f}' for v in result['cv_prediction'][:5]])}, ...]")
        print(f"  控制增量: {result['mv_delta']:.2f}")
        
        # 打印JEPA状态
        if 'jepa_prediction' in result:
            print(f"  JEPA能量: {result['jepa_energy']:.4f}")
            print(f"  JEPA权重: {result['jepa_weight']:.2f}")
        
        # 模拟系统响应
        import time
        time.sleep(0.1)
