/**
 * PTZ云台摄像头控制组件
 * 支持真实的云台转动、变焦、对焦等物理操作
 */

import React, { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface PTZStatus {
  connected: boolean;
  protocol?: string;
  connection_type?: string;
  position?: {
    pan: number;
    tilt: number;
    zoom: number;
  };
  presets?: Record<number, any>;
}

interface PTZControlProps {
  apiClient: any;
}

export const PTZControl: React.FC<PTZControlProps> = ({ apiClient }) => {
  const [ptzStatus, setPtzStatus] = useState<PTZStatus>({ connected: false });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  
  // 连接配置
  const [protocol, setProtocol] = useState('pelco_d');
  const [connectionType, setConnectionType] = useState('serial');
  const [serialPort, setSerialPort] = useState('/dev/ttyUSB0');
  const [baudrate, setBaudrate] = useState(9600);
  const [networkHost, setNetworkHost] = useState('192.168.1.100');
  const [networkPort, setNetworkPort] = useState(5000);
  const [httpUrl, setHttpUrl] = useState('http://192.168.1.100');
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('admin');
  
  // 控制参数
  const [speed, setSpeed] = useState(50);
  const [presetId, setPresetId] = useState(1);
  const [presetName, setPresetName] = useState('');
  
  // 位置控制
  const [targetPan, setTargetPan] = useState(0);
  const [targetTilt, setTargetTilt] = useState(0);
  const [targetZoom, setTargetZoom] = useState(1.0);

  // 加载PTZ状态
  useEffect(() => {
    checkPTZStatus();
  }, []);

  const checkPTZStatus = async () => {
    try {
      const res = await apiClient.get('/api/camera/ptz/status');
      if (res.success && res.data) {
        setPtzStatus(res.data);
      }
    } catch (error) {
      console.error('获取PTZ状态失败:', error);
    }
  };

  // 连接PTZ
  const connectPTZ = async () => {
    setLoading(true);
    setError('');
    
    try {
      const params: any = {
        protocol,
        connection_type: connectionType
      };
      
      if (connectionType === 'serial') {
        params.port = serialPort;
        params.baudrate = baudrate;
        params.address = 1;
      } else if (connectionType === 'network') {
        params.host = networkHost;
        params.network_port = networkPort;
        params.address = 1;
      } else if (connectionType === 'http') {
        params.base_url = httpUrl;
        params.username = username;
        params.password = password;
      }
      
      const res = await apiClient.post('/api/camera/ptz/connect', params);
      
      if (res.success) {
        setError('');
        await checkPTZStatus();
      } else {
        setError(res.message || '连接失败');
      }
    } catch (err: any) {
      setError(err.message || '连接失败');
    } finally {
      setLoading(false);
    }
  };

  // 断开PTZ
  const disconnectPTZ = async () => {
    setLoading(true);
    try {
      const res = await apiClient.post('/api/camera/ptz/disconnect');
      if (res.success) {
        setPtzStatus({ connected: false });
        setError('');
      }
    } catch (err: any) {
      setError(err.message || '断开失败');
    } finally {
      setLoading(false);
    }
  };

  // 执行PTZ动作
  const executeAction = async (action: string) => {
    if (!ptzStatus.connected) {
      setError('请先连接PTZ云台');
      return;
    }
    
    try {
      const params: any = { action, speed };
      
      if (action === 'preset_set' || action === 'preset_goto') {
        params.preset_id = presetId;
      }
      
      const res = await apiClient.post('/api/camera/ptz/action', params);
      
      if (res.success) {
        setError('');
        await checkPTZStatus();
      } else {
        setError(res.message || '操作失败');
      }
    } catch (err: any) {
      setError(err.message || '操作失败');
    }
  };

  // 移动到位置
  const moveToPosition = async () => {
    if (!ptzStatus.connected) {
      setError('请先连接PTZ云台');
      return;
    }
    
    try {
      const res = await apiClient.post('/api/camera/ptz/move', {
        pan: targetPan,
        tilt: targetTilt,
        zoom: targetZoom,
        speed
      });
      
      if (res.success) {
        setError('');
        await checkPTZStatus();
      } else {
        setError(res.message || '移动失败');
      }
    } catch (err: any) {
      setError(err.message || '移动失败');
    }
  };

  // 设置预置位
  const setPreset = async () => {
    if (!ptzStatus.connected) {
      setError('请先连接PTZ云台');
      return;
    }
    
    try {
      const res = await apiClient.post('/api/camera/ptz/preset/set', {
        preset_id: presetId,
        name: presetName || `预置位${presetId}`
      });
      
      if (res.success) {
        setError('');
        await checkPTZStatus();
      } else {
        setError(res.message || '设置失败');
      }
    } catch (err: any) {
      setError(err.message || '设置失败');
    }
  };

  // 转到预置位
  const gotoPreset = async () => {
    await executeAction('preset_goto');
  };

  return (
    <div className="space-y-4">
      {/* 标题 */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold text-cyber-cyan">PTZ云台控制</h3>
        <div className={cn(
          "px-3 py-1 rounded-lg text-xs font-bold",
          ptzStatus.connected
            ? "bg-green-500/20 text-green-400 border border-green-500/30"
            : "bg-gray-700/50 text-gray-400 border border-gray-600/30"
        )}>
          {ptzStatus.connected ? '已连接' : '未连接'}
        </div>
      </div>

      {/* 连接配置 */}
      {!ptzStatus.connected && (
        <div className="space-y-3 p-4 rounded-xl bg-white/5 border border-white/10">
          <h4 className="text-sm font-bold text-gray-300">连接配置</h4>
          
          {/* 协议选择 */}
          <div>
            <label className="text-xs text-gray-400 block mb-1">控制协议</label>
            <select
              value={protocol}
              onChange={(e) => setProtocol(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
            >
              <option value="pelco_d">Pelco-D（最常用）</option>
              <option value="pelco_p">Pelco-P</option>
              <option value="visca">VISCA（Sony）</option>
              <option value="onvif">ONVIF</option>
              <option value="http">HTTP API</option>
            </select>
          </div>

          {/* 连接类型 */}
          <div>
            <label className="text-xs text-gray-400 block mb-1">连接类型</label>
            <select
              value={connectionType}
              onChange={(e) => setConnectionType(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
            >
              <option value="serial">串口（RS-485/RS-232）</option>
              <option value="network">网络（TCP/IP）</option>
              <option value="http">HTTP接口</option>
            </select>
          </div>

          {/* 串口配置 */}
          {connectionType === 'serial' && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">串口</label>
                <input
                  type="text"
                  value={serialPort}
                  onChange={(e) => setSerialPort(e.target.value)}
                  placeholder="/dev/ttyUSB0 或 COM3"
                  className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">波特率</label>
                <select
                  value={baudrate}
                  onChange={(e) => setBaudrate(Number(e.target.value))}
                  className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
                >
                  <option value={2400}>2400</option>
                  <option value={4800}>4800</option>
                  <option value={9600}>9600</option>
                  <option value={19200}>19200</option>
                  <option value={38400}>38400</option>
                </select>
              </div>
            </>
          )}

          {/* 网络配置 */}
          {connectionType === 'network' && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">IP地址</label>
                <input
                  type="text"
                  value={networkHost}
                  onChange={(e) => setNetworkHost(e.target.value)}
                  placeholder="192.168.1.100"
                  className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">端口</label>
                <input
                  type="number"
                  value={networkPort}
                  onChange={(e) => setNetworkPort(Number(e.target.value))}
                  placeholder="5000"
                  className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
                />
              </div>
            </>
          )}

          {/* HTTP配置 */}
          {connectionType === 'http' && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">URL地址</label>
                <input
                  type="text"
                  value={httpUrl}
                  onChange={(e) => setHttpUrl(e.target.value)}
                  placeholder="http://192.168.1.100"
                  className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-gray-400 block mb-1">用户名</label>
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="admin"
                    className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">密码</label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="admin"
                    className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
                  />
                </div>
              </div>
            </>
          )}

          <button
            onClick={connectPTZ}
            disabled={loading}
            className={cn(
              "w-full px-4 py-2 rounded-lg font-bold text-sm transition-all",
              "bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/30",
              "hover:bg-cyber-cyan/30 hover:border-cyber-cyan/50",
              loading && "opacity-50 cursor-not-allowed"
            )}
          >
            {loading ? '连接中...' : '连接PTZ云台'}
          </button>
        </div>
      )}

      {/* PTZ控制面板 */}
      {ptzStatus.connected && (
        <>
          {/* 当前位置 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <h4 className="text-sm font-bold text-gray-300 mb-3">当前位置</h4>
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center">
                <div className="text-xs text-gray-400">水平角度</div>
                <div className="text-lg font-bold text-cyber-cyan">
                  {ptzStatus.position?.pan.toFixed(1)}°
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400">垂直角度</div>
                <div className="text-lg font-bold text-cyber-cyan">
                  {ptzStatus.position?.tilt.toFixed(1)}°
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400">变焦倍数</div>
                <div className="text-lg font-bold text-cyber-cyan">
                  {ptzStatus.position?.zoom.toFixed(1)}x
                </div>
              </div>
            </div>
          </div>

          {/* 方向控制 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <h4 className="text-sm font-bold text-gray-300 mb-3">方向控制</h4>
            <div className="flex flex-col items-center space-y-2">
              {/* 上 */}
              <button
                onClick={() => executeAction('tilt_up')}
                className="px-6 py-3 rounded-lg bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/30 hover:bg-cyber-cyan/30 font-bold"
              >
                ▲
              </button>
              
              {/* 左右 */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => executeAction('pan_left')}
                  className="px-6 py-3 rounded-lg bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/30 hover:bg-cyber-cyan/30 font-bold"
                >
                  ◄
                </button>
                <button
                  onClick={() => executeAction('stop')}
                  className="px-6 py-3 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 font-bold"
                >
                  停止
                </button>
                <button
                  onClick={() => executeAction('pan_right')}
                  className="px-6 py-3 rounded-lg bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/30 hover:bg-cyber-cyan/30 font-bold"
                >
                  ►
                </button>
              </div>
              
              {/* 下 */}
              <button
                onClick={() => executeAction('tilt_down')}
                className="px-6 py-3 rounded-lg bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/30 hover:bg-cyber-cyan/30 font-bold"
              >
                ▼
              </button>
            </div>
          </div>

          {/* 变焦控制 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <h4 className="text-sm font-bold text-gray-300 mb-3">变焦控制</h4>
            <div className="flex items-center justify-center space-x-2">
              <button
                onClick={() => executeAction('zoom_out')}
                className="px-6 py-2 rounded-lg bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/30 hover:bg-cyber-cyan/30 font-bold"
              >
                拉远 -
              </button>
              <button
                onClick={() => executeAction('zoom_in')}
                className="px-6 py-2 rounded-lg bg-cyber-cyan/20 text-cyber-cyan border border-cyber-cyan/30 hover:bg-cyber-cyan/30 font-bold"
              >
                拉近 +
              </button>
            </div>
          </div>

          {/* 速度控制 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <h4 className="text-sm font-bold text-gray-300 mb-2">速度: {speed}</h4>
            <input
              type="range"
              min="0"
              max="100"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-full"
            />
          </div>

          {/* 预置位控制 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <h4 className="text-sm font-bold text-gray-300 mb-3">预置位控制</h4>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-gray-400 block mb-1">预置位编号</label>
                  <input
                    type="number"
                    value={presetId}
                    onChange={(e) => setPresetId(Number(e.target.value))}
                    min="1"
                    max="256"
                    className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">名称（可选）</label>
                  <input
                    type="text"
                    value={presetName}
                    onChange={(e) => setPresetName(e.target.value)}
                    placeholder="如：大门"
                    className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-white text-sm"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={setPreset}
                  className="px-4 py-2 rounded-lg bg-green-500/20 text-green-400 border border-green-500/30 hover:bg-green-500/30 font-bold text-sm"
                >
                  设置预置位
                </button>
                <button
                  onClick={gotoPreset}
                  className="px-4 py-2 rounded-lg bg-blue-500/20 text-blue-400 border border-blue-500/30 hover:bg-blue-500/30 font-bold text-sm"
                >
                  转到预置位
                </button>
              </div>
            </div>
          </div>

          {/* 断开连接 */}
          <button
            onClick={disconnectPTZ}
            className="w-full px-4 py-2 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 font-bold text-sm"
          >
            断开连接
          </button>
        </>
      )}

      {/* 错误提示 */}
      {error && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30">
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}

      {/* 使用说明 */}
      <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
        <p className="text-xs text-yellow-400 font-bold mb-1">📖 PTZ云台控制说明</p>
        <ul className="text-[10px] text-yellow-300/70 space-y-1">
          <li>• 支持真实的云台物理转动，而非软件模拟</li>
          <li>• 适用于农业监控、无人机控制、智能安防等场景</li>
          <li>• 支持多种协议：Pelco-D/P、VISCA、ONVIF、HTTP</li>
          <li>• 支持预置位：可保存常用位置快速切换</li>
          <li>• Windows系统串口格式：COM3，Linux/Mac格式：/dev/ttyUSB0</li>
        </ul>
      </div>
    </div>
  );
};
