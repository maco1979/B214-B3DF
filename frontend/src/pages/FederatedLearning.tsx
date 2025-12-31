import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { 
  Users, 
  Cpu, 
  Database, 
  Shield, 
  TrendingUp,
  Play,
  Pause,
  RefreshCw,
  Plus,
  Settings,
  BarChart3,
  Clock
} from 'lucide-react'
import { apiClient } from '@/services/api'

interface FederatedClient {
  client_id: string
  info: any
  registered_at: string
  status: string
}

interface TrainingRound {
  round_id: string
  start_time: string
  status: string
  participants: string[]
  participants_count?: number
}

interface PrivacyStatus {
  enabled: boolean
  epsilon: number
  delta: number
  privacy_spent: any
  rounds_with_privacy: number
}

export function FederatedLearning() {
  const [clients, setClients] = useState<FederatedClient[]>([])
  const [rounds, setRounds] = useState<TrainingRound[]>([])
  const [privacyStatus, setPrivacyStatus] = useState<PrivacyStatus | null>(null)
  const [serverStatus, setServerStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [newClientId, setNewClientId] = useState('')
  const [roundConfig, setRoundConfig] = useState({
    client_fraction: 0.5,
    learning_rate: 0.01,
    epochs: 10
  })

  const fetchFederatedData = async () => {
    try {
      setLoading(true)
      
      const [clientsRes, roundsRes, privacyRes, statusRes] = await Promise.all([
        apiClient.getFederatedClients(),
        apiClient.getFederatedRounds(),
        apiClient.getFederatedPrivacyStatus(),
        apiClient.getFederatedStatus()
      ])

      if (clientsRes.success) setClients(Array.isArray(clientsRes.data) ? clientsRes.data : [])
      if (roundsRes.success) setRounds(Array.isArray(roundsRes.data) ? roundsRes.data : [])
      if (privacyRes.success) setPrivacyStatus(privacyRes.data)
      if (statusRes.success) setServerStatus(statusRes.data)
    } catch (error) {
      console.error('获取联邦学习数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchFederatedData()
  }, [])

  const registerClient = async () => {
    if (!newClientId.trim()) return

    try {
      const response = await apiClient.registerFederatedClient({
        client_id: newClientId,
        device_type: 'edge_device',
        capabilities: ['training', 'inference']
      })

      if (response.success) {
        setNewClientId('')
        fetchFederatedData()
      }
    } catch (error) {
      console.error('注册客户端失败:', error)
    }
  }

  const startTrainingRound = async () => {
    try {
      const response = await apiClient.startFederatedRound(roundConfig)
      
      if (response.success) {
        fetchFederatedData()
      }
    } catch (error) {
      console.error('启动训练轮次失败:', error)
    }
  }

  const aggregateRound = async (roundId: string) => {
    try {
      const response = await apiClient.aggregateFederatedRound(roundId)
      
      if (response.success) {
        fetchFederatedData()
      }
    } catch (error) {
      console.error('聚合轮次失败:', error)
    }
  }

  const updatePrivacyConfig = async (config: any) => {
    try {
      const response = await apiClient.updateFederatedPrivacyConfig(config)
      
      if (response.success) {
        fetchFederatedData()
      }
    } catch (error) {
      console.error('更新隐私配置失败:', error)
    }
  }

  const stats = [
    {
      label: '注册客户端',
      value: serverStatus?.total_clients || 0,
      icon: Users,
      color: 'text-blue-400',
      change: `活跃: ${serverStatus?.active_clients || 0}`
    },
    {
      label: '完成轮次',
      value: serverStatus?.rounds_completed || 0,
      icon: TrendingUp,
      color: 'text-green-400',
      change: '持续训练中'
    },
    {
      label: '隐私保护',
      value: privacyStatus?.enabled ? '启用' : '禁用',
      icon: Shield,
      color: 'text-purple-400',
      change: `ε=${privacyStatus?.epsilon || 0}`
    },
    {
      label: '当前轮次',
      value: serverStatus?.current_round ? '进行中' : '未开始',
      icon: Cpu,
      color: 'text-yellow-400',
      change: serverStatus?.current_round?.participants_count ? `${serverStatus.current_round.participants_count}客户端` : '--'
    }
  ]

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">联邦学习系统</h1>
          <p className="text-gray-300">分布式AI训练，保护数据隐私，实现协同学习</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button 
            variant="outline" 
            onClick={fetchFederatedData}
            disabled={loading}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>刷新</span>
          </Button>
          <Button variant="tech" className="flex items-center space-x-2">
            <Settings className="w-4 h-4" />
            <span>系统设置</span>
          </Button>
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon
          return (
            <Card key={index} className="glass-effect hover:neon-glow transition-all duration-300">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  {stat.label}
                </CardTitle>
                <Icon className={`w-4 h-4 ${stat.color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">{stat.value}</div>
                <p className="text-xs text-gray-400 mt-1">{stat.change}</p>
              </CardContent>
            </Card>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 客户端管理 */}
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Users className="w-5 h-5 text-tech-primary" />
                <span>客户端管理</span>
              </div>
              <span className="text-sm text-gray-400">{clients.length} 客户端</span>
            </CardTitle>
            <CardDescription>注册和管理联邦学习客户端</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* 注册新客户端 */}
            <div className="flex space-x-2">
              <Input
                placeholder="输入客户端ID"
                value={newClientId}
                onChange={(e) => setNewClientId(e.target.value)}
                className="flex-1"
              />
              <Button 
                variant="tech" 
                onClick={registerClient}
                disabled={!newClientId.trim()}
                className="flex items-center space-x-2"
              >
                <Plus className="w-4 h-4" />
                <span>注册</span>
              </Button>
            </div>

            {/* 客户端列表 */}
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {clients.map((client) => (
                <div key={client.client_id} className="flex items-center justify-between p-2 bg-tech-dark/30 rounded">
                  <div>
                    <div className="font-medium text-white">{client.client_id}</div>
                    <div className="text-xs text-gray-400">
                      注册时间: {new Date(client.registered_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div className={`text-xs px-2 py-1 rounded ${
                    client.status === 'active' 
                      ? 'bg-green-500/20 text-green-400' 
                      : 'bg-gray-500/20 text-gray-400'
                  }`}>
                    {client.status === 'active' ? '活跃' : '离线'}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* 训练轮次控制 */}
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Cpu className="w-5 h-5 text-tech-primary" />
              <span>训练控制</span>
            </CardTitle>
            <CardDescription>配置和启动联邦学习轮次</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="text-sm text-gray-400">客户端比例</label>
                <Input
                  type="number"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={roundConfig.client_fraction}
                  onChange={(e) => setRoundConfig({
                    ...roundConfig,
                    client_fraction: parseFloat(e.target.value)
                  })}
                />
              </div>
              <div>
                <label className="text-sm text-gray-400">学习率</label>
                <Input
                  type="number"
                  min="0.001"
                  max="0.1"
                  step="0.001"
                  value={roundConfig.learning_rate}
                  onChange={(e) => setRoundConfig({
                    ...roundConfig,
                    learning_rate: parseFloat(e.target.value)
                  })}
                />
              </div>
              <div>
                <label className="text-sm text-gray-400">训练轮数</label>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={roundConfig.epochs}
                  onChange={(e) => setRoundConfig({
                    ...roundConfig,
                    epochs: parseInt(e.target.value)
                  })}
                />
              </div>
            </div>

            <Button 
              variant="tech" 
              onClick={startTrainingRound}
              disabled={clients.length === 0}
              className="w-full flex items-center space-x-2"
            >
              <Play className="w-4 h-4" />
              <span>启动训练轮次</span>
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* 训练历史 */}
      <Card className="glass-effect">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-tech-primary" />
            <span>训练历史</span>
          </CardTitle>
          <CardDescription>联邦学习训练轮次记录</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {rounds.slice().reverse().map((round, index) => (
              <div key={`${round.round_id}_${index}`} className="flex items-center justify-between p-3 bg-tech-dark/30 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    round.status === 'completed' ? 'bg-green-400' : 
                    round.status === 'active' ? 'bg-yellow-400' : 'bg-gray-400'
                  }`}></div>
                  <div>
                    <div className="font-medium text-white">{round.round_id}</div>
                    <div className="text-sm text-gray-400">
                      开始时间: {new Date(round.start_time).toLocaleString()}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-400">
                    {round.participants_count || (round.participants && round.participants.length) || 0} 客户端参与
                  </span>
                  {round.status === 'active' && (
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => aggregateRound(round.round_id)}
                    >
                      聚合
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 隐私保护配置 */}
      {privacyStatus && (
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="w-5 h-5 text-purple-400" />
              <span>差分隐私保护</span>
            </CardTitle>
            <CardDescription>配置隐私保护参数，平衡隐私和模型性能</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <label className="text-sm text-gray-400">隐私预算 (ε)</label>
                <Input
                  type="number"
                  min="0.1"
                  max="10.0"
                  step="0.1"
                  value={privacyStatus.epsilon}
                  onChange={(e) => updatePrivacyConfig({
                    epsilon: parseFloat(e.target.value)
                  })}
                />
                <div className="text-xs text-gray-400">值越小，隐私保护越强</div>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm text-gray-400">失败概率 (δ)</label>
                <Input
                  type="number"
                  min="1e-6"
                  max="1e-3"
                  step="1e-6"
                  value={privacyStatus.delta}
                  onChange={(e) => updatePrivacyConfig({
                    delta: parseFloat(e.target.value)
                  })}
                />
                <div className="text-xs text-gray-400">通常设置为很小的值</div>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm text-gray-400">隐私保护</label>
                <Button
                  variant={privacyStatus.enabled ? "tech" : "outline"}
                  onClick={() => updatePrivacyConfig({
                    enabled: !privacyStatus.enabled
                  })}
                  className="w-full"
                >
                  {privacyStatus.enabled ? '已启用' : '已禁用'}
                </Button>
                <div className="text-xs text-gray-400">
                  已消耗隐私: {privacyStatus.privacy_spent?.epsilon_spent?.toFixed(3) || '0.000'}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}