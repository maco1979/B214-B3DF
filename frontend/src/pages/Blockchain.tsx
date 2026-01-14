import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '@/config';
import { apiClient } from '@/services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Link2,
  CheckCircle,
  Clock,
  Users,
  Database,
  TrendingUp,
  Shield,
  FileText,
  RefreshCw,
  AlertCircle,
  Check,
  X,
} from 'lucide-react';

interface BlockchainStatus {
  status: string
  initialized: boolean
  latest_block?: any
  timestamp: string
}

interface Transaction {
  hash: string
  type: string
  status: string
  time: string
}

interface SmartContract {
  name: string
  address: string
  status: string
  calls: string
}

export function Blockchain() {
  const [blockchainStatus, setBlockchainStatus] = useState<BlockchainStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [modelId, setModelId] = useState('');
  const [verificationResult, setVerificationResult] = useState<any>(null);
  const [transactionHistory, setTransactionHistory] = useState<any[]>([]);

  const [blockchainStats, setBlockchainStats] = useState([
    { label: '总区块数', value: '加载中...', change: '', icon: Database },
    { label: '活跃节点', value: '加载中...', change: '', icon: Users },
    { label: '交易总量', value: '加载中...', change: '', icon: FileText },
    { label: '网络延迟', value: '加载中...', change: '', icon: Clock },
  ]);

  const [recentTransactions, setRecentTransactions] = useState([
    { hash: '加载中...', type: '加载中...', status: '加载中...', time: '加载中...' },
  ]);

  const [smartContracts, setSmartContracts] = useState([
    { name: '加载中...', address: '加载中...', status: '加载中...', calls: '加载中...' },
  ]);

  const [networkInfo, setNetworkInfo] = useState({
    consensus: '未知',
    blockSize: '未知',
    blockInterval: '未知',
    networkLatency: '未知',
    dataSync: '未知',
    security: '未知',
  });

  useEffect(() => {
    fetchBlockchainStatus();
    fetchBlockchainStats();
  }, []);

  const fetchBlockchainStats = async () => {
    try {
      setLoading(true);

      // 获取区块链状态数据
      const statusResponse = await apiClient.getBlockchainStatus();
      if (statusResponse.success && statusResponse.data) {
        const stats = [
          {
            label: '总区块数',
            value: statusResponse.data.latest_block?.block_number ? statusResponse.data.latest_block.block_number.toString() : 'N/A',
            change: '',
            icon: Database,
          },
          {
            label: '活跃节点',
            value: '42', // 这里应该从API获取实际节点数
            change: '',
            icon: Users,
          },
          {
            label: '交易总量',
            value: statusResponse.data.latest_block?.transaction_count ? statusResponse.data.latest_block.transaction_count.toString() : 'N/A',
            change: '',
            icon: FileText,
          },
          {
            label: '网络延迟',
            value: '245ms', // 这里应该从API获取实际延迟
            change: '',
            icon: Clock,
          },
        ];
        setBlockchainStats(stats);
      }

      /*
       * 获取最近交易
       * 这里应该从API获取实际交易数据
       */
      setRecentTransactions([
        { hash: '0x1a2b3c...', type: '模型验证', status: '已确认', time: '5分钟前' },
        { hash: '0x4d5e6f...', type: '数据上传', status: '已确认', time: '15分钟前' },
        { hash: '0x7g8h9i...', type: '训练任务', status: '处理中', time: '30分钟前' },
      ]);

      /*
       * 获取智能合约信息
       * 这里应该从API获取实际合约数据
       */
      setSmartContracts([
        { name: '模型验证合约', address: '0xabc123...', status: '活跃', calls: '1,234' },
        { name: '数据溯源合约', address: '0xdef456...', status: '活跃', calls: '892' },
        { name: '权限管理合约', address: '0xghi789...', status: '活跃', calls: '567' },
      ]);

      /*
       * 获取网络信息
       * 这里应该从API获取实际网络信息
       */
      setNetworkInfo({
        consensus: 'PBFT',
        blockSize: '2MB',
        blockInterval: '2秒',
        networkLatency: '低',
        dataSync: '完成',
        security: '高',
      });
    } catch (error) {
      console.error('获取区块链统计失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchBlockchainStatus = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getBlockchainStatus();
      if (response.success) {
        setBlockchainStatus(response.data || null);
      }
    } catch (error) {
      console.error('获取区块链状态失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const verifyModelIntegrity = async () => {
    if (!modelId.trim()) {
 return;
}

    try {
      const response = await apiClient.verifyModelIntegrity(modelId, 'test_hash');
      setVerificationResult(response);
    } catch (error) {
      console.error('验证模型完整性失败:', error);
    }
  };

  const getModelHistory = async () => {
    if (!modelId.trim()) {
 return;
}

    try {
      const response = await apiClient.getModelHistory(modelId);
      if (response.success) {
        setTransactionHistory(response.data?.history || []);
      }
    } catch (error) {
      console.error('获取模型历史失败:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-3xl font-bold gradient-text mb-2">区块链集成</h1>
        <p className="text-gray-300">基于Hyperledger Fabric的AI模型版本控制和数据溯源系统</p>
      </div>

      {/* 区块链统计 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {blockchainStats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index} className="glass-effect hover:neon-glow transition-all duration-300">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  {stat.label}
                </CardTitle>
                <Icon className="w-4 h-4 text-tech-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">{stat.value}</div>
                <p className="text-xs text-green-400">{stat.change}</p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 最近交易 */}
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="w-5 h-5 text-tech-primary" />
              <span>最近交易</span>
            </CardTitle>
            <CardDescription>区块链上的最新交易记录</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {recentTransactions.map((tx, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-tech-dark/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${
                    tx.status === '已确认' ? 'bg-green-500' : 'bg-yellow-500'
                  }`}></div>
                  <div>
                    <div className="font-medium text-white text-sm">{tx.type}</div>
                    <div className="text-xs text-gray-400">{tx.hash}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-xs font-medium ${
                    tx.status === '已确认' ? 'text-green-400' : 'text-yellow-400'
                  }`}>
                    {tx.status}
                  </div>
                  <div className="text-xs text-gray-400">{tx.time}</div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* 智能合约 */}
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="w-5 h-5 text-tech-primary" />
              <span>智能合约</span>
            </CardTitle>
            <CardDescription>部署在区块链上的智能合约</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {smartContracts.map((contract, index) => (
              <div key={index} className="p-3 bg-tech-dark/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-medium text-white">{contract.name}</div>
                  <div className={`text-xs px-2 py-1 rounded ${
                    contract.status === '活跃' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {contract.status}
                  </div>
                </div>
                <div className="text-xs text-gray-400 mb-1">{contract.address}</div>
                <div className="text-xs text-gray-500">调用次数: {contract.calls}</div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* 区块链网络状态 */}
      <Card className="glass-effect">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Link2 className="w-5 h-5 text-tech-primary" />
            <span>网络状态</span>
          </CardTitle>
          <CardDescription>区块链节点和共识机制状态</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">共识算法</span>
                <span className="text-white">{networkInfo.consensus}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">区块大小</span>
                <span className="text-white">{networkInfo.blockSize}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">出块间隔</span>
                <span className="text-white">{networkInfo.blockInterval}</span>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">网络延迟</span>
                <span className="text-green-400">{networkInfo.networkLatency}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">数据同步</span>
                <span className="text-green-400">{networkInfo.dataSync}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">安全性</span>
                <span className="text-green-400">{networkInfo.security}</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 模型验证和查询 */}
      <Card className="glass-effect">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <CheckCircle className="w-5 h-5 text-tech-primary" />
            <span>模型验证与查询</span>
          </CardTitle>
          <CardDescription>验证模型完整性和查询区块链记录</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex space-x-4">
            <Input
              placeholder="输入模型ID"
              value={modelId}
              onChange={e => setModelId(e.target.value)}
              className="flex-1"
            />
            <Button
              variant="tech"
              onClick={verifyModelIntegrity}
              disabled={!modelId.trim() || loading}
              className="flex items-center space-x-2"
            >
              <CheckCircle className="w-4 h-4" />
              <span>验证完整性</span>
            </Button>
            <Button
              variant="outline"
              onClick={getModelHistory}
              disabled={!modelId.trim() || loading}
              className="flex items-center space-x-2"
            >
              <FileText className="w-4 h-4" />
              <span>查询历史</span>
            </Button>
          </div>

          {verificationResult && (
            <div className={`p-3 rounded-lg ${
              verificationResult.success ? 'bg-green-500/20 border border-green-500/30' : 'bg-red-500/20 border border-red-500/30'
            }`}>
              <div className="flex items-center space-x-2">
                {verificationResult.success ?
(
                  <Check className="w-4 h-4 text-green-400" />
                ) :
(
                  <X className="w-4 h-4 text-red-400" />
                )}
                <span className={verificationResult.success ? 'text-green-400' : 'text-red-400'}>
                  {verificationResult.success ? '模型完整性验证通过' : `验证失败: ${verificationResult.error}`}
                </span>
              </div>
            </div>
          )}

          {transactionHistory.length > 0 && (
            <div className="space-y-2">
              <h4 className="font-medium text-white">交易历史</h4>
              {transactionHistory.map((tx, index) => (
                <div key={index} className="p-2 bg-tech-dark/30 rounded text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-300">交易ID: {tx.transaction_id}</span>
                    <span className="text-gray-400">{tx.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 操作按钮 */}
      <div className="flex flex-wrap gap-4">
        <Button
          variant="tech"
          onClick={fetchBlockchainStatus}
          disabled={loading}
          className="flex items-center space-x-2"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>刷新状态</span>
        </Button>
        <Button variant="outline" className="flex items-center space-x-2">
          <TrendingUp className="w-4 h-4" />
          <span>查看统计</span>
        </Button>
        <Button variant="outline" className="flex items-center space-x-2">
          <Shield className="w-4 h-4" />
          <span>安全设置</span>
        </Button>
      </div>

      {/* 系统状态指示器 */}
      {blockchainStatus && (
        <div className={`fixed top-4 right-4 px-3 py-1 rounded-full text-xs font-medium ${
          blockchainStatus.status === 'healthy' ?
            'bg-green-500/20 text-green-400 border border-green-500/30' :
            'bg-red-500/20 text-red-400 border border-red-500/30'
        }`}>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              blockchainStatus.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'
            }`}></div>
            <span>区块链: {blockchainStatus.status === 'healthy' ? '运行中' : '异常'}</span>
          </div>
        </div>
      )}
    </div>
  );
}
