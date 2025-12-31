import React, { useState } from 'react'
import { API_BASE_URL } from '@/config'
import { apiClient } from '@/services/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { 
  Settings as SettingsIcon, 
  Save, 
  RefreshCw,
  Shield,
  Database,
  Cpu,
  Link2,
  Bell,
  User
} from 'lucide-react'

// 类型定义
interface SettingOption {
  value: string
  label: string
}

interface Setting {
  label: string
  key: string
  type: 'text' | 'number' | 'checkbox' | 'select'
  placeholder?: string
  min?: number
  max?: number
  options?: SettingOption[]
}

interface SettingSection {
  title: string
  icon: React.ElementType
  description: string
  settings: Setting[]
}

interface SettingsState {
  apiEndpoint: string
  maxConcurrentJobs: number
  autoSaveInterval: number
  notificationEnabled: boolean
  darkMode: boolean
  language: string
  blockchainNode?: string
}

export function Settings() {
  const [settings, setSettings] = useState<SettingsState>({
    apiEndpoint: API_BASE_URL,
    maxConcurrentJobs: 5,
    autoSaveInterval: 30,
    notificationEnabled: true,
    darkMode: true,
    language: 'zh-CN'
  })

  const handleSettingChange = (key: keyof SettingsState, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }))
  }

  const saveSettings = async () => {
    try {
      const response = await apiClient.saveSettings(settings)
      if (response.success) {
        alert('设置已成功保存')
      } else {
        alert(`保存失败: ${response.error || '未知错误'}`)
      }
    } catch (error) {
      console.error('保存设置失败:', error)
      alert('保存设置时发生错误')
    }
  }

  const settingSections: SettingSection[] = [
    {
      title: 'API设置',
      icon: SettingsIcon,
      description: '配置后端API连接和接口参数',
      settings: [
        {
          label: 'API端点',
          key: 'apiEndpoint',
          type: 'text',
          placeholder: '输入API端点地址'
        },
        {
          label: '最大并发任务数',
          key: 'maxConcurrentJobs',
          type: 'number',
          min: 1,
          max: 20
        }
      ]
    },
    {
      title: 'AI模型设置',
      icon: Cpu,
      description: '配置AI模型训练和推理参数',
      settings: [
        {
          label: '自动保存间隔(分钟)',
          key: 'autoSaveInterval',
          type: 'number',
          min: 5,
          max: 60
        }
      ]
    },
    {
      title: '区块链设置',
      icon: Link2,
      description: '配置区块链网络和智能合约',
      settings: [
        {
          label: '区块链节点地址',
          key: 'blockchainNode',
          type: 'text',
          placeholder: '输入区块链节点地址'
        }
      ]
    },
    {
      title: '通知设置',
      icon: Bell,
      description: '配置系统通知和提醒',
      settings: [
        {
          label: '启用通知',
          key: 'notificationEnabled',
          type: 'checkbox'
        }
      ]
    },
    {
      title: '界面设置',
      icon: User,
      description: '配置用户界面和显示选项',
      settings: [
        {
          label: '深色模式',
          key: 'darkMode',
          type: 'checkbox'
        },
        {
          label: '语言',
          key: 'language',
          type: 'select',
          options: [
            { value: 'zh-CN', label: '简体中文' },
            { value: 'en-US', label: 'English' }
          ]
        }
      ]
    }
  ]

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">系统设置</h1>
          <p className="text-gray-300">配置AI平台的各项参数和功能选项</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="tech" onClick={saveSettings} className="flex items-center space-x-2">
            <Save className="w-4 h-4" />
            <span>保存设置</span>
          </Button>
          <Button variant="outline" className="flex items-center space-x-2">
            <RefreshCw className="w-4 h-4" />
            <span>重置默认</span>
          </Button>
        </div>
      </div>

      {/* 设置卡片 */}
      <div className="space-y-6">
        {settingSections.map((section, index) => {
          const Icon = section.icon
          return (
            <Card key={index} className="glass-effect">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Icon className="w-5 h-5 text-tech-primary" />
                  <span>{section.title}</span>
                </CardTitle>
                <CardDescription>{section.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {section.settings.map((setting) => (
                  <div key={setting.key} className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                    <label className="text-sm font-medium text-gray-300 sm:w-1/3">
                      {setting.label}
                    </label>
                    <div className="sm:w-2/3">
                      {setting.type === 'text' && (
                        <Input
                          type="text"
                          value={settings[setting.key as keyof SettingsState] as string || ''}
                          onChange={(e) => handleSettingChange(setting.key as keyof SettingsState, e.target.value)}
                          placeholder={setting.placeholder}
                          className="w-full"
                        />
                      )}
                      {setting.type === 'number' && (
                        <Input
                          type="number"
                          value={settings[setting.key as keyof SettingsState] as number}
                          onChange={(e) => handleSettingChange(setting.key as keyof SettingsState, parseInt(e.target.value))}
                          min={setting.min}
                          max={setting.max}
                          className="w-full"
                        />
                      )}
                      {setting.type === 'checkbox' && (
                        <input
                          type="checkbox"
                          checked={settings[setting.key as keyof SettingsState] as boolean}
                          onChange={(e) => handleSettingChange(setting.key as keyof SettingsState, e.target.checked)}
                          className="w-4 h-4 text-tech-primary bg-gray-700 border-gray-600 rounded focus:ring-tech-primary focus:ring-offset-gray-800"
                        />
                      )}
                      {setting.type === 'select' && (
                        <select
                          value={settings[setting.key as keyof SettingsState] as string}
                          onChange={(e) => handleSettingChange(setting.key as keyof SettingsState, e.target.value)}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:border-tech-primary"
                        >
                          {setting.options?.map((option: SettingOption) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* 系统信息 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Database className="w-5 h-5 text-tech-primary" />
              <span>系统信息</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">前端版本</span>
              <span className="text-white">v1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">后端版本</span>
              <span className="text-white">v1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">构建时间</span>
              <span className="text-white">2024-01-20</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="w-5 h-5 text-tech-primary" />
              <span>安全状态</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">API加密</span>
              <span className="text-green-400">已启用</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">数据备份</span>
              <span className="text-green-400">自动</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">防火墙</span>
              <span className="text-green-400">运行中</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}