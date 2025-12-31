import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useAuth } from '@/hooks/useAuth'
import { useNavigate } from 'react-router-dom'
import { WeChatLogin, AlipayLogin } from '@/components/auth/QRCodeLogin'
import { Mail, Lock, User, Github, ExternalLink, AlertCircle } from 'lucide-react'
import { apiClient } from '@/services/api'

const LoginPage = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [registrationCode, setRegistrationCode] = useState('')
  const [isQRLogin, setIsQRLogin] = useState(false)
  const [isRegistering, setIsRegistering] = useState(false)
  const [selectedProvider, setSelectedProvider] = useState<'wechat' | 'alipay'>('wechat')
  const [error, setError] = useState<string | null>(null)
  const { login } = useAuth()
  const navigate = useNavigate()

  // 检测移动设备并自动切换到二维码登录
  useEffect(() => {
    const isMobile = () => {
      // 检测屏幕宽度
      const screenWidth = window.innerWidth < 768
      // 检测用户代理
      const userAgent = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
      return screenWidth || userAgent
    }

    if (isMobile()) {
      setIsQRLogin(true)
    }
  }, [])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    console.log('handleLogin called with email:', email, 'password:', password)
    try {
      console.log('Attempting login...')
      await login(email, password)
      console.log('Login successful, navigating to home')
      navigate('/')
    } catch (error) {
      console.error('Login failed:', error)
    }
  }

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault()
    console.log('handleRegister called with code:', registrationCode, 'email:', email, 'password:', password)
    try {
      console.log('Attempting registration...')
      const response = await apiClient.registerWithCode(registrationCode, email, password)
      if (response.success) {
        console.log('Registration successful, logging in...')
        await login(email, password)
        navigate('/')
      }
    } catch (error) {
      console.error('Registration failed:', error)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-4 bg-gradient-to-br from-tech-dark via-tech-gray to-tech-light">
      <Card className="w-full max-w-md bg-tech-dark border-tech-light">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center text-white">赛博有机体</CardTitle>
          <CardDescription className="text-center text-gray-400">
            {isRegistering ? '使用产品注册码注册' : '登录您的账户以继续'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {error && (
            <div className="mb-4 p-3 rounded bg-red-500/10 border border-red-500/20 text-red-500 text-sm flex items-center space-x-2">
              <AlertCircle className="w-4 h-4" />
              <span>{error}</span>
            </div>
          )}
          {!isQRLogin ? (
            <form onSubmit={isRegistering ? handleRegister : handleLogin} className="space-y-4">
              {isRegistering && (
                <div className="space-y-2">
                  <label htmlFor="registrationCode" className="text-sm font-medium text-white">
                    产品注册码
                  </label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                    <Input
                      id="registrationCode"
                      type="text"
                      placeholder="输入您的产品注册码"
                      value={registrationCode}
                      onChange={(e) => setRegistrationCode(e.target.value)}
                      className="pl-10 bg-cyber-black border-tech-light text-white"
                      required
                    />
                  </div>
                </div>
              )}
              <div className="space-y-2">
                <label htmlFor="email" className="text-sm font-medium text-white">
                  邮箱
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="example@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="pl-10 bg-cyber-black border-tech-light text-white"
                    required
                  />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label htmlFor="password" className="text-sm font-medium text-white">
                    密码
                  </label>
                  {!isRegistering && (
                    <a href="#" className="text-sm text-tech-primary hover:text-tech-secondary">
                      忘记密码？
                    </a>
                  )}
                </div>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                  <Input
                    id="password"
                    type="password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10 bg-cyber-black border-tech-light text-white"
                    required
                  />
                </div>
              </div>
              <Button
                type="submit"
                className="w-full bg-gradient-to-r from-tech-primary to-tech-secondary hover:from-tech-primary/90 hover:to-tech-secondary/90 text-white"
              >
                {isRegistering ? '注册' : '登录'}
              </Button>
              
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-tech-light"></div>
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-tech-dark px-2 text-gray-400">或者使用其他方式登录</span>
                </div>
              </div>
              
              <div className="space-y-2">
                <Button
                  variant="ghost"
                  className="w-full border border-tech-light hover:bg-tech-gray text-white"
                  onClick={() => setIsQRLogin(true)}
                >
                  <User className="mr-2 h-4 w-4" />
                  扫码登录
                </Button>
                <Button
                  variant="ghost"
                  className="w-full border border-tech-light hover:bg-tech-gray text-white"
                >
                  <Github className="mr-2 h-4 w-4" />
                  GitHub登录
                </Button>
              </div>
              
              <div className="flex justify-center">
                <Button
                  variant="link"
                  className="text-sm text-tech-primary hover:text-tech-secondary"
                  onClick={() => setIsRegistering(!isRegistering)}
                >
                  {isRegistering ? '已有账号？返回登录' : '没有账号？使用产品注册码注册'}
                </Button>
              </div>
              <div className="flex justify-center">
                <div className="text-xs text-gray-500 mt-4">
                  测试账号: test@example.com / 密码: test123456
                </div>
              </div>
            </form>
          ) : (
            <div className="space-y-4">
              <div className="flex justify-center space-x-4">
                <Button
                  variant="ghost"
                  className={`flex-1 ${selectedProvider === 'wechat' ? 'bg-green-600 hover:bg-green-700 text-white' : 'border border-tech-light'}`}
                  onClick={() => setSelectedProvider('wechat')}
                >
                  <span className="mr-2">微信</span>
                  <ExternalLink className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  className={`flex-1 ${selectedProvider === 'alipay' ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'border border-tech-light'}`}
                  onClick={() => setSelectedProvider('alipay')}
                >
                  <span className="mr-2">支付宝</span>
                  <ExternalLink className="h-4 w-4" />
                </Button>
              </div>
              
              {selectedProvider === 'wechat' ? (
                <WeChatLogin onLoginSuccess={() => console.log('微信登录成功')} />
              ) : (
                <AlipayLogin onLoginSuccess={() => console.log('支付宝登录成功')} />
              )}
              
              <Button
                variant="ghost"
                className="w-full border border-tech-light hover:bg-tech-gray text-white"
                onClick={() => setIsQRLogin(false)}
              >
                返回账号密码登录
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default LoginPage
