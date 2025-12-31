import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient } from '@/services/api'
import { User as ApiUser } from '@/services/api'

interface AuthContextType {
  user: ApiUser | null
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  isAuthenticated: boolean
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<ApiUser | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    // 检查本地存储中的用户信息和令牌
    const savedUser = localStorage.getItem('ai-project-user')
    const savedToken = localStorage.getItem('ai-project-token')
    if (savedUser && savedToken) {
      setUser(JSON.parse(savedUser))
      setToken(savedToken)
    }
  }, [])

  const login = async (email: string, password: string) => {
    // 调用实际的后端API进行登录
    console.log('useAuth login called with email:', email, 'password:', password)
    try {
      console.log('Calling apiClient.login method')
      const response = await apiClient.login(email, password)
      console.log('Login API response:', response)
      if (response.success && response.data) {
        const userInfo = response.data.user_info
        const accessToken = response.data.access_token
        
        setUser(userInfo)
        setToken(accessToken)
        localStorage.setItem('ai-project-user', JSON.stringify(userInfo))
        localStorage.setItem('ai-project-token', accessToken)
      } else {
        throw new Error(response.error || '登录失败')
      }
    } catch (error) {
      console.error('登录失败:', error)
      throw error
    }
  }

  const logout = () => {
    setUser(null)
    setToken(null)
    localStorage.removeItem('ai-project-user')
    localStorage.removeItem('ai-project-token')
    navigate('/login')
  }

  const value: AuthContextType = {
    user,
    login,
    logout,
    isAuthenticated: !!user || (!!localStorage.getItem('ai-project-user') && !!localStorage.getItem('ai-project-token'))
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

// 外部可调用的 token 清理（供拦截器使用）
export function clearAuthAndRedirect() {
  localStorage.removeItem('ai-project-user')
  localStorage.removeItem('ai-project-token')
  if (typeof window !== 'undefined' && window.location) {
    if (typeof window.location.assign === 'function') {
      window.location.assign('/login')
    } else {
      window.location.href = '/login'
    }
  }
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth必须在AuthProvider内部使用')
  }
  return context
}