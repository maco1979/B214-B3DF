import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig, AxiosResponse } from 'axios'
import toast from 'react-hot-toast'
import { API_BASE_URL, API_KEY } from '@/config'
import { clearAuthAndRedirect } from '@/hooks/useAuth'


// 本地存储的 token key，保持与 useAuth 一致
const TOKEN_KEY = 'ai-project-token'

const http: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: false,
  timeout: 30000,  // 30秒超时，防止请求无限等待
})

// 请求拦截器：注入 API Key 与 Bearer Token
http.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const headers = config.headers ?? {}
  const token = localStorage.getItem(TOKEN_KEY)

  if (API_KEY && !headers['X-API-KEY']) {
    headers['X-API-KEY'] = API_KEY
  }
  if (token && !headers.Authorization) {
    headers.Authorization = `Bearer ${token}`
  }

  config.headers = headers
  return config
})

// 响应拦截器：统一解包 success/data 或直接返回 data；401/403 清理并跳转
http.interceptors.response.use(
  (response: AxiosResponse) => {
    const data = response.data
    if (data && typeof data === 'object' && 'success' in data) {
      return data
    }
    return { success: true, data }
  },
  (error: AxiosError) => {
    const status = error.response?.status
    if (status === 401 || status === 403) {
      clearAuthAndRedirect()
    }

    const data = error.response?.data as any
    const message = data && typeof data === 'object' && 'error' in data
      ? data.error
      : error.message || '请求失败'

    if (!status) {
      toast.error('网络异常，请检查连接')
    } else if (status >= 500) {
      toast.error(`服务器错误(${status})：${message}`)
    } else if (status !== 401 && status !== 403) {
      toast.error(`请求失败(${status})：${message}`)
    }

    return Promise.resolve({ success: false, error: message })
  }
)


export { http }

