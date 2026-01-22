import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { RefreshCw, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { apiClient } from '@/services/api';
import { useAuth } from '@/hooks/useAuth';

interface QRLoginProps {
  onLoginSuccess?: () => void
}

interface QRCodeData {
  qr_id: string
  qr_code_url: string
  expires_in: number
  created_at: string
}

interface QRLoginStatus {
  qr_id: string
  status: 'pending' | 'scanned' | 'confirmed' | 'expired'
  user_info?: {
    id: string
    name: string
    email: string
    avatar?: string
    source: string
  }
  access_token?: string
}

// 微信扫码登录组件
export const WeChatLogin: React.FC<QRLoginProps> = ({ onLoginSuccess }) => {
  const [qrCode, setQRCode] = useState<QRCodeData | null>(null);
  const [status, setStatus] = useState<string>('');
  const [countdown, setCountdown] = useState<number>(0);
  const { login } = useAuth();

  // 生成二维码
  const generateQRCode = async () => {
    try {
      const response = await apiClient.post<QRCodeData>('/auth/qr/generate', {
        provider: 'wechat',
      });
      if (!response.success) {
        throw new Error(response.error || '生成二维码失败');
      }
      if (!response.data) {
        throw new Error('生成二维码失败：返回数据为空');
      }
      setQRCode(response.data);
      setStatus('等待扫码');
      setCountdown(response.data.expires_in);
    } catch (error) {
      console.error('生成微信二维码失败:', error);
      setStatus('生成二维码失败');
    }
  };

  // 轮询检查登录状态
  useEffect(() => {
    if (!qrCode) {
 return;
}

    const interval = setInterval(async () => {
      try {
        const response = await apiClient.get<QRLoginStatus>(`/auth/qr/status/${qrCode.qr_id}`);
        const data: QRLoginStatus = response.data!;
        setStatus(getStatusText(data.status));

        if (data.status === 'confirmed' && data.user_info) {
          // 登录成功
          await login(data.user_info.email, '');
          onLoginSuccess?.();
          clearInterval(interval);
        } else if (data.status === 'expired') {
          clearInterval(interval);
        }
      } catch (error) {
        console.error('检查登录状态失败:', error);
      }
    }, 2000); // 每2秒轮询一次

    return () => clearInterval(interval);
  }, [qrCode, login, onLoginSuccess]);

  // 倒计时
  useEffect(() => {
    if (countdown <= 0 || !qrCode) {
 return;
}

    const timer = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(timer);
          setStatus('二维码已过期');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [countdown, qrCode]);

  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending':
        return '等待扫码';
      case 'scanned':
        return '已扫码，等待确认';
      case 'confirmed':
        return '登录成功';
      case 'expired':
        return '二维码已过期';
      default:
        return status;
    }
  };

  useEffect(() => {
    generateQRCode();
  }, []);

  return (
    <div className="space-y-4">
      <Card className="bg-tech-dark border-tech-light">
        <CardHeader className="text-center">
          <CardTitle className="text-xl text-green-500">微信扫码登录</CardTitle>
          <CardDescription className="text-gray-400">
            请使用微信扫描下方二维码
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center space-y-4">
            {qrCode && (
              <div className="p-4 bg-white rounded-lg">
                <img
                  src={qrCode.qr_code_url}
                  alt="微信登录二维码"
                  className="w-48 h-48 object-contain"
                />
              </div>
            )}

            <div className="flex items-center space-x-2 text-center">
              {status === '等待扫码' && <Clock className="w-4 h-4 text-gray-400" />}
              {status === '已扫码，等待确认' && <CheckCircle className="w-4 h-4 text-yellow-500" />}
              {status === '登录成功' && <CheckCircle className="w-4 h-4 text-green-500" />}
              {status === '二维码已过期' && <AlertCircle className="w-4 h-4 text-red-500" />}
              <span className="text-sm font-medium">{status}</span>
              {countdown > 0 && (
                <span className="text-xs text-gray-400">({Math.floor(countdown / 60)}:{(countdown % 60).toString().padStart(2, '0')})</span>
              )}
            </div>

            <Button
              variant="ghost"
              className="border border-tech-light hover:bg-tech-gray"
              onClick={generateQRCode}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              刷新二维码
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// 支付宝扫码登录组件
export const AlipayLogin: React.FC<QRLoginProps> = ({ onLoginSuccess }) => {
  const [qrCode, setQRCode] = useState<QRCodeData | null>(null);
  const [status, setStatus] = useState<string>('');
  const [countdown, setCountdown] = useState<number>(0);
  const { login } = useAuth();

  // 生成二维码
  const generateQRCode = async () => {
    try {
      const response = await apiClient.post<QRCodeData>('/auth/qr/generate', {
        provider: 'alipay',
      });
      if (!response.success) {
        throw new Error(response.error || '生成二维码失败');
      }
      if (!response.data) {
        throw new Error('生成二维码失败：返回数据为空');
      }
      setQRCode(response.data);
      setStatus('等待扫码');
      setCountdown(response.data.expires_in);
    } catch (error) {
      console.error('生成支付宝二维码失败:', error);
      setStatus('生成二维码失败');
    }
  };

  // 轮询检查登录状态
  useEffect(() => {
    if (!qrCode) {
 return;
}

    const interval = setInterval(async () => {
      try {
        const response = await apiClient.get<QRLoginStatus>(`/auth/qr/status/${qrCode.qr_id}`);
        const data: QRLoginStatus = response.data!;
        setStatus(getStatusText(data.status));

        if (data.status === 'confirmed' && data.user_info) {
          // 登录成功
          await login(data.user_info.email, '');
          onLoginSuccess?.();
          clearInterval(interval);
        } else if (data.status === 'expired') {
          clearInterval(interval);
        }
      } catch (error) {
        console.error('检查登录状态失败:', error);
      }
    }, 2000); // 每2秒轮询一次

    return () => clearInterval(interval);
  }, [qrCode, login, onLoginSuccess]);

  // 倒计时
  useEffect(() => {
    if (countdown <= 0 || !qrCode) {
 return;
}

    const timer = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(timer);
          setStatus('二维码已过期');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [countdown, qrCode]);

  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending':
        return '等待扫码';
      case 'scanned':
        return '已扫码，等待确认';
      case 'confirmed':
        return '登录成功';
      case 'expired':
        return '二维码已过期';
      default:
        return status;
    }
  };

  useEffect(() => {
    generateQRCode();
  }, []);

  return (
    <div className="space-y-4">
      <Card className="bg-tech-dark border-tech-light">
        <CardHeader className="text-center">
          <CardTitle className="text-xl text-blue-500">支付宝扫码登录</CardTitle>
          <CardDescription className="text-gray-400">
            请使用支付宝扫描下方二维码
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center space-y-4">
            {qrCode && (
              <div className="p-4 bg-white rounded-lg">
                <img
                  src={qrCode.qr_code_url}
                  alt="支付宝登录二维码"
                  className="w-48 h-48 object-contain"
                />
              </div>
            )}

            <div className="flex items-center space-x-2 text-center">
              {status === '等待扫码' && <Clock className="w-4 h-4 text-gray-400" />}
              {status === '已扫码，等待确认' && <CheckCircle className="w-4 h-4 text-yellow-500" />}
              {status === '登录成功' && <CheckCircle className="w-4 h-4 text-green-500" />}
              {status === '二维码已过期' && <AlertCircle className="w-4 h-4 text-red-500" />}
              <span className="text-sm font-medium">{status}</span>
              {countdown > 0 && (
                <span className="text-xs text-gray-400">({Math.floor(countdown / 60)}:{(countdown % 60).toString().padStart(2, '0')})</span>
              )}
            </div>

            <Button
              variant="ghost"
              className="border border-tech-light hover:bg-tech-gray"
              onClick={generateQRCode}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              刷新二维码
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
