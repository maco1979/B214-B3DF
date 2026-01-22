import { AuthenticationConfig, HttpRequestConfig } from '../../types';

/**
 * 认证管理器，负责处理各种认证方式
 */
export class AuthManager {
  private authConfigs: Map<string, AuthenticationConfig> = new Map();

  /**
   * 注册认证配置
   */
  registerAuthConfig(id: string, config: AuthenticationConfig): void {
    this.authConfigs.set(id, config);
  }

  /**
   * 获取认证配置
   */
  getAuthConfig(id: string): AuthenticationConfig | undefined {
    return this.authConfigs.get(id);
  }

  /**
   * 应用认证到请求配置
   */
  applyAuthToRequest(requestConfig: HttpRequestConfig, authConfig: AuthenticationConfig): HttpRequestConfig {
    const headers = { ...requestConfig.headers };
    const params = { ...requestConfig.params };

    switch (authConfig.type) {
      case 'apiKey':
        return this.applyApiKeyAuth(requestConfig, authConfig, headers, params);
      case 'basic':
        return this.applyBasicAuth(requestConfig, authConfig, headers);
      case 'jwt':
        return this.applyJwtAuth(requestConfig, authConfig, headers);
      case 'oauth2':
        return this.applyOAuth2Auth(requestConfig, authConfig, headers);
      case 'none':
      default:
        return requestConfig;
    }
  }

  /**
   * 应用API Key认证
   */
  private applyApiKeyAuth(
    requestConfig: HttpRequestConfig,
    authConfig: { type: 'apiKey'; key: string; value: string; location: 'header' | 'query' | 'cookie' },
    headers: Record<string, string>,
    params: Record<string, any>
  ): HttpRequestConfig {
    const { key, value, location } = authConfig;

    switch (location) {
      case 'header':
        headers[key] = value;
        break;
      case 'query':
        params[key] = value;
        break;
      case 'cookie':
        const cookieHeader = headers['Cookie'] || '';
        headers['Cookie'] = cookieHeader ? `${cookieHeader}; ${key}=${value}` : `${key}=${value}`;
        break;
    }

    return {
      ...requestConfig,
      headers,
      params
    };
  }

  /**
   * 应用Basic认证
   */
  private applyBasicAuth(
    requestConfig: HttpRequestConfig,
    authConfig: { type: 'basic'; username: string; password: string },
    headers: Record<string, string>
  ): HttpRequestConfig {
    const { username, password } = authConfig;
    const authString = Buffer.from(`${username}:${password}`).toString('base64');
    headers['Authorization'] = `Basic ${authString}`;

    return {
      ...requestConfig,
      headers
    };
  }

  /**
   * 应用JWT认证
   */
  private applyJwtAuth(
    requestConfig: HttpRequestConfig,
    authConfig: { type: 'jwt'; token: string },
    headers: Record<string, string>
  ): HttpRequestConfig {
    headers['Authorization'] = `Bearer ${authConfig.token}`;

    return {
      ...requestConfig,
      headers
    };
  }

  /**
   * 应用OAuth2认证
   */
  private applyOAuth2Auth(
    requestConfig: HttpRequestConfig,
    authConfig: { type: 'oauth2'; accessToken: string },
    headers: Record<string, string>
  ): HttpRequestConfig {
    headers['Authorization'] = `Bearer ${authConfig.accessToken}`;

    return {
      ...requestConfig,
      headers
    };
  }

  /**
   * 刷新OAuth2令牌
   */
  async refreshOAuth2Token(authConfig: { type: 'oauth2'; clientId: string; clientSecret: string; refreshToken: string; tokenUrl: string }): Promise<{ accessToken: string; refreshToken?: string; expiresAt?: number }> {
    const response = await fetch(authConfig.tokenUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        client_id: authConfig.clientId,
        client_secret: authConfig.clientSecret,
        refresh_token: authConfig.refreshToken,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to refresh OAuth2 token: ${response.statusText}`);
    }

    const data = await response.json() as { access_token: string; refresh_token?: string; expires_in?: number };
    const expiresAt = data.expires_in ? Date.now() + data.expires_in * 1000 : undefined;

    return {
      accessToken: data.access_token,
      refreshToken: data.refresh_token || authConfig.refreshToken,
      expiresAt,
    };
  }

  /**
   * 检查JWT令牌是否过期
   */
  isJwtExpired(authConfig: { type: 'jwt'; expiresAt?: number }): boolean {
    if (!authConfig.expiresAt) return false;
    return Date.now() > authConfig.expiresAt;
  }

  /**
   * 检查OAuth2令牌是否过期
   */
  isOAuth2Expired(authConfig: { type: 'oauth2'; expiresAt?: number }): boolean {
    if (!authConfig.expiresAt) return false;
    return Date.now() > authConfig.expiresAt;
  }
}
