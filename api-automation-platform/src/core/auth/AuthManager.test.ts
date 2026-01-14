import { AuthManager } from './AuthManager';
import { ApiKeyAuth, BasicAuth, JwtAuth, OAuth2Auth, HttpRequestConfig, HttpMethod } from '../../types';

describe('AuthManager', () => {
  let authManager: AuthManager;

  beforeEach(() => {
    authManager = new AuthManager();
  });

  describe('registerAuthConfig', () => {
    it('should register an API key auth config', () => {
      const authConfig: ApiKeyAuth = {
        type: 'apiKey',
        key: 'X-API-Key',
        value: 'test-api-key',
        location: 'header',
      };

      authManager.registerAuthConfig('test-api-key-auth', authConfig);
      const retrievedConfig = authManager.getAuthConfig('test-api-key-auth');

      expect(retrievedConfig).toEqual(authConfig);
    });

    it('should register a basic auth config', () => {
      const authConfig: BasicAuth = {
        type: 'basic',
        username: 'testuser',
        password: 'testpassword',
      };

      authManager.registerAuthConfig('test-basic-auth', authConfig);
      const retrievedConfig = authManager.getAuthConfig('test-basic-auth');

      expect(retrievedConfig).toEqual(authConfig);
    });
  });

  describe('applyAuthToRequest', () => {
    it('should apply API key auth to request headers', () => {
      const authConfig: ApiKeyAuth = {
        type: 'apiKey' as const,
        key: 'X-API-Key',
        value: 'test-api-key',
        location: 'header' as const,
      };

      const requestConfig: HttpRequestConfig = {
        method: 'GET' as HttpMethod,
        url: 'https://api.example.com/test',
        headers: {},
        params: {},
      };

      const result = authManager.applyAuthToRequest(requestConfig, authConfig);

      expect(result.headers).toHaveProperty('X-API-Key', 'test-api-key');
    });

    it('should apply API key auth to request query params', () => {
      const authConfig: ApiKeyAuth = {
        type: 'apiKey' as const,
        key: 'api_key',
        value: 'test-api-key',
        location: 'query' as const,
      };

      const requestConfig: HttpRequestConfig = {
        method: 'GET' as HttpMethod,
        url: 'https://api.example.com/test',
        headers: {},
        params: {},
      };

      const result = authManager.applyAuthToRequest(requestConfig, authConfig);

      expect(result.params).toHaveProperty('api_key', 'test-api-key');
    });

    it('should apply basic auth to request headers', () => {
      const authConfig: BasicAuth = {
        type: 'basic' as const,
        username: 'testuser',
        password: 'testpassword',
      };

      const requestConfig: HttpRequestConfig = {
        method: 'GET' as HttpMethod,
        url: 'https://api.example.com/test',
        headers: {},
        params: {},
      };

      const result = authManager.applyAuthToRequest(requestConfig, authConfig);

      expect(result.headers).toHaveProperty('Authorization');
      expect(result.headers?.Authorization).toMatch(/^Basic /);
    });

    it('should apply JWT auth to request headers', () => {
      const authConfig: JwtAuth = {
        type: 'jwt' as const,
        token: 'test-jwt-token',
      };

      const requestConfig: HttpRequestConfig = {
        method: 'GET' as HttpMethod,
        url: 'https://api.example.com/test',
        headers: {},
        params: {},
      };

      const result = authManager.applyAuthToRequest(requestConfig, authConfig);

      expect(result.headers).toHaveProperty('Authorization', 'Bearer test-jwt-token');
    });

    it('should apply OAuth2 auth to request headers', () => {
      const authConfig: OAuth2Auth = {
        type: 'oauth2' as const,
        clientId: 'test-client-id',
        clientSecret: 'test-client-secret',
        accessToken: 'test-access-token',
        tokenUrl: 'https://auth.example.com/token',
      };

      const requestConfig: HttpRequestConfig = {
        method: 'GET' as HttpMethod,
        url: 'https://api.example.com/test',
        headers: {},
        params: {},
      };

      const result = authManager.applyAuthToRequest(requestConfig, authConfig);

      expect(result.headers).toHaveProperty('Authorization', 'Bearer test-access-token');
    });
  });

  describe('isJwtExpired', () => {
    it('should return false for a non-expired JWT token', () => {
      const authConfig: JwtAuth = {
        type: 'jwt' as const,
        token: 'test-jwt-token',
        expiresAt: Date.now() + 3600000, // 1 hour from now
      };

      const result = authManager.isJwtExpired(authConfig);

      expect(result).toBe(false);
    });

    it('should return true for an expired JWT token', () => {
      const authConfig: JwtAuth = {
        type: 'jwt' as const,
        token: 'test-jwt-token',
        expiresAt: Date.now() - 3600000, // 1 hour ago
      };

      const result = authManager.isJwtExpired(authConfig);

      expect(result).toBe(true);
    });

    it('should return false for a JWT token without expiration time', () => {
      const authConfig: JwtAuth = {
        type: 'jwt' as const,
        token: 'test-jwt-token',
      };

      const result = authManager.isJwtExpired(authConfig);

      expect(result).toBe(false);
    });
  });

  describe('isOAuth2Expired', () => {
    it('should return false for a non-expired OAuth2 token', () => {
      const authConfig: OAuth2Auth = {
        type: 'oauth2' as const,
        clientId: 'test-client-id',
        clientSecret: 'test-client-secret',
        accessToken: 'test-access-token',
        tokenUrl: 'https://auth.example.com/token',
        expiresAt: Date.now() + 3600000, // 1 hour from now
      };

      const result = authManager.isOAuth2Expired(authConfig);

      expect(result).toBe(false);
    });

    it('should return true for an expired OAuth2 token', () => {
      const authConfig: OAuth2Auth = {
        type: 'oauth2' as const,
        clientId: 'test-client-id',
        clientSecret: 'test-client-secret',
        accessToken: 'test-access-token',
        tokenUrl: 'https://auth.example.com/token',
        expiresAt: Date.now() - 3600000, // 1 hour ago
      };

      const result = authManager.isOAuth2Expired(authConfig);

      expect(result).toBe(true);
    });

    it('should return false for an OAuth2 token without expiration time', () => {
      const authConfig: OAuth2Auth = {
        type: 'oauth2' as const,
        clientId: 'test-client-id',
        clientSecret: 'test-client-secret',
        accessToken: 'test-access-token',
        tokenUrl: 'https://auth.example.com/token',
      };

      const result = authManager.isOAuth2Expired(authConfig);

      expect(result).toBe(false);
    });
  });
});
