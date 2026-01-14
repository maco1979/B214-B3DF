console.log('loaded useAuth.test');
import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';

import { renderHook, act, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AuthProvider, useAuth, clearAuthAndRedirect } from './useAuth';

console.log('before describe useAuth');

const { loginMock, mockNavigate } = vi.hoisted(() => ({

  loginMock: vi.fn(),
  mockNavigate: vi.fn(),
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<any>('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

vi.mock('@/services/api', async () => {
  const actual = await vi.importActual<any>('@/services/api');
  return {
    ...actual,
    apiClient: { ...actual.apiClient, login: loginMock },
  };
});


function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <MemoryRouter initialEntries={['/']}>
      <AuthProvider>{children}</AuthProvider>
    </MemoryRouter>
  );
}

describe('useAuth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockNavigate.mockReset();
  });

  it('login 成功后应写入 token 与用户并标记已认证', async () => {
    loginMock.mockResolvedValue({
      success: true,
      data: {
        user_info: {
          id: '1',
          username: 'tester',
          email: 't@example.com',
          role: 'user',
          source: 'local',
          created_at: 'now',
        },
        access_token: 'token-123',
        token_type: 'bearer',
        message: 'ok',
      },
    });

    const { result } = renderHook(() => useAuth(), { wrapper });

    await act(async () => {
      await result.current.login('t@example.com', 'pass');
    });

    await waitFor(() => expect(result.current.isAuthenticated).toBe(true));
    expect(localStorage.getItem('ai-project-token')).toBe('token-123');
    expect(localStorage.getItem('ai-project-user')).toContain('tester');
  });

  it('logout 应清除本地存储并跳转登录', async () => {
    loginMock.mockResolvedValue({
      success: true,
      data: {
        user_info: {
          id: '1',
          username: 'tester',
          email: 't@example.com',
          role: 'user',
          source: 'local',
          created_at: 'now',
        },
        access_token: 'token-123',
        token_type: 'bearer',
        message: 'ok',
      },
    });

    const { result } = renderHook(() => useAuth(), { wrapper });

    await act(async () => {
      await result.current.login('t@example.com', 'pass');
    });

    await act(async () => {
      result.current.logout();
    });

    expect(localStorage.getItem('ai-project-token')).toBeNull();
    expect(localStorage.getItem('ai-project-user')).toBeNull();
    expect(mockNavigate).toHaveBeenCalledWith('/login');
  });
});

describe('clearAuthAndRedirect', () => {
  beforeEach(() => {
    localStorage.setItem('ai-project-token', 'token');
    localStorage.setItem('ai-project-user', 'user');
    // jsdom 允许直接赋值 href
    window.location.href = 'http://localhost/any';
  });

  it('应清理本地存储并跳转到 /login', () => {
    clearAuthAndRedirect();
    expect(localStorage.getItem('ai-project-token')).toBeNull();
    expect(localStorage.getItem('ai-project-user')).toBeNull();
    expect(window.location.pathname).toBe('/login');
  });
});

