import axios from 'axios';
import { API_BASE_URL } from '@/config';

export interface RefreshResponse {
  success: boolean
  data?: { access_token?: string }
  error?: string
}

/**
 * Attempt to refresh access token using refresh endpoint.
 * Returns the new token string when available, or null on failure.
 */
export async function refreshAccessToken(): Promise<string | null> {
  try {
    const client = axios.create({ baseURL: API_BASE_URL || undefined, timeout: 10000, withCredentials: true });
    const res = await client.post<RefreshResponse>('/api/v1/auth/refresh');
    const { data } = res;
    if (data && data.success && data.data?.access_token) {
      console.debug('[refresh] token refreshed');
      return data.data.access_token;
    }
    console.warn('[refresh] refresh endpoint returned no token', data);
    return null;
  } catch (err: any) {
    console.warn('[refresh] refresh failed', err?.message ? err.message : err);
    return null;
  }
}

export default refreshAccessToken;

