import { API_BASE_URL } from '@/config';

export interface ErrorReport {
  message?: string
  stack?: string | null
  extra?: any
  user?: string | null
}

export async function reportError(report: ErrorReport) {
  // Fallback: POST to backend logging endpoint (requires backend to accept)
  try {
    await fetch(`${API_BASE_URL || ''}/api/v1/system/logs`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: report.message, stack: report.stack, extra: report.extra, user: report.user }),
    });
  } catch (e) {
    // best-effort
    console.warn('[reportError] fallback reporting failed', e);
  }
}

export default reportError;

