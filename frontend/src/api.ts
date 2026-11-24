import { ChatResponse } from './types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function sendChat(
  sessionId: string,
  message: string,
  systemPrompt?: string
): Promise<ChatResponse> {
  const body: Record<string, string> = {
    session_id: sessionId,
    message,
  };
  if (systemPrompt !== undefined && systemPrompt.trim() !== '') {
    body.system_prompt = systemPrompt.trim();
  }

  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const data = await res.json();

  if (!res.ok) {
    const detail = data?.detail;
    if (typeof detail === 'string') {
      throw new Error(detail);
    }
    if (Array.isArray(detail)) {
      const msgs = detail.map((e: { msg?: string }) => e.msg || JSON.stringify(e)).join('; ');
      throw new Error(msgs);
    }
    throw new Error(`Server error: ${res.status}`);
  }

  return data as ChatResponse;
}
