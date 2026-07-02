import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 上传 PDF
export async function uploadPDF(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

// 流式问答（使用 EventSource）
export function queryStream(
  question: string,
  sessionId: string | null,
  onChunk: (text: string) => void,
  onReferences: (refs: string[]) => void,
  onSession: (sid: string) => void,
  onDone: () => void,
  onError: (err: string) => void
) {
  const url = new URL(`${API_BASE}/query`);
  url.searchParams.set('q', question);
  if (sessionId) url.searchParams.set('session_id', sessionId);

  const eventSource = new EventSource(url.toString());

  eventSource.addEventListener('session', (e) => {
    onSession(e.data);
  });

  eventSource.addEventListener('references', (e) => {
    try {
      const refs = JSON.parse(e.data);
      onReferences(Array.isArray(refs) ? refs : [e.data]);
    } catch {
      onReferences([e.data]);
    }
  });

  eventSource.addEventListener('chunk', (e) => {
    onChunk(e.data);
  });

  eventSource.addEventListener('done', () => {
    onDone();
    eventSource.close();
  });

  eventSource.addEventListener('error', (e) => {
    onError('连接中断，请重试');
    eventSource.close();
  });

  eventSource.onerror = () => {
    eventSource.close();
  };

  return eventSource;
}