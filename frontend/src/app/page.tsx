'use client';

import { useState, useRef, useEffect } from 'react';
import { uploadPDF, queryStream } from '@/lib/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  references?: string[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.pdf')) {
      alert('请上传 PDF 文件');
      return;
    }

    setIsUploading(true);
    setUploadStatus('上传中...');

    try {
      const result = await uploadPDF(file);
      setUploadStatus(`✅ 上传成功！共 ${result.chunks} 个文本块`);
      e.target.value = '';
    } catch (err) {
      setUploadStatus('❌ 上传失败，请重试');
      console.error(err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const assistantId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '', references: [] },
    ]);

    let fullContent = '';
    let refs: string[] = [];

    eventSourceRef.current = queryStream(
      userMessage.content,
      sessionId,
      (chunk) => {
        fullContent += chunk;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId ? { ...msg, content: fullContent } : msg
          )
        );
      },
      (references) => {
        refs = references;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId ? { ...msg, references: refs } : msg
          )
        );
      },
      (sid) => {
        setSessionId(sid);
      },
      () => {
        setIsLoading(false);
      },
      (err) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: `❌ 错误: ${err}` }
              : msg
          )
        );
        setIsLoading(false);
      }
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-800">
            📚 RAG 知识库问答
          </h1>
          <div className="flex items-center gap-3">
            {sessionId && (
              <span className="text-xs text-gray-400">
                会话: {sessionId.slice(0, 8)}...
              </span>
            )}
            <label
              className={`px-4 py-2 rounded-lg text-sm cursor-pointer ${
                isUploading
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              }`}
            >
              {isUploading ? '上传中...' : '📤 上传 PDF'}
              <input
                type="file"
                accept=".pdf"
                onChange={handleUpload}
                disabled={isUploading}
                className="hidden"
              />
            </label>
          </div>
        </div>
        {uploadStatus && (
          <div className="max-w-4xl mx-auto px-4 pb-2">
            <p className={`text-sm ${uploadStatus.startsWith('✅') ? 'text-green-600' : 'text-red-600'}`}>
              {uploadStatus}
            </p>
          </div>
        )}
      </header>

      <main className="max-w-4xl mx-auto px-4 py-6">
        <div className="bg-white rounded-xl shadow-sm border min-h-[500px] max-h-[70vh] overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-400">
              <p className="text-6xl mb-4">🤖</p>
              <p className="text-lg">上传 PDF 后，开始提问吧！</p>
              <p className="text-sm">支持多轮对话，自动记住上下文</p>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`mb-4 flex ${
                  msg.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {msg.role === 'assistant' && msg.references && msg.references.length > 0 && (
                    <div className="text-xs text-gray-500 mb-2">
                      📖 引用 {msg.references.length} 个文档片段
                    </div>
                  )}

                  <div className="whitespace-pre-wrap break-words">
                    {msg.content || (msg.role === 'assistant' && '思考中...')}
                  </div>

                  {msg.role === 'assistant' && msg.references && msg.references.length > 0 && (
                    <details className="mt-2 text-xs text-gray-500">
                      <summary className="cursor-pointer">查看引用原文</summary>
                      <div className="mt-1 space-y-1 max-h-40 overflow-y-auto">
                        {msg.references.map((ref, idx) => (
                          <div key={idx} className="bg-white/50 p-2 rounded border border-gray-200">
                            <span className="font-medium">[引用{idx + 1}]</span> {ref.slice(0, 150)}...
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-lg px-4 py-3 text-gray-400">
                <span className="animate-pulse">●</span>{' '}
                <span className="animate-pulse delay-75">●</span>{' '}
                <span className="animate-pulse delay-150">●</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="mt-4 flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="输入你的问题..."
            disabled={isLoading}
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
          >
            {isLoading ? '生成中...' : '发送'}
          </button>
        </form>
      </main>
    </div>
  );
}