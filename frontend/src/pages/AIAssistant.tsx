import React, { useState, useRef, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import type { ApiResponse } from '@/services/api';
import { apiClient } from '@/services/api';

interface ChatMessage {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: number
  input_type?: string
}

const AIAssistant: React.FC = () => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 状态管理
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'system-1',
      type: 'assistant',
      content: '你好！我是智能AI助手，很高兴为您服务。有什么我可以帮助您的吗？',
      timestamp: Date.now(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [inputType, setInputType] = useState('text');
  const [contextId, setContextId] = useState<string | undefined>(undefined);
  const [isRecording, setIsRecording] = useState(false);

  // AI助手响应
  const getAIAssistantResponseMutation = useMutation<ApiResponse<any>, Error, { input_text: string, input_type?: string }, { userMessage: ChatMessage, assistantTypingId: string }>({
    mutationFn: async ({ input_text, input_type }) => apiClient.getAIAssistantResponse(input_text, input_type, contextId),
    onMutate: ({ input_text, input_type }) => {
      // 添加用户消息到聊天记录
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        type: 'user',
        content: input_text,
        timestamp: Date.now(),
        input_type,
      };
      setMessages(prev => [...prev, userMessage]);

      // 添加AI助手正在输入的占位符
      const assistantTyping: ChatMessage = {
        id: `assistant-typing-${Date.now()}`,
        type: 'assistant',
        content: '...',
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, assistantTyping]);

      // 清空输入框
      setInputText('');

      return { userMessage, assistantTypingId: assistantTyping.id };
    },
    onSuccess: (data, variables, context) => {
      // 更新AI助手的响应
      if (data.success && data.data) {
        setMessages(prev => {
          // 移除正在输入的占位符
          const updatedMessages = prev.filter(msg => msg.id !== context.assistantTypingId);

          // 添加AI助手的实际响应
          const assistantMessage: ChatMessage = {
            id: `assistant-${Date.now()}`,
            type: 'assistant',
            content: data.data.response || '抱歉，我无法理解您的请求。',
            timestamp: Date.now(),
          };

          return [...updatedMessages, assistantMessage];
        });

        // 更新上下文ID
        if (data.data.context_id) {
          setContextId(data.data.context_id);
        }
      } else {
        // 移除正在输入的占位符并显示错误
        setMessages(prev => prev.filter(msg => msg.id !== context.assistantTypingId));

        const errorMessage: ChatMessage = {
          id: `assistant-error-${Date.now()}`,
          type: 'assistant',
          content: `抱歉，处理您的请求时出错: ${data.error}`,
          timestamp: Date.now(),
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    },
    onError: (error, variables, context) => {
      // 移除正在输入的占位符并显示错误
      setMessages(prev => prev.filter(msg => msg.id !== context?.assistantTypingId));

      const errorMessage: ChatMessage = {
        id: `assistant-error-${Date.now()}`,
        type: 'assistant',
        content: `抱歉，处理您的请求时出错: ${error.message}`,
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, errorMessage]);
    },
  });

  // 滚动到最新消息
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // 处理发送消息
  const handleSendMessage = () => {
    if (!inputText.trim()) {
      alert('消息内容不能为空');
      return;
    }

    getAIAssistantResponseMutation.mutate({ input_text: inputText, input_type: inputType });
  };

  // 处理语音输入
  const handleVoiceInput = () => {
    setIsRecording(!isRecording);
    alert(isRecording ? '语音输入已停止' : '正在录音...');

    // 这里应该实现实际的语音识别逻辑
    if (!isRecording) {
      setTimeout(() => {
        setIsRecording(false);
        setInputText('这是一段语音识别的示例文本');
        alert('语音识别完成');
      }, 3000);
    }
  };

  // 处理文件输入
  const handleFileInput = () => {
    // 这里应该实现实际的文件上传和处理逻辑
    alert('文件上传功能开发中');
  };

  // 处理图片输入
  const handleImageInput = () => {
    // 这里应该实现实际的图片上传和处理逻辑
    alert('图片上传功能开发中');
  };

  // 处理键盘事件
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="space-y-6 p-4 md:p-6 bg-background text-foreground">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-foreground">AI助手交互界面</h1>
        <div className="flex gap-2">
          <button
            onClick={() => {
              setMessages([
                {
                  id: 'system-1',
                  type: 'assistant',
                  content: '你好！我是智能AI助手，很高兴为您服务。有什么我可以帮助您的吗？',
                  timestamp: Date.now(),
                },
              ]);
              setContextId(undefined);
              alert('已重置对话');
            }}
            className="bg-muted hover:bg-muted/90 text-foreground font-bold py-2 px-4 rounded border"
          >
            重置对话
          </button>
        </div>
      </div>

      <div className="bg-card p-6 rounded-lg shadow-md border">
        <div className="flex flex-col h-[600px]">
          {/* 聊天消息区域 */}
          <div className="flex-1 overflow-y-auto pr-4 mb-4 space-y-4">
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[70%] p-4 rounded-lg ${message.type === 'user' ? 'bg-blue-100 text-foreground border border-blue-200' : 'bg-muted text-foreground border'}`}
                >
                  {message.content}
                  <div className="flex items-center justify-between mt-2 text-xs opacity-70">
                    <span className="text-muted-foreground">{new Date(message.timestamp).toLocaleTimeString()}</span>
                    {message.input_type && message.type === 'user' && (
                      <span className="ml-2 px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">
                        {message.input_type === 'text' ? '文本' : message.input_type === 'voice' ? '语音' : '其他'}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* 输入区域 */}
          <div className="border-t pt-4">
            <div className="flex items-center mb-2 gap-2">
              <label htmlFor="input-type" className="block text-sm font-medium text-foreground">输入类型</label>
              <select
                id="input-type"
                value={inputType}
                onChange={e => setInputType(e.target.value)}
                className="border bg-background text-foreground rounded-md p-2 w-24"
              >
                <option value="text">文本</option>
                <option value="voice">语音</option>
                <option value="image">图片</option>
                <option value="file">文件</option>
              </select>

              {/* 输入类型对应的操作按钮 */}
              {inputType === 'voice' && (
                <button
                  onClick={handleVoiceInput}
                  className={`p-2 rounded-full border ${isRecording ? 'bg-red-100 text-red-700 hover:bg-red-200' : 'bg-muted text-foreground hover:bg-muted/90'}`}
                >
                  🎤
                </button>
              )}

              {inputType === 'image' && (
                <button
                  onClick={handleImageInput}
                  className="p-2 rounded-full bg-muted text-foreground hover:bg-muted/90 border"
                >
                  📷
                </button>
              )}

              {inputType === 'file' && (
                <button
                  onClick={handleFileInput}
                  className="p-2 rounded-full bg-muted text-foreground hover:bg-muted/90 border"
                >
                  📁
                </button>
              )}
            </div>

            <div className="flex gap-2">
              <textarea
                placeholder={`请输入${inputType === 'text' ? '文本' : inputType === 'voice' ? '语音' : '内容'}`}
                value={inputText}
                onChange={e => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                className="border bg-background text-foreground rounded-md p-2 flex-1 min-h-[100px] disabled:opacity-50 placeholder:text-muted-foreground"
                disabled={isRecording}
              />
              <button
                onClick={handleSendMessage}
                disabled={getAIAssistantResponseMutation.isPending || isRecording}
                className="p-2 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 disabled:bg-muted disabled:cursor-not-allowed border"
              >
                {getAIAssistantResponseMutation.isPending ? '⏳' : '➤'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 功能说明 */}
      <div className="bg-card p-6 rounded-lg shadow-md border">
        <h2 className="text-xl font-semibold mb-2 text-foreground">AI助手功能</h2>
        <p className="text-muted-foreground mb-4">
          智能AI助手可以帮助您完成各种任务，包括但不限于：
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">智能体管理</span>
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">自动化检查</span>
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">用户习惯分析</span>
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">设备自动化控制</span>
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">场景管理</span>
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">调度服务</span>
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">代码质量检查</span>
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">错误监控</span>
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">自动测试</span>
          <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-muted text-foreground border">全面检查</span>
        </div>
      </div>
    </div>
  );
};

export default AIAssistant;
