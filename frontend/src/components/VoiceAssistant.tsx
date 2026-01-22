import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2, VolumeX, Loader2 } from 'lucide-react';
import { Button } from './ui/button';
import axios from 'axios';

// 添加Web Speech API类型定义
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
}

interface SpeechRecognitionResult {
  [index: number]: SpeechRecognitionAlternative;
  length: number;
  isFinal: boolean;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionResultList {
  [index: number]: SpeechRecognitionResult;
  length: number;
}

interface SpeechRecognition {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: (event: SpeechRecognitionEvent) => void;
  onerror: (event: SpeechRecognitionErrorEvent) => void;
  onend: () => void;
  start: () => void;
  stop: () => void;
}

// 扩展Window接口
interface Window {
  SpeechRecognition?: new () => SpeechRecognition;
  webkitSpeechRecognition?: new () => SpeechRecognition;
}

const VoiceAssistant: React.FC = () => {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState('');
  const [history, setHistory] = useState<{ type: 'user' | 'assistant'; text: string; interactionId?: string }[]>([]);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const synthesisRef = useRef<SpeechSynthesis>(window.speechSynthesis);

  // 生成唯一交互ID
  const generateInteractionId = () => Date.now().toString(36) + Math.random().toString(36).substring(2, 5);

  // 提交反馈
  const submitFeedback = async (response: string, feedbackType: 'positive' | 'negative') => {
    try {
      // 找到对应的交互ID
      const interactionEntry = history.find(item => item.type === 'assistant' && item.text === response);
      const interactionId = interactionEntry?.interactionId;

      await axios.post('http://localhost:8001/api/ai-assistant/feedback', {
        response,
        type: feedbackType,
        timestamp: new Date().toISOString(),
        interaction_id: interactionId,
      });
      console.log('反馈提交成功');
    } catch (error) {
      console.error('反馈提交失败:', error);
    }
  };

  // 处理语音命令
  const handleVoiceCommand = async (command: string) => {
    setIsLoading(true);
    setError(null);

    try {
      // 生成唯一交互ID
      const interactionId = generateInteractionId();

      // 调用后端API获取响应
      const response = await axios.post('http://localhost:8001/api/ai-assistant/get-response', {
        input_text: command,
        input_type: 'text',
        context_id: interactionId,
      });

      const aiResponse = response.data.response;
      setHistory(prev => [...prev, { type: 'assistant', text: aiResponse, interactionId }]);
      speak(aiResponse);

      // 如果是本地控制命令，可能需要额外处理
      if (response.data.type === 'local_control') {
        console.log('本地控制命令执行成功:', aiResponse);
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.response || '抱歉，处理请求失败，请稍后重试';
      setError(errorMessage);
      setHistory(prev => [...prev, { type: 'assistant', text: errorMessage }]);
      speak(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // 语音合成
  const speak = (text: string) => {
    // 清理之前的语音
    synthesisRef.current.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'zh-CN';
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;

    utterance.onstart = () => {
      setIsSpeaking(true);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
    };

    synthesisRef.current.speak(utterance);
  };

  // 切换语音识别状态
  const toggleListening = () => {
    if (!recognitionRef.current) {
      alert('您的浏览器不支持语音识别功能');
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  // 停止语音合成
  const stopSpeaking = () => {
    synthesisRef.current.cancel();
    setIsSpeaking(false);
  };

  useEffect(() => {
    // 初始化语音识别
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      // 使用非空断言，因为我们已经检查过SpeechRecognition存在
      recognitionRef.current!.continuous = false;
      recognitionRef.current!.interimResults = true;
      recognitionRef.current!.lang = 'zh-CN';

      recognitionRef.current!.onresult = (event: any) => {
        const result = event.results[event.results.length - 1];
        setTranscript(result[0].transcript);
        if (result.isFinal) {
          const finalTranscript = result[0].transcript;
          setHistory(prev => [...prev, { type: 'user', text: finalTranscript }]);
          handleVoiceCommand(finalTranscript);
          setTranscript('');
        }
      };

      recognitionRef.current!.onerror = (event: any) => {
        console.error('语音识别错误:', event.error);
        setIsListening(false);
      };

      recognitionRef.current!.onend = () => {
        setIsListening(false);
      };
    }

    // 清理语音识别实例
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  return (
    <div className="fixed bottom-8 right-8 z-50">
      {/* 错误提示 */}
      {error && (
        <div className="mb-4 max-w-xs bg-red-900 bg-opacity-70 border border-red-700 rounded-lg p-3 shadow-xl">
          <p className="text-sm text-foreground">{error}</p>
        </div>
      )}

      {/* 语音历史记录 */}
      {history.length > 0 && (
        <div className="mb-4 max-w-xs bg-gray-800 border border-gray-700 rounded-lg p-4 shadow-xl">
          <h3 className="text-sm font-semibold text-gray-300 dark:text-gray-200 mb-2">语音历史</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {history.slice(-5).map((item, index) => (
              <div key={index} className={`text-sm ${item.type === 'user' ? 'text-blue-400' : 'text-green-400'} flex flex-col`}>
                <div>
                  <span className="font-medium">{item.type === 'user' ? '你: ' : '助手: '}</span>
                  {item.text}
                </div>
                {/* 只对助手的响应显示反馈按钮 */}
                {item.type === 'assistant' && (
                  <div className="mt-1 flex space-x-2">
                    <button
                      onClick={async () => submitFeedback(item.text, 'positive')}
                      className="text-xs text-green-500 hover:text-green-400 flex items-center"
                      title="有用"
                    >
                      👍
                    </button>
                    <button
                      onClick={async () => submitFeedback(item.text, 'negative')}
                      className="text-xs text-red-500 hover:text-red-400 flex items-center"
                      title="没用"
                    >
                      👎
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 正在识别的文本 */}
      {isListening && transcript && (
        <div className="mb-4 max-w-xs bg-blue-900 bg-opacity-70 border border-blue-700 rounded-lg p-3 shadow-xl">
          <p className="text-sm text-foreground">正在识别: {transcript}</p>
        </div>
      )}

      {/* 语音控制按钮 */}
      <div className="flex space-x-3">
        {/* 加载状态 */}
        {isLoading ?
(
          <Button
            disabled
            className="bg-gray-600 text-foreground"
            size="icon"
          >
            <Loader2 className="h-6 w-6 animate-spin" />
          </Button>
        ) :
isSpeaking ?
(
          <Button
            onClick={stopSpeaking}
            className="bg-red-600 hover:bg-red-700 text-foreground"
            size="icon"
          >
            <VolumeX className="h-6 w-6" />
          </Button>
        ) :
(
          <Button
            onClick={toggleListening}
            className={isListening ? 'bg-red-600 hover:bg-red-700 text-foreground' : 'bg-blue-600 hover:bg-blue-700 text-foreground'}
            size="icon"
          >
            {isListening ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
          </Button>
        )}
      </div>
    </div>
  );
};

export default VoiceAssistant;
