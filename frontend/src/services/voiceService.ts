/**
 * 语音服务类
 * 实现语音识别（ASR）和语音合成（TTS）功能
 */

export interface VoiceServiceCallbacks {
  onResult: (text: string, confidence?: number) => void;
  onError: (error: string) => void;
  onStatusChange: (isListening: boolean) => void;
  onSpeakingChange?: (isSpeaking: boolean) => void;
  onInterimResult?: (text: string) => void;
}

export interface SpeakOptions {
  voice?: SpeechSynthesisVoice;
  rate?: number;
  pitch?: number;
  volume?: number;
  emotion?: 'neutral' | 'happy' | 'sad' | 'angry' | 'fearful' | 'surprised';
  onProgress?: (charIndex: number, charLength: number) => void;
}

export class VoiceService {
  private recognition: SpeechRecognition | null = null;
  private readonly synthesis: SpeechSynthesis;
  private isListening = false;
  private isSpeaking = false;
  private isPaused = false;
  private readonly callbacks: VoiceServiceCallbacks;
  private currentUtterance: SpeechSynthesisUtterance | null = null;
  private currentUtteranceOptions: SpeakOptions | null = null;
  private utteranceQueue: { text: string; options: SpeakOptions }[] = [];
  private currentLanguage = 'zh-CN';
  private _isInitialized = false;
  private contextHistory: string[] = [];
  private maxContextHistory = 5;

  constructor(callbacks: VoiceServiceCallbacks) {
    this.synthesis = window.speechSynthesis;
    this.callbacks = callbacks;
    this.initRecognition();
    this.initSynthesis();
  }

  /**
   * 初始化语音识别
   */
  private initRecognition(): void {
    // 获取SpeechRecognition API（兼容不同浏览器）
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const SpeechGrammarList = (window as any).SpeechGrammarList || (window as any).webkitSpeechGrammarList;

    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.lang = this.currentLanguage;
      this.recognition.continuous = true;
      this.recognition.interimResults = true; // 返回中间结果
      this.recognition.maxAlternatives = 3; // 返回多个结果，用于置信度判断

      // 设置事件处理
      this.recognition.onresult = (event: SpeechRecognitionEvent) => {
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          const text = result[0].transcript;
          const { confidence } = result[0];

          if (result.isFinal) {
            // 最终结果
            this.callbacks.onResult(text, confidence);
            // 添加到上下文历史
            this.addToContextHistory(text);
          } else {
            // 中间结果
            this.callbacks.onInterimResult?.(text);
          }
        }
      };

      this.recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        let errorMessage = '语音识别错误';
        switch (event.error) {
          case 'no-speech':
            errorMessage = '未检测到语音';
            break;
          case 'audio-capture':
            errorMessage = '无法访问麦克风';
            break;
          case 'not-allowed':
            errorMessage = '麦克风权限被拒绝';
            break;
          case 'network':
            errorMessage = '网络错误';
            break;
          case 'service-not-allowed':
            errorMessage = '语音服务不可用';
            break;
          case 'bad-grammar':
          case 'language-not-supported':
            errorMessage = '不支持的语言或语法错误';
            break;
        }
        this.callbacks.onError(errorMessage);
      };

      this.recognition.onstart = () => {
        this.isListening = true;
        this.callbacks.onStatusChange(true);
      };

      this.recognition.onend = () => {
        this.isListening = false;
        this.callbacks.onStatusChange(false);
        // 如果是意外结束，自动重启
        if (this.isListening) {
          setTimeout(() => this.startListening(), 500);
        }
      };

      this.recognition.onsoundstart = () => {
        console.log('开始听到声音');
      };

      this.recognition.onsoundend = () => {
        console.log('声音结束');
      };

      this._isInitialized = true;
    } else {
      this.callbacks.onError('您的浏览器不支持语音识别功能');
    }
  }

  /**
   * 初始化语音合成
   */
  private initSynthesis(): void {
    this.synthesis.onvoiceschanged = () => {
      // 语音列表加载完成时触发
      console.log('可用语音列表已更新');
    };
  }

  /**
   * 开始监听语音
   */
  startListening(): void {
    if (!this.recognition) {
      this.callbacks.onError('语音识别未初始化');
      return;
    }

    try {
      this.recognition.start();
      this.isListening = true;
    } catch (error) {
      this.callbacks.onError(`启动语音识别失败: ${error}`);
    }
  }

  /**
   * 停止监听语音
   */
  stopListening(): void {
    if (!this.recognition) {
 return;
}

    try {
      this.recognition.stop();
      this.isListening = false;
    } catch (error) {
      this.callbacks.onError(`停止语音识别失败: ${error}`);
    }
  }

  /**
   * 获取语音列表
   */
  getVoices(): SpeechSynthesisVoice[] {
    return this.synthesis.getVoices();
  }

  /**
   * 查找特定语言的语音
   */
  findVoiceByLang(lang: string): SpeechSynthesisVoice | undefined {
    return this.getVoices().find(voice => voice.lang === lang);
  }

  /**
   * 语音合成，将文本转换为语音
   * 支持队列管理
   */
  speak(text: string, options?: SpeakOptions): void {
    // 如果正在说话，添加到队列
    if (this.isSpeaking && !this.isPaused) {
      this.utteranceQueue.push({ text, options: options || {} });
      return;
    }

    this._speakInternal(text, options || {});
  }

  /**
   * 内部语音合成方法
   */
  private _speakInternal(text: string, options: SpeakOptions): void {
    const utterance = new SpeechSynthesisUtterance(text);

    // 根据情感调整参数
    const emotionParams = this.getEmotionParams(options.emotion || 'neutral');

    // 设置选项
    if (options.voice) {
      utterance.voice = options.voice;
    } else {
      // 默认使用中文语音
      const chineseVoice = this.findVoiceByLang('zh-CN') ||
                         this.findVoiceByLang('zh-Hans-CN') ||
                         this.getVoices()[0];
      utterance.voice = chineseVoice;
    }

    utterance.rate = options.rate || emotionParams.rate || 1.0;
    utterance.pitch = options.pitch || emotionParams.pitch || 1.0;
    utterance.volume = options.volume || emotionParams.volume || 1.0;

    // 保存当前语音实例和选项
    this.currentUtterance = utterance;
    this.currentUtteranceOptions = options;

    // 设置事件处理
    utterance.onstart = () => {
      this.isSpeaking = true;
      this.isPaused = false;
      this.callbacks.onSpeakingChange?.(true);
    };

    utterance.onend = () => {
      this.isSpeaking = false;
      this.isPaused = false;
      this.callbacks.onSpeakingChange?.(false);
      this.currentUtterance = null;
      this.currentUtteranceOptions = null;

      // 播放队列中的下一个
      if (this.utteranceQueue.length > 0) {
        const next = this.utteranceQueue.shift();
        if (next) {
          this._speakInternal(next.text, next.options);
        }
      }
    };

    utterance.onerror = event => {
      this.isSpeaking = false;
      this.isPaused = false;
      this.callbacks.onSpeakingChange?.(false);
      this.callbacks.onError(`语音合成错误: ${event.error}`);
      this.currentUtterance = null;
      this.currentUtteranceOptions = null;
    };

    // 进度监听（模拟实现）
    if (options.onProgress) {
      let charIndex = 0;
      const charLength = text.length;
      const progressInterval = setInterval(() => {
        if (this.isSpeaking && !this.isPaused && this.currentUtterance === utterance) {
          charIndex = Math.min(charIndex + Math.ceil(charLength / (text.length * 2)), charLength);
          options.onProgress!(charIndex, charLength);
          if (charIndex >= charLength) {
            clearInterval(progressInterval);
          }
        } else {
          clearInterval(progressInterval);
        }
      }, 100);
    }

    // 开始播放
    this.synthesis.speak(utterance);
  }

  /**
   * 根据情感获取语音参数
   */
  private getEmotionParams(emotion: string): Partial<SpeakOptions> {
    switch (emotion) {
      case 'happy':
        return { rate: 1.2, pitch: 1.3, volume: 1.0 };
      case 'sad':
        return { rate: 0.8, pitch: 0.7, volume: 0.8 };
      case 'angry':
        return { rate: 1.3, pitch: 0.5, volume: 1.0 };
      case 'fearful':
        return { rate: 0.9, pitch: 1.5, volume: 0.7 };
      case 'surprised':
        return { rate: 1.1, pitch: 1.4, volume: 0.9 };
      default:
        return { rate: 1.0, pitch: 1.0, volume: 1.0 };
    }
  }

  /**
   * 暂停语音合成
   */
  pauseSpeaking(): void {
    if (this.isSpeaking && !this.isPaused) {
      this.synthesis.pause();
      this.isPaused = true;
      this.callbacks.onSpeakingChange?.(false);
    }
  }

  /**
   * 恢复语音合成
   */
  resumeSpeaking(): void {
    if (this.isPaused) {
      this.synthesis.resume();
      this.isPaused = false;
      this.isSpeaking = true;
      this.callbacks.onSpeakingChange?.(true);
    }
  }

  /**
   * 停止语音合成
   */
  stopSpeaking(): void {
    this.synthesis.cancel();
    this.isSpeaking = false;
    this.isPaused = false;
    this.utteranceQueue = [];
    this.callbacks.onSpeakingChange?.(false);
    this.currentUtterance = null;
    this.currentUtteranceOptions = null;
  }

  /**
   * 清空语音队列
   */
  clearSpeakingQueue(): void {
    this.utteranceQueue = [];
  }

  /**
   * 获取语音队列长度
   */
  getSpeakingQueueLength(): number {
    return this.utteranceQueue.length;
  }

  /**
   * 获取当前是否暂停
   */
  isPausedSpeaking(): boolean {
    return this.isPaused;
  }

  /**
   * 获取当前监听状态
   */
  getIsListening(): boolean {
    return this.isListening;
  }

  /**
   * 获取当前说话状态
   */
  getIsSpeaking(): boolean {
    return this.isSpeaking;
  }

  /**
   * 设置识别语言
   */
  setLanguage(lang: string): void {
    if (this.currentLanguage !== lang) {
      this.currentLanguage = lang;
      if (this.recognition) {
        this.recognition.lang = lang;
      }
    }
  }

  /**
   * 获取当前语言
   */
  getCurrentLanguage(): string {
    return this.currentLanguage;
  }

  /**
   * 添加到上下文历史
   */
  private addToContextHistory(text: string): void {
    this.contextHistory.push(text);
    if (this.contextHistory.length > this.maxContextHistory) {
      this.contextHistory.shift();
    }
  }

  /**
   * 清除上下文历史
   */
  clearContextHistory(): void {
    this.contextHistory = [];
  }

  /**
   * 获取上下文历史
   */
  getContextHistory(): string[] {
    return [...this.contextHistory];
  }

  /**
   * 设置最大上下文历史长度
   */
  setMaxContextHistory(length: number): void {
    this.maxContextHistory = length;
    // 裁剪历史记录
    while (this.contextHistory.length > length) {
      this.contextHistory.shift();
    }
  }

  /**
   * 检查是否已初始化
   */
  isInitialized(): boolean {
    return this._isInitialized;
  }

  /**
   * 销毁语音服务
   */
  destroy(): void {
    this.stopListening();
    this.stopSpeaking();
    this.recognition = null;
    this.contextHistory = [];
  }
}

