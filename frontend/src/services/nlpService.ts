/**
 * NLP服务类
 * 实现自然语言处理功能，包括意图识别、语义理解、上下文管理等
 */

// 定义意图类型
export type IntentType =
  | 'TOGGLE_MASTER_CONTROL'
  | 'OPEN_CAMERA'
  | 'CLOSE_CAMERA'
  | 'PTZ_CONTROL'
  | 'START_AI'
  | 'STOP_AI'
  | 'QUERY_STATUS'
  | 'SET_PARAMETER'
  | 'GET_PARAMETER'
  | 'RESET_SYSTEM'
  | 'SAVE_CONFIG'
  | 'LOAD_CONFIG'
  | 'HELP'
  | 'GO_HOME'
  | 'GO_TO_PRESET'
  | 'SET_PRESET'
  | 'DELETE_PRESET'
  | 'UNKNOWN';

// 定义实体类型
export interface Entity {
  type: string;
  value: any;
  start: number;
  end: number;
}

// 定义情感类型
export type EmotionType = 'neutral' | 'happy' | 'sad' | 'angry' | 'fearful' | 'surprised';

// 定义意图识别结果
export interface IntentResult {
  intent: IntentType;
  confidence: number;
  entities: Entity[];
  text: string;
  context: Record<string, any>;
  emotion: EmotionType;
  emotionConfidence: number;
}

// 定义意图规则
export interface IntentRule {
  pattern: RegExp;
  intent: IntentType;
  extractEntities?: (text: string) => Entity[];
}

// 上下文管理
export interface Context {
  lastIntent?: IntentType;
  lastEntities?: Entity[];
  conversationHistory: string[];
  timestamp: number;
}

export class NLPService {
  private intentRules: IntentRule[];
  private context: Context;
  private readonly maxHistoryLength = 10;

  constructor() {
    this.intentRules = this.initializeIntentRules();
    this.context = this.initializeContext();
  }

  /**
   * 初始化上下文
   */
  private initializeContext(): Context {
    return {
      conversationHistory: [],
      timestamp: Date.now(),
    };
  }

  /**
   * 初始化意图规则
   */
  private initializeIntentRules(): IntentRule[] {
    return [
      {
        pattern: /(开启|启动|打开).*主控/g,
        intent: 'TOGGLE_MASTER_CONTROL',
        extractEntities: (text: string) => [
          {
            type: 'action',
            value: 'start',
            start: text.includes('开启') ?
text.indexOf('开启') :
                   text.includes('启动') ?
text.indexOf('启动') :
                   text.includes('打开') ? text.indexOf('打开') : 0,
            end: text.indexOf('主控') + 2,
          },
        ],
      },
      {
        pattern: /(关闭|停止).*主控/g,
        intent: 'TOGGLE_MASTER_CONTROL',
        extractEntities: (text: string) => [
          {
            type: 'action',
            value: 'stop',
            start: text.includes('关闭') ?
text.indexOf('关闭') :
                   text.includes('停止') ? text.indexOf('停止') : 0,
            end: text.indexOf('主控') + 2,
          },
        ],
      },
      {
        pattern: /(打开|开启).*摄像头/g,
        intent: 'OPEN_CAMERA',
        extractEntities: (text: string) => {
          const entities: Entity[] = [];

          // 提取摄像头索引
          const indexMatch = text.match(/(\d+)/g);
          if (indexMatch) {
            const index = parseInt(indexMatch[0]);
            entities.push({
              type: 'camera_index',
              value: index,
              start: text.indexOf(indexMatch[0]),
              end: text.indexOf(indexMatch[0]) + indexMatch[0].length,
            });
          }

          return entities;
        },
      },
      {
        pattern: /(关闭|停止).*摄像头/g,
        intent: 'CLOSE_CAMERA',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(左转|右转|上转|下转|放大|缩小)/g,
        intent: 'PTZ_CONTROL',
        extractEntities: (text: string) => {
          const entities: Entity[] = [];

          // 提取PTZ动作
          if (text.includes('左转')) {
            entities.push({
              type: 'ptz_action',
              value: 'left',
              start: text.indexOf('左转'),
              end: text.indexOf('左转') + 2,
            });
          } else if (text.includes('右转')) {
            entities.push({
              type: 'ptz_action',
              value: 'right',
              start: text.indexOf('右转'),
              end: text.indexOf('右转') + 2,
            });
          } else if (text.includes('上转')) {
            entities.push({
              type: 'ptz_action',
              value: 'up',
              start: text.indexOf('上转'),
              end: text.indexOf('上转') + 2,
            });
          } else if (text.includes('下转')) {
            entities.push({
              type: 'ptz_action',
              value: 'down',
              start: text.indexOf('下转'),
              end: text.indexOf('下转') + 2,
            });
          } else if (text.includes('放大')) {
            entities.push({
              type: 'ptz_action',
              value: 'zoom_in',
              start: text.indexOf('放大'),
              end: text.indexOf('放大') + 2,
            });
          } else if (text.includes('缩小')) {
            entities.push({
              type: 'ptz_action',
              value: 'zoom_out',
              start: text.indexOf('缩小'),
              end: text.indexOf('缩小') + 2,
            });
          }

          return entities;
        },
      },
      {
        pattern: /(开始|启动).*AI/g,
        intent: 'START_AI',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(停止|关闭).*AI/g,
        intent: 'STOP_AI',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(查询|查看|检查).*状态/g,
        intent: 'QUERY_STATUS',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(设置|调整).*参数/g,
        intent: 'SET_PARAMETER',
        extractEntities: (text: string) => {
          const entities: Entity[] = [];

          // 提取参数名和值
          const paramMatch = /(\w+)\s*(为|设置为|调到)\s*([^\s]+)/.exec(text);
          if (paramMatch) {
            entities.push({
              type: 'parameter_name',
              value: paramMatch[1],
              start: paramMatch.index,
              end: paramMatch.index + paramMatch[1].length,
            });
            entities.push({
              type: 'parameter_value',
              value: paramMatch[3],
              start: paramMatch.index + paramMatch[1].length + paramMatch[2].length,
              end: paramMatch.index + paramMatch[0].length,
            });
          }

          return entities;
        },
      },
      {
        pattern: /(获取|查看).*参数/g,
        intent: 'GET_PARAMETER',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(重置|重启).*系统/g,
        intent: 'RESET_SYSTEM',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(保存|存储).*配置/g,
        intent: 'SAVE_CONFIG',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(加载|导入).*配置/g,
        intent: 'LOAD_CONFIG',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(帮助|帮助信息|使用说明)/g,
        intent: 'HELP',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(回家|回到)原位/g,
        intent: 'GO_HOME',
        extractEntities: (text: string) => [],
      },
      {
        pattern: /(去|到|前往).*预设/g,
        intent: 'GO_TO_PRESET',
        extractEntities: (text: string) => {
          const entities: Entity[] = [];

          // 提取预设编号
          const presetMatch = /预设(\d+)/.exec(text);
          if (presetMatch) {
            entities.push({
              type: 'preset_id',
              value: parseInt(presetMatch[1]),
              start: presetMatch.index,
              end: presetMatch.index + presetMatch[0].length,
            });
          }

          return entities;
        },
      },
      {
        pattern: /(设置|保存).*预设/g,
        intent: 'SET_PRESET',
        extractEntities: (text: string) => {
          const entities: Entity[] = [];

          // 提取预设编号
          const presetMatch = /预设(\d+)/.exec(text);
          if (presetMatch) {
            entities.push({
              type: 'preset_id',
              value: parseInt(presetMatch[1]),
              start: presetMatch.index,
              end: presetMatch.index + presetMatch[0].length,
            });
          }

          return entities;
        },
      },
      {
        pattern: /(删除|移除).*预设/g,
        intent: 'DELETE_PRESET',
        extractEntities: (text: string) => {
          const entities: Entity[] = [];

          // 提取预设编号
          const presetMatch = /预设(\d+)/.exec(text);
          if (presetMatch) {
            entities.push({
              type: 'preset_id',
              value: parseInt(presetMatch[1]),
              start: presetMatch.index,
              end: presetMatch.index + presetMatch[0].length,
            });
          }

          return entities;
        },
      },
    ];
  }

  /**
   * 处理文本，识别意图和实体
   */
  processText(text: string): IntentResult {
    // 更新上下文
    this.updateContext(text);

    // 识别意图
    const result = this.recognizeIntent(text);

    // 更新上下文的最后意图和实体
    this.context.lastIntent = result.intent;
    this.context.lastEntities = result.entities;
    this.context.timestamp = Date.now();

    return result;
  }

  /**
   * 识别意图
   */
  private recognizeIntent(text: string): IntentResult {
    let bestIntent: IntentType = 'UNKNOWN';
    let bestConfidence = 0;
    let bestEntities: Entity[] = [];
    let bestMatch: RegExpExecArray | null = null;

    // 遍历所有规则，找到最佳匹配
    for (const rule of this.intentRules) {
      const match = rule.pattern.exec(text);
      if (match) {
        // 智能计算置信度
        const confidence = this.calculateConfidence(match, text, rule);

        if (confidence > bestConfidence) {
          bestConfidence = confidence;
          bestIntent = rule.intent;
          bestMatch = match;

          // 提取实体
          if (rule.extractEntities) {
            bestEntities = rule.extractEntities(text);
          } else {
            bestEntities = this.extractEntitiesFromMatch(match, text, rule.intent);
          }
        }
      }
    }

    // 情感分析
    const emotionResult = this.analyzeEmotion(text);

    return {
      intent: bestIntent,
      confidence: bestConfidence,
      entities: bestEntities,
      text,
      context: this.getContext(),
      emotion: emotionResult.emotion,
      emotionConfidence: emotionResult.confidence,
    };
  }

  /**
   * 智能计算置信度
   */
  private calculateConfidence(match: RegExpExecArray, text: string, rule: IntentRule): number {
    // 基础置信度
    let confidence = 0.7;

    // 根据匹配位置调整置信度
    if (match.index === 0) {
      confidence += 0.2; // 开头匹配更可信
    }

    // 根据匹配长度调整置信度
    const matchLength = match[0].length;
    const textLength = text.length;
    const matchRatio = matchLength / textLength;
    confidence += matchRatio * 0.1;

    /*
     * 根据规则优先级调整置信度
     * 这里可以根据规则的复杂度或特异性设置不同的优先级
     */

    return Math.min(confidence, 1.0);
  }

  /**
   * 从匹配结果中提取实体
   */
  private extractEntitiesFromMatch(match: RegExpExecArray, text: string, intent: IntentType): Entity[] {
    const entities: Entity[] = [];

    // 根据意图类型提取实体
    switch (intent) {
      case 'PTZ_CONTROL':
        // 提取PTZ动作
        const actionMap: Record<string, string> = {
          左转: 'left',
          右转: 'right',
          上转: 'up',
          下转: 'down',
          放大: 'zoom_in',
          缩小: 'zoom_out',
        };

        for (const [key, value] of Object.entries(actionMap)) {
          if (text.includes(key)) {
            entities.push({
              type: 'ptz_action',
              value,
              start: text.indexOf(key),
              end: text.indexOf(key) + key.length,
            });
          }
        }
        break;

      case 'GO_TO_PRESET':
      case 'SET_PRESET':
      case 'DELETE_PRESET':
        // 提取预设编号
        const presetMatch = /预设(\d+)/.exec(text);
        if (presetMatch) {
          entities.push({
            type: 'preset_id',
            value: parseInt(presetMatch[1]),
            start: presetMatch.index,
            end: presetMatch.index + presetMatch[0].length,
          });
        }
        break;
    }

    return entities;
  }

  /**
   * 情感分析
   */
  private analyzeEmotion(text: string): { emotion: EmotionType; confidence: number } {
    // 情感关键词列表
    const emotionKeywords: Record<EmotionType, string[]> = {
      happy: ['高兴', '开心', '快乐', '愉快', '棒', '好', '不错'],
      sad: ['难过', '悲伤', '伤心', '失望', '糟糕', '不好'],
      angry: ['生气', '愤怒', '恼火', '讨厌', '烦'],
      fearful: ['害怕', '恐惧', '担心', '紧张'],
      surprised: ['惊讶', '吃惊', '没想到', '哇'],
      neutral: [],
    };

    let bestEmotion: EmotionType = 'neutral';
    let bestScore = 0;

    // 统计每种情感的关键词出现次数
    for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
      let score = 0;
      for (const keyword of keywords) {
        if (text.includes(keyword)) {
          score++;
        }
      }

      if (score > bestScore) {
        bestScore = score;
        bestEmotion = emotion as EmotionType;
      }
    }

    // 计算置信度
    const confidence = bestScore > 0 ? Math.min(0.7 + (bestScore * 0.1), 1.0) : 0.5;

    return {
      emotion: bestEmotion,
      confidence,
    };
  }

  /**
   * 更新上下文
   */
  private updateContext(text: string): void {
    // 添加到对话历史
    this.context.conversationHistory.push(text);

    // 限制历史长度
    if (this.context.conversationHistory.length > this.maxHistoryLength) {
      this.context.conversationHistory.shift();
    }

    this.context.timestamp = Date.now();
  }

  /**
   * 获取上下文
   */
  getContext(): Record<string, any> {
    return {
      lastIntent: this.context.lastIntent,
      lastEntities: this.context.lastEntities,
      conversationHistory: [...this.context.conversationHistory],
      timestamp: this.context.timestamp,
    };
  }

  /**
   * 清除上下文
   */
  clearContext(): void {
    this.context = this.initializeContext();
  }

  /**
   * 提取实体
   */
  extractEntities(text: string): Entity[] {
    // 遍历所有规则，提取实体
    for (const rule of this.intentRules) {
      if (rule.pattern.test(text)) {
        if (rule.extractEntities) {
          return rule.extractEntities(text);
        }
      }
    }

    return [];
  }

  /**
   * 获取意图规则
   */
  getIntentRules(): IntentRule[] {
    return [...this.intentRules];
  }

  /**
   * 添加自定义意图规则
   */
  addIntentRule(rule: IntentRule): void {
    this.intentRules.push(rule);
  }

  /**
   * 更新意图规则
   */
  updateIntentRule(index: number, rule: IntentRule): void {
    // 更新现有规则
    if (index >= 0 && index < this.intentRules.length) {
      this.intentRules[index] = rule;
    }
  }

  /**
   * 处理复杂指令
   * 例如："打开客厅的灯并把空调调到26度"
   */
  processComplexCommand(text: string): IntentResult[] {
    // 简单实现：按连接词分割命令
    const commands = text.split(/(并|和|然后)/g).filter(cmd =>
      cmd.trim() && !/(并|和|然后)/.test(cmd),
    );

    return commands.map(cmd => this.processText(cmd.trim()));
  }
}

