import React from 'react';

interface AIFFactoryLogoProps {
  size?: number;
  color?: string;
  className?: string;
  showText?: boolean;
}

/**
 * AI-Factory LOGO组件
 * 神经网络与文字融为一体的设计
 */
export const AIFFactoryLogo: React.FC<AIFFactoryLogoProps> = ({
  size = 24,
  color = '#0099ff',
  className = '',
  showText = false,
}) => (
    <div className={`flex items-center justify-center ${className}`}>
      <div className="flex items-center">
        {/* 神经网络结构 - 延伸到文字的设计 */}
        <svg
          width={size}
          height={size}
          viewBox="0 0 100 100"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* 定义渐变 */}
          <defs>
            <linearGradient id="neuralGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={color} stopOpacity="0.6" />
              <stop offset="50%" stopColor={color} stopOpacity="1" />
              <stop offset="100%" stopColor={color} stopOpacity="0.6" />
            </linearGradient>
          </defs>

          {/* 背景光晕 */}
          <circle cx="45" cy="50" r="45" fill={color} opacity="0.1" />

          {/* 神经网络 - 垂直多层结构 */}

          {/* 输入层 - 左侧 */}
          <g id="input-layer">
            <circle cx="20" cy="20" r="4" fill="url(#neuralGradient)" />
            <circle cx="20" cy="40" r="4" fill="url(#neuralGradient)" />
            <circle cx="20" cy="60" r="4" fill="url(#neuralGradient)" />
            <circle cx="20" cy="80" r="4" fill="url(#neuralGradient)" />
          </g>

          {/* 隐藏层1 - 中间偏左 */}
          <g id="hidden-layer-1">
            <circle cx="35" cy="25" r="3" fill="url(#neuralGradient)" opacity="0.8" />
            <circle cx="35" cy="45" r="3" fill="url(#neuralGradient)" opacity="0.8" />
            <circle cx="35" cy="65" r="3" fill="url(#neuralGradient)" opacity="0.8" />
          </g>

          {/* 隐藏层2 - 中间 */}
          <g id="hidden-layer-2">
            <circle cx="50" cy="30" r="3" fill="url(#neuralGradient)" opacity="0.7" />
            <circle cx="50" cy="50" r="3" fill="url(#neuralGradient)" opacity="0.7" />
            <circle cx="50" cy="70" r="3" fill="url(#neuralGradient)" opacity="0.7" />
          </g>

          {/* 输出层 - 右侧 */}
          <g id="output-layer">
            <circle cx="65" cy="25" r="3.5" fill="url(#neuralGradient)" />
            <circle cx="65" cy="50" r="3.5" fill="url(#neuralGradient)" />
            <circle cx="65" cy="75" r="3.5" fill="url(#neuralGradient)" />
          </g>

          {/* 密集连接线 - 典型神经网络特征 */}

          {/* 输入层到隐藏层1 - 密集连接 */}
          {[20, 40, 60, 80].map((y, i) => [25, 45, 65].map((hy, j) => (
                <path
                  key={`input-hidden1-${i}-${j}`}
                  d={`M24 ${y} L32 ${hy}`}
                  stroke={color}
                  strokeWidth="0.6"
                  strokeLinecap="round"
                  opacity="0.4"
                />
              )))}

          {/* 隐藏层1到隐藏层2 - 密集连接 */}
          {[25, 45, 65].map((y, i) => [30, 50, 70].map((hy, j) => (
                <path
                  key={`hidden1-hidden2-${i}-${j}`}
                  d={`M38 ${y} L47 ${hy}`}
                  stroke={color}
                  strokeWidth="0.6"
                  strokeLinecap="round"
                  opacity="0.3"
                />
              )))}

          {/* 隐藏层2到输出层 - 密集连接 */}
          {[30, 50, 70].map((y, i) => [25, 50, 75].map((hy, j) => (
                <path
                  key={`hidden2-output-${i}-${j}`}
                  d={`M53 ${y} L62 ${hy}`}
                  stroke={color}
                  strokeWidth="0.6"
                  strokeLinecap="round"
                  opacity="0.4"
                />
              )))}

          {/* 从输出层延伸到文字的连接线 */}
          <path d="M68.5 25 L75 30" stroke={color} strokeWidth="1" strokeLinecap="round" />
          <path d="M68.5 50 L75 40" stroke={color} strokeWidth="1" strokeLinecap="round" />
          <path d="M68.5 75 L75 50" stroke={color} strokeWidth="1" strokeLinecap="round" />
        </svg>

        {/* AI-Factory文字 - 与神经网络融为一体 */}
        {showText && (
          <div className="relative ml-1">
            <span className="text-2xl font-bold text-gray-800 dark:text-white">
              <tspan x="0" dy="0">AI-</tspan>
              <tspan x="0" dy="0">Factory</tspan>
            </span>
            {/* 文字下方的神经网络延伸效果 */}
            <svg width="100" height="20" viewBox="0 0 100 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M0 10 L15 10" stroke={color} strokeWidth="0.8" strokeLinecap="round" opacity="0.5" />
              <path d="M25 10 L40 10" stroke={color} strokeWidth="0.8" strokeLinecap="round" opacity="0.5" />
              <path d="M50 10 L65 10" stroke={color} strokeWidth="0.8" strokeLinecap="round" opacity="0.5" />
              <path d="M75 10 L90 10" stroke={color} strokeWidth="0.8" strokeLinecap="round" opacity="0.5" />
              <circle cx="15" cy="10" r="1.5" fill={color} opacity="0.6" />
              <circle cx="40" cy="10" r="1.5" fill={color} opacity="0.6" />
              <circle cx="65" cy="10" r="1.5" fill={color} opacity="0.6" />
              <circle cx="90" cy="10" r="1.5" fill={color} opacity="0.6" />
            </svg>
          </div>
        )}
      </div>
    </div>
  );

export default AIFFactoryLogo;
