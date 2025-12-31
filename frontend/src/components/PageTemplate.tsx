import React from 'react'
import { Button } from './ui/button'

interface PageTemplateProps {
  title: string
  description?: string
  actions?: React.ReactNode[]
  children: React.ReactNode
}

export function PageTemplate({ title, description, actions = [], children }: PageTemplateProps) {
  return (
    <div className="space-y-6">
      {/* 页面标题和操作区域 */}
      <div className="bg-gradient-to-r from-tech-primary/10 to-tech-secondary/10 rounded-lg p-6 border border-tech-primary/20">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold gradient-text mb-1">{title}</h1>
            {description && (
              <p className="text-gray-300 text-sm md:text-base">{description}</p>
            )}
          </div>
          
          {/* 操作按钮 */}
          {actions.length > 0 && (
            <div className="flex flex-wrap gap-3">
              {actions.map((action, index) => (
                <React.Fragment key={index}>{action}</React.Fragment>
              ))}
            </div>
          )}
        </div>
      </div>
      
      {/* 页面内容 */}
      <div className="space-y-6">
        {children}
      </div>
    </div>
  )
}

// 页面卡片组件（用于内容区域）
export function PageCard({ children, className = '' }: { children: React.ReactNode, className?: string }) {
  return (
    <div className={`glass-effect rounded-lg p-6 border border-tech-light/50 ${className}`}>
      {children}
    </div>
  )
}

// 页面网格布局组件
export function PageGrid({ children, columns = '1 md:2 lg:3' }: { children: React.ReactNode, columns?: string }) {
  const colClasses = {
    '1': 'grid grid-cols-1 gap-6',
    '1 md:2': 'grid grid-cols-1 md:grid-cols-2 gap-6',
    '1 md:2 lg:3': 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6',
    '1 md:2 lg:4': 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6',
    '2 md:4': 'grid grid-cols-2 md:grid-cols-4 gap-6',
    '3 md:6': 'grid grid-cols-3 md:grid-cols-6 gap-6'
  }[columns] || 'grid grid-cols-1 gap-6'
  
  return (
    <div className={colClasses}>
      {children}
    </div>
  )
}