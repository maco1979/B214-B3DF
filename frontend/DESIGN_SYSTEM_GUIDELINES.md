# 设计系统指南

本文档介绍了前端UI设计系统的规范和使用方法，旨在确保整个项目UI的一致性。

## 1. 字体系统

### 字体家族
- **主字体**: Inter (用于正文、标题)
- **等宽字体**: JetBrains Mono (用于代码、数字)

### 字体大小层级
| 类名 | 大小 (px) | CSS变量 |
|------|-----------|---------|
| text-xs | 12px | `--font-size-xs` |
| text-sm | 14px | `--font-size-sm` |
| text-base | 16px | `--font-size-base` (默认) |
| text-lg | 18px | `--font-size-lg` |
| text-xl | 20px | `--font-size-xl` |
| text-2xl | 24px | `--font-size-2xl` |
| text-3xl | 30px | `--font-size-3xl` |
| text-4xl | 36px | `--font-size-4xl` |
| text-5xl | 48px | `--font-size-5xl` |

### 字重
| 类名 | 字重值 | CSS变量 |
|------|--------|---------|
| font-thin | 100 | `--font-weight-thin` |
| font-light | 300 | `--font-weight-light` |
| font-normal | 400 | `--font-weight-normal` |
| font-medium | 500 | `--font-weight-medium` |
| font-semibold | 600 | `--font-weight-semibold` |
| font-bold | 700 | `--font-weight-bold` |
| font-extrabold | 800 | `--font-weight-extrabold` |

### 行高
| 类名 | 行高值 | CSS变量 |
|------|--------|---------|
| leading-tight | 1.25 | `--line-height-tight` |
| leading-snug | 1.375 | `--line-height-snug` |
| leading-normal | 1.5 | `--line-height-normal` |
| leading-relaxed | 1.625 | `--line-height-relaxed` |

## 2. 颜色系统

### 主题颜色
- `--primary`: 主色调
- `--secondary`: 次要色调
- `--muted`: 柔和色调
- `--destructive`: 错误/危险色调
- `--background`: 背景色
- `--foreground`: 前景色
- `--card`: 卡片背景色
- `--card-foreground`: 卡片前景色
- `--border`: 边框色
- `--input`: 输入框边框色
- `--ring`: 焦点环色

### 颜色工具类
- `.text-primary` - 主色文字
- `.text-secondary` - 次色文字
- `.text-muted` - 柔和色文字
- `.text-destructive` - 错误色文字
- `.bg-primary` - 主色背景
- `.bg-secondary` - 次色背景
- `.bg-muted` - 柔和色背景
- `.bg-destructive` - 错误色背景

## 3. 布局组件

### 卡片 (Card)
```jsx
<div className="card">
  <h3 className="title-md">卡片标题</h3>
  <p className="text-body">卡片内容</p>
</div>
```

### 按钮 (Button)
```jsx
<button className="btn btn-primary">主要按钮</button>
<button className="btn">普通按钮</button>
```

### 输入框 (Input)
```jsx
<input type="text" className="input" placeholder="输入文本" />
```

## 4. 布局工具类

### 弹性布局
- `.flex-center` - 居中对齐
- `.flex-between` - 两端对齐
- `.flex-column` - 垂直排列

### 间距
- `.m-auto` - 自动外边距
- `.mx-auto` - 水平居中外边距
- `.my-auto` - 垂直居中外边距
- `.p-section` - 区域内边距
- `.p-subsection` - 子区域内边距

### 阴影
- `.elevate-1` - 轻微阴影
- `.elevate-2` - 中等阴影
- `.elevate-3` - 强烈阴影

### 圆角
- `.rounded-sm` - 小圆角
- `.rounded-md` - 默认圆角
- `.rounded-lg` - 大圆角
- `.rounded-xl` - 特大圆角

## 5. 响应式设计

### 断点
- Mobile: `< 640px`
- Tablet: `>= 640px`
- Desktop: `>= 1024px`

### 显示控制
- `.show-mobile` - 仅在移动端显示
- `.show-tablet` - 仅在平板端显示
- `.show-desktop` - 仅在桌面端显示
- `.hide-mobile` - 在移动端隐藏
- `.hide-tablet` - 在平板端隐藏
- `.hide-desktop` - 在桌面端隐藏

## 6. 动效

### 过渡动画
- `.transition-fast` - 快速过渡 (0.15s)
- `.transition-normal` - 普通过渡 (0.25s)
- `.transition-slow` - 慢速过渡 (0.35s)

### 悬停效果
- `.hover-lift` - 悬停提升效果

## 7. 实践建议

1. **始终使用设计系统类**: 避免直接编写CSS，优先使用设计系统提供的类
2. **保持一致性**: 在相似组件间保持相同的视觉风格
3. **响应式优先**: 考虑不同屏幕尺寸下的显示效果
4. **可访问性**: 确保颜色对比度和交互元素大小满足可访问性标准
5. **语义化**: 使用有意义的类名和标签结构

## 8. 扩展

如果需要添加新的设计元素，请遵循现有模式并确保与整体设计系统保持一致。