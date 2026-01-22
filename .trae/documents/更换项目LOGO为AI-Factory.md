# 更换项目LOGO为AI-Factory

## 目标
将项目LOGO从当前的Brain图标+"智能系统"文字更换为新的AI-Factory LOGO，符合提供的设计图要求。

## 计划步骤

1. **创建新LOGO组件**
   - 创建一个新的SVG组件，实现设计图中的三个蓝色人形组成三角形的LOGO
   - 确保LOGO可以水平显示
   - 适配不同尺寸和颜色需求

2. **更新Sidebar组件**
   - 替换当前的Brain图标为新创建的LOGO
   - 将文字从"智能系统"改为"AI-Factory"
   - 确保LOGO在折叠和展开状态下都能正常显示
   - 保持现有的动画和交互效果

3. **优化LOGO显示效果**
   - 确保LOGO与背景的对比度合适
   - 调整文字字体和大小，使其更符合项目风格
   - 确保LOGO在深色和浅色主题下都能正常显示

4. **测试和验证**
   - 检查所有页面是否正常显示新LOGO
   - 确保折叠/展开功能正常工作
   - 验证响应式设计是否符合要求

## 预期效果
- 统一的AI-Factory品牌标识
- 清晰的水平布局LOGO
- 适配所有主题和状态
- 保持现有的视觉风格和交互体验

## 文件修改
- `src/components/AIFFactoryLogo.tsx` - 新LOGO组件
- `src/components/layout/Sidebar.tsx` - 更新LOGO和文字

## 技术实现
- 使用React和SVG创建可复用的LOGO组件
- 利用Tailwind CSS进行样式管理
- 保持与现有代码的兼容性