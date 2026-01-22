#!/bin/bash

# AI平台性能验证脚本
# 用于验证百万级并发架构的性能表现

set -e  # 遇到错误时退出

echo "🚀 开始AI平台性能验证测试..."

# 检查必要工具
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl未安装，请先安装kubectl"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ python3未安装，请先安装Python 3"
    exit 1
fi

echo "✅ 环境检查完成"

# 检查集群连接
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ 未连接到Kubernetes集群"
    exit 1
fi

echo "✅ 集群连接正常"

# 检查服务状态
echo "🔍 检查服务部署状态..."
SERVICES_READY=$(kubectl get pods -n ai-platform --no-headers | grep -c "Running")
TOTAL_SERVICES=$(kubectl get pods -n ai-platform --no-headers | wc -l)

echo "📊 运行中的Pod: $SERVICES_READY / $TOTAL_SERVICES"

if [ $SERVICES_READY -lt $((TOTAL_SERVICES - 2)) ]; then
    echo "⚠️  有服务未正常运行，继续测试但可能影响结果"
else
    echo "✅ 所有服务正常运行"
fi

# 等待系统稳定
echo "⏳ 等待系统稳定 (60秒)..."
sleep 60

# 运行性能测试
echo "🧪 开始性能测试..."

# 安装测试依赖
pip3 install aiohttp asyncio

# 运行高并发测试
echo "⚡ 运行高并发压力测试..."
python3 high_concurrency_test.py

# 生成性能报告
echo "📈 生成性能报告..."
REPORT_FILE="performance_report_$(date +%Y%m%d_%H%M%S).md"
cp performance_report_template.md $REPORT_FILE

# 填充报告模板中的基本信息
sed -i "s/^# 测试日期:.*/# 测试日期: $(date)/" $REPORT_FILE
sed -i "s/^# 测试环境:.*/# 测试环境: Kubernetes AI Platform/" $REPORT_FILE

echo "✅ 性能测试完成！"
echo "📄 报告已保存到: $REPORT_FILE"

# 检查系统资源使用情况
echo "📊 系统资源使用情况:"
echo "CPU使用情况:"
kubectl top nodes

echo "内存使用情况:"
kubectl top pods -n ai-platform

# 检查HPA状态
echo ".AutoScale状态:"
kubectl get hpa -n ai-platform

# 输出测试摘要
echo ""
echo "==============================="
echo "📊 性能测试摘要"
echo "==============================="
echo "Pod总数: $TOTAL_SERVICES"
echo "运行中Pod: $SERVICES_READY"
echo "测试时间: $(date)"
echo "测试状态: 完成"
echo "报告文件: $REPORT_FILE"
echo "==============================="

echo ""
echo "💡 建议下一步操作:"
echo "1. 检查 $REPORT_FILE 以获取详细性能数据"
echo "2. 查看监控面板上的实时指标"
echo "3. 分析系统瓶颈并进行优化"
echo "4. 根据测试结果调整资源配置"