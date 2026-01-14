#!/usr/bin/env python3
"""
CVE监控Web界面
基于Flask框架开发，用于查看CVE信息、资产影响分析和配置管理
"""

from flask import Flask, render_template, jsonify, request
import os
import json
import datetime
import logging
from cve_monitor import CVEMonitor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 配置
app.config['SECRET_KEY'] = 'your-secret-key'  # 用于会话管理

# CVE数据目录
CVE_DATA_DIR = 'cve_data'

# 创建CVEMonitor实例
cve_monitor = CVEMonitor()


def load_cve_data(search_query=None, year=None, status=None):
    """加载本地CVE数据，支持搜索和过滤"""
    cve_data = []
    
    if not os.path.exists(CVE_DATA_DIR):
        return cve_data
    
    # 遍历年份目录
    for year_dir in os.listdir(CVE_DATA_DIR):
        # 按年份过滤
        if year and year_dir != year:
            continue
            
        year_path = os.path.join(CVE_DATA_DIR, year_dir)
        if os.path.isdir(year_path):
            # 遍历CVE文件
            for file_name in os.listdir(year_path):
                if file_name.endswith('.json') and file_name.startswith('CVE-'):
                    file_path = os.path.join(year_path, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            cve = json.load(f)
                            
                            # 按状态过滤
                            if status and cve.get('cveMetadata', {}).get('state') != status:
                                continue
                                
                            # 按搜索查询过滤
                            if search_query:
                                search_lower = search_query.lower()
                                # 检查CVE ID、状态、发布日期等字段
                                cve_id = cve.get('cveMetadata', {}).get('cveId', '').lower()
                                cve_state = cve.get('cveMetadata', {}).get('state', '').lower()
                                cve_date = cve.get('cveMetadata', {}).get('datePublished', '').lower()
                                
                                # 检查是否匹配搜索条件
                                if (search_lower in cve_id or 
                                    search_lower in cve_state or 
                                    search_lower in cve_date):
                                    cve_data.append(cve)
                            else:
                                cve_data.append(cve)
                    except Exception as e:
                        logger.error(f"加载CVE数据失败: {file_path}, 错误: {e}")
    
    # 按发布日期降序排序
    cve_data.sort(key=lambda x: x.get('cveMetadata', {}).get('datePublished', ''), reverse=True)
    
    return cve_data


def get_cve_statistics():
    """获取CVE统计数据，用于图表展示"""
    cve_data = load_cve_data()
    
    # 按年份统计
    year_stats = {}
    # 按状态统计
    status_stats = {}
    # 按月份统计（近12个月）
    month_stats = {}
    
    for cve in cve_data:
        # 年份统计
        year = cve.get('cveMetadata', {}).get('cveId', '').split('-')[1] if '-' in cve.get('cveMetadata', {}).get('cveId', '') else 'Unknown'
        year_stats[year] = year_stats.get(year, 0) + 1
        
        # 状态统计
        status = cve.get('cveMetadata', {}).get('state', 'Unknown')
        status_stats[status] = status_stats.get(status, 0) + 1
        
        # 月份统计
        date_published = cve.get('cveMetadata', {}).get('datePublished', '')
        if date_published:
            month = date_published[:7]  # 格式：YYYY-MM
            month_stats[month] = month_stats.get(month, 0) + 1
    
    # 处理月份统计，确保最近12个月都有数据
    import datetime
    current_date = datetime.datetime.now()
    last_12_months = []
    
    for i in range(11, -1, -1):
        month_date = current_date - datetime.timedelta(days=i*30)
        month_str = month_date.strftime('%Y-%m')
        last_12_months.append(month_str)
        if month_str not in month_stats:
            month_stats[month_str] = 0
    
    # 按月份排序
    sorted_month_stats = {month: month_stats[month] for month in last_12_months}
    
    return {
        'total_cves': len(cve_data),
        'year_stats': year_stats,
        'status_stats': status_stats,
        'month_stats': sorted_month_stats
    }


def load_delta_data():
    """加载delta.json数据"""
    delta_data = []
    
    if not os.path.exists(CVE_DATA_DIR):
        return delta_data
    
    # 遍历目录，找到所有delta文件
    for file_name in os.listdir(CVE_DATA_DIR):
        if file_name.startswith('delta_') and file_name.endswith('.json'):
            file_path = os.path.join(CVE_DATA_DIR, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    delta = json.load(f)
                    delta['file_name'] = file_name
                    delta_data.append(delta)
            except Exception as e:
                logger.error(f"加载delta数据失败: {file_path}, 错误: {e}")
    
    # 按时间倒序排序
    delta_data.sort(key=lambda x: x.get('fetchTime', ''), reverse=True)
    return delta_data


# 路由定义

@app.route('/')
def index():
    """首页"""
    # 获取统计数据
    stats = get_cve_statistics()
    return render_template('index.html', stats=stats)


@app.route('/dashboard')
def dashboard():
    """统计仪表盘"""
    stats = get_cve_statistics()
    return render_template('dashboard.html', stats=stats)


@app.route('/cves')
def cve_list():
    """CVE列表，支持搜索和过滤"""
    # 获取请求参数
    search_query = request.args.get('search', None)
    year = request.args.get('year', None)
    status = request.args.get('status', None)
    
    # 加载过滤后的CVE数据
    cve_data = load_cve_data(search_query, year, status)
    
    # 获取可用的年份和状态选项
    years = set()
    statuses = set()
    
    # 遍历所有CVE数据获取年份和状态选项（不考虑当前过滤条件）
    all_cves = load_cve_data()
    for cve in all_cves:
        cve_year = cve.get('cveMetadata', {}).get('cveId', '').split('-')[1] if '-' in cve.get('cveMetadata', {}).get('cveId', '') else 'Unknown'
        years.add(cve_year)
        cve_status = cve.get('cveMetadata', {}).get('state', 'Unknown')
        statuses.add(cve_status)
    
    return render_template('cve_list.html', 
                           cves=cve_data, 
                           search_query=search_query, 
                           year=year, 
                           status=status,
                           years=sorted(years, reverse=True),
                           statuses=sorted(statuses))


@app.route('/cve/<cve_id>')
def cve_detail(cve_id):
    """CVE详情"""
    # 查找CVE数据
    cve_data = None
    year = cve_id.split('-')[1] if '-' in cve_id else '2025'
    file_path = os.path.join(CVE_DATA_DIR, year, f'{cve_id}.json')
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cve_data = json.load(f)
        except Exception as e:
            logger.error(f"加载CVE详情失败: {file_path}, 错误: {e}")
    
    return render_template('cve_detail.html', cve=cve_data)


@app.route('/deltas')
def delta_list():
    """Delta列表"""
    delta_data = load_delta_data()
    return render_template('delta_list.html', deltas=delta_data)


@app.route('/assets')
def asset_list():
    """资产列表"""
    # 加载资产数据
    assets = cve_monitor._load_assets()
    return render_template('asset_list.html', assets=assets)


@app.route('/impact')
def impact_analysis():
    """资产影响分析"""
    # 这里可以实现更复杂的影响分析逻辑
    assets = cve_monitor._load_assets()
    cve_data = load_cve_data()
    
    # 简单的影响分析示例
    impact_data = []
    for cve in cve_data:
        affected_assets = cve_monitor.check_asset_impact(cve)
        if affected_assets:
            impact_data.append({
                'cve': cve,
                'affected_assets': affected_assets
            })
    
    return render_template('impact_analysis.html', impact_data=impact_data)


@app.route('/api/cves', methods=['GET'])
def api_cve_list():
    """CVE列表API"""
    cve_data = load_cve_data()
    return jsonify(cve_data)


@app.route('/api/deltas', methods=['GET'])
def api_delta_list():
    """Delta列表API"""
    delta_data = load_delta_data()
    return jsonify(delta_data)


@app.route('/api/run-monitor', methods=['POST'])
def api_run_monitor():
    """运行CVE监控"""
    try:
        cve_monitor.run()
        return jsonify({'status': 'success', 'message': 'CVE监控已启动'})
    except Exception as e:
        logger.error(f"运行CVE监控失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/clear-cache', methods=['POST'])
def api_clear_cache():
    """清理缓存"""
    try:
        cve_monitor.clear_cache()
        return jsonify({'status': 'success', 'message': '缓存已清理'})
    except Exception as e:
        logger.error(f"清理缓存失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/cache-stats', methods=['GET'])
def api_cache_stats():
    """获取缓存统计"""
    stats = cve_monitor.show_cache_stats()
    return jsonify(stats)


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """获取CVE统计数据"""
    stats = get_cve_statistics()
    return jsonify(stats)


@app.route('/api/cves/search', methods=['GET'])
def api_cve_search():
    """搜索CVE数据"""
    # 获取请求参数
    search_query = request.args.get('search', None)
    year = request.args.get('year', None)
    status = request.args.get('status', None)
    
    # 加载过滤后的CVE数据
    cve_data = load_cve_data(search_query, year, status)
    return jsonify(cve_data)


if __name__ == '__main__':
    # 创建模板目录
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir, exist_ok=True)
        
        # 创建简单的HTML模板
        index_html = '''<!DOCTYPE html>
<html>
<head>
    <title>CVE监控系统</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            padding-top: 20px;
        }
        .card {
            margin-bottom: 20px;
        }
        .stat-card {
            margin-bottom: 20px;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-bottom: 30px;
        }
        .filter-form {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">CVE监控系统</h1>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        导航菜单
                    </div>
                    <div class="list-group list-group-flush">
                        <a href="/" class="list-group-item list-group-item-action active">首页</a>
                        <a href="/dashboard" class="list-group-item list-group-item-action">统计仪表盘</a>
                        <a href="/cves" class="list-group-item list-group-item-action">CVE列表</a>
                        <a href="/deltas" class="list-group-item list-group-item-action">Delta记录</a>
                        <a href="/assets" class="list-group-item list-group-item-action">资产列表</a>
                        <a href="/impact" class="list-group-item list-group-item-action">影响分析</a>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                <div class="card">
                    <div class="card-header">
                        系统概览
                    </div>
                    <div class="card-body">
                        <h5 class="card-title">欢迎使用CVE监控系统</h5>
                        <p class="card-text">本系统用于监控和管理CVE漏洞信息，支持资产影响分析和多渠道通知。</p>
                        <div class="mt-4">
                            <button id="run-monitor" class="btn btn-primary me-2">运行监控</button>
                            <button id="clear-cache" class="btn btn-secondary me-2">清理缓存</button>
                            <button id="show-stats" class="btn btn-info">缓存统计</button>
                        </div>
                        <div id="status-message" class="mt-3"></div>
                    </div>
                </div>
                
                <!-- 统计卡片 -->
                <div class="row mt-4">
                    <div class="col-md-3">
                        <div class="card stat-card bg-primary text-white">
                            <div class="card-body">
                                <h6 class="card-title">总CVE数量</h6>
                                <h3 class="card-text">{{ stats.total_cves }}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card bg-success text-white">
                            <div class="card-body">
                                <h6 class="card-title">状态分类</h6>
                                <h3 class="card-text">{{ stats.status_stats|length }}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card bg-info text-white">
                            <div class="card-body">
                                <h6 class="card-title">年份范围</h6>
                                <h3 class="card-text">{{ stats.year_stats|length }}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card bg-warning text-white">
                            <div class="card-body">
                                <h6 class="card-title">最近更新</h6>
                                <h3 class="card-text">{{ stats.month_stats|length }}个月</h3>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- 月份统计图表 -->
                <div class="card mt-4">
                    <div class="card-header">
                        近12个月CVE发布趋势
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="monthlyChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 运行监控按钮
        document.getElementById('run-monitor').addEventListener('click', function() {
            fetch('/api/run-monitor', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    const msgEl = document.getElementById('status-message');
                    msgEl.innerHTML = `<div class="alert alert-${data.status === 'success' ? 'success' : 'danger'}">${data.message}</div>`;
                });
        });
        
        // 清理缓存按钮
        document.getElementById('clear-cache').addEventListener('click', function() {
            fetch('/api/clear-cache', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    const msgEl = document.getElementById('status-message');
                    msgEl.innerHTML = `<div class="alert alert-${data.status === 'success' ? 'success' : 'danger'}">${data.message}</div>`;
                });
        });
        
        // 缓存统计按钮
        document.getElementById('show-stats').addEventListener('click', function() {
            fetch('/api/cache-stats')
                .then(response => response.json())
                .then(data => {
                    const msgEl = document.getElementById('status-message');
                    msgEl.innerHTML = `<div class="alert alert-info">缓存统计: ${JSON.stringify(data, null, 2)}</div>`;
                });
        });
        
        // 绘制月份统计图表
        window.onload = function() {
            // 月份数据
            const monthData = {{ stats.month_stats|tojson }};
            const labels = Object.keys(monthData);
            const values = Object.values(monthData);
            
            // 创建图表
            const ctx = document.getElementById('monthlyChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'CVE数量',
                        data: values,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        };
    </script>
</body>
</html>'''
        
        cve_list_html = '''<!DOCTYPE html>
<html>
<head>
    <title>CVE列表 - CVE监控系统</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {
            padding-top: 20px;
        }
        .card {
            margin-bottom: 20px;
        }
        .filter-form {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .filter-row {
            margin-bottom: 15px;
        }
        .no-results {
            text-align: center;
            padding: 30px;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">CVE列表</h1>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        导航菜单
                    </div>
                    <div class="list-group list-group-flush">
                        <a href="/" class="list-group-item list-group-item-action">首页</a>
                        <a href="/dashboard" class="list-group-item list-group-item-action">统计仪表盘</a>
                        <a href="/cves" class="list-group-item list-group-item-action active">CVE列表</a>
                        <a href="/deltas" class="list-group-item list-group-item-action">Delta记录</a>
                        <a href="/assets" class="list-group-item list-group-item-action">资产列表</a>
                        <a href="/impact" class="list-group-item list-group-item-action">影响分析</a>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                <div class="card">
                    <div class="card-header">
                        CVE记录列表
                    </div>
                    <div class="card-body">
                        <!-- 搜索和过滤表单 -->
                        <form method="GET" action="/cves" class="filter-form">
                            <div class="filter-row">
                                <div class="row">
                                    <div class="col-md-5">
                                        <input type="text" class="form-control" name="search" placeholder="搜索CVE ID、状态或日期" value="{{ search_query or '' }}">
                                    </div>
                                    <div class="col-md-3">
                                        <select class="form-select" name="year">
                                            <option value="">所有年份</option>
                                            {% for y in years %}
                                            <option value="{{ y }}" {% if year == y %}selected{% endif %}>{{ y }}</option>
                                            {% endfor %}
                                        </select>
                                    </div>
                                    <div class="col-md-3">
                                        <select class="form-select" name="status">
                                            <option value="">所有状态</option>
                                            {% for s in statuses %}
                                            <option value="{{ s }}" {% if status == s %}selected{% endif %}>{{ s }}</option>
                                            {% endfor %}
                                        </select>
                                    </div>
                                    <div class="col-md-1">
                                        <button type="submit" class="btn btn-primary">搜索</button>
                                    </div>
                                </div>
                            </div>
                        </form>
                        
                        <!-- 搜索结果统计 -->
                        <div class="mb-3">
                            <strong>搜索结果:</strong> 找到 {{ cves|length }} 个CVE记录
                        </div>
                        
                        <!-- CVE表格 -->
                        {% if cves %}
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>CVE ID</th>
                                        <th>状态</th>
                                        <th>发布日期</th>
                                        <th>操作</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for cve in cves %}
                                    <tr>
                                        <td>{{ cve.cveMetadata.cveId }}</td>
                                        <td>{{ cve.cveMetadata.state }}</td>
                                        <td>{{ cve.cveMetadata.datePublished }}</td>
                                        <td>
                                            <a href="/cve/{{ cve.cveMetadata.cveId }}" class="btn btn-sm btn-primary">查看详情</a>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                        <div class="no-results">
                            <h5>未找到匹配的CVE记录</h5>
                            <p>请尝试调整搜索条件</p>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''
        
        cve_detail_html = '''<!DOCTYPE html>
<html>
<head>
    <title>CVE详情 - CVE监控系统</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        .json-display {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: monospace;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">CVE详情</h1>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        导航菜单
                    </div>
                    <div class="list-group list-group-flush">
                        <a href="/" class="list-group-item list-group-item-action">首页</a>
                        <a href="/cves" class="list-group-item list-group-item-action">CVE列表</a>
                        <a href="/deltas" class="list-group-item list-group-item-action">Delta记录</a>
                        <a href="/assets" class="list-group-item list-group-item-action">资产列表</a>
                        <a href="/impact" class="list-group-item list-group-item-action">影响分析</a>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                {% if cve %}
                <div class="card mb-3">
                    <div class="card-header">
                        {{ cve.cveMetadata.cveId }}
                    </div>
                    <div class="card-body">
                        <h5 class="card-title">基本信息</h5>
                        <table class="table table-bordered">
                            <tr>
                                <th>状态</th>
                                <td>{{ cve.cveMetadata.state }}</td>
                            </tr>
                            <tr>
                                <th>分配机构</th>
                                <td>{{ cve.cveMetadata.assignerShortName }}</td>
                            </tr>
                            <tr>
                                <th>保留日期</th>
                                <td>{{ cve.cveMetadata.dateReserved }}</td>
                            </tr>
                            <tr>
                                <th>发布日期</th>
                                <td>{{ cve.cveMetadata.datePublished }}</td>
                            </tr>
                            <tr>
                                <th>更新日期</th>
                                <td>{{ cve.cveMetadata.dateUpdated }}</td>
                            </tr>
                        </table>
                    </div>
                </div>
                
                <div class="card mb-3">
                    <div class="card-header">
                        详细信息
                    </div>
                    <div class="card-body">
                        <div class="json-display">
                            {{ cve | tojson(indent=2) }}
                        </div>
                    </div>
                </div>
                {% else %}
                <div class="alert alert-danger">
                    未找到CVE详情
                </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''
        
        delta_list_html = '''<!DOCTYPE html>
<html>
<head>
    <title>Delta记录 - CVE监控系统</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Delta记录</h1>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        导航菜单
                    </div>
                    <div class="list-group list-group-flush">
                        <a href="/" class="list-group-item list-group-item-action">首页</a>
                        <a href="/cves" class="list-group-item list-group-item-action">CVE列表</a>
                        <a href="/deltas" class="list-group-item list-group-item-action active">Delta记录</a>
                        <a href="/assets" class="list-group-item list-group-item-action">资产列表</a>
                        <a href="/impact" class="list-group-item list-group-item-action">影响分析</a>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                <div class="card">
                    <div class="card-header">
                        Delta记录列表
                    </div>
                    <div class="card-body">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>文件名</th>
                                    <th>获取时间</th>
                                    <th>变更数量</th>
                                    <th>新增</th>
                                    <th>更新</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for delta in deltas %}
                                <tr>
                                    <td>{{ delta.file_name }}</td>
                                    <td>{{ delta.fetchTime }}</td>
                                    <td>{{ delta.numberOfChanges }}</td>
                                    <td>{{ delta.new | length }}</td>
                                    <td>{{ delta.updated | length }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''
        
        asset_list_html = '''<!DOCTYPE html>
<html>
<head>
    <title>资产列表 - CVE监控系统</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body>
    <div class="container">
        <h1 class="mb-4">资产列表</h1>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        导航菜单
                    </div>
                    <div class="list-group list-group-flush">
                        <a href="/" class="list-group-item list-group-item-action">首页</a>
                        <a href="/cves" class="list-group-item list-group-item-action">CVE列表</a>
                        <a href="/deltas" class="list-group-item list-group-item-action">Delta记录</a>
                        <a href="/assets" class="list-group-item list-group-item-action active">资产列表</a>
                        <a href="/impact" class="list-group-item list-group-item-action">影响分析</a>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                <div class="card">
                    <div class="card-header">
                        组织资产清单
                    </div>
                    <div class="card-body">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>名称</th>
                                    <th>厂商</th>
                                    <th>产品</th>
                                    <th>版本</th>
                                    <th>其他信息</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for asset in assets %}
                                <tr>
                                    <td>{{ asset.name }}</td>
                                    <td>{{ asset.vendor }}</td>
                                    <td>{{ asset.product }}</td>
                                    <td>{{ asset.version }}</td>
                                    <td>
                                        {% if asset.ip %}{{ asset.ip }}{% elif asset.url %}{{ asset.url }}{% endif %}
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''
        
        impact_html = '''<!DOCTYPE html>
<html>
<head>
    <title>影响分析 - CVE监控系统</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body>
    <div class="container">
        <h1 class="mb-4">资产影响分析</h1>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        导航菜单
                    </div>
                    <div class="list-group list-group-flush">
                        <a href="/" class="list-group-item list-group-item-action">首页</a>
                        <a href="/dashboard" class="list-group-item list-group-item-action">统计仪表盘</a>
                        <a href="/cves" class="list-group-item list-group-item-action">CVE列表</a>
                        <a href="/deltas" class="list-group-item list-group-item-action">Delta记录</a>
                        <a href="/assets" class="list-group-item list-group-item-action">资产列表</a>
                        <a href="/impact" class="list-group-item list-group-item-action active">影响分析</a>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                <div class="card">
                    <div class="card-header">
                        影响分析结果
                    </div>
                    <div class="card-body">
                        {% if impact_data %}
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>CVE ID</th>
                                    <th>影响资产数量</th>
                                    <th>影响资产</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for item in impact_data %}
                                <tr>
                                    <td><a href="/cve/{{ item.cve.cveMetadata.cveId }}">{{ item.cve.cveMetadata.cveId }}</a></td>
                                    <td>{{ item.affected_assets | length }}</td>
                                    <td>
                                        <ul>
                                            {% for asset in item.affected_assets %}
                                            <li>{{ asset }}</li>
                                            {% endfor %}
                                        </ul>
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                        {% else %}
                        <div class="alert alert-info">
                            未发现受影响的资产
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''
        
        # 统计仪表盘模板
        dashboard_html = '''<!DOCTYPE html>
<html>
<head>
    <title>统计仪表盘 - CVE监控系统</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            padding-top: 20px;
        }
        .card {
            margin-bottom: 20px;
        }
        .stat-card {
            margin-bottom: 20px;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-bottom: 30px;
        }
        .chart-section {
            margin-bottom: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">CVE统计仪表盘</h1>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        导航菜单
                    </div>
                    <div class="list-group list-group-flush">
                        <a href="/" class="list-group-item list-group-item-action">首页</a>
                        <a href="/dashboard" class="list-group-item list-group-item-action active">统计仪表盘</a>
                        <a href="/cves" class="list-group-item list-group-item-action">CVE列表</a>
                        <a href="/deltas" class="list-group-item list-group-item-action">Delta记录</a>
                        <a href="/assets" class="list-group-item list-group-item-action">资产列表</a>
                        <a href="/impact" class="list-group-item list-group-item-action">影响分析</a>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                <!-- 统计卡片 -->
                <div class="row mt-4">
                    <div class="col-md-3">
                        <div class="card stat-card bg-primary text-white">
                            <div class="card-body">
                                <h6 class="card-title">总CVE数量</h6>
                                <h3 class="card-text">{{ stats.total_cves }}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card bg-success text-white">
                            <div class="card-body">
                                <h6 class="card-title">状态分类</h6>
                                <h3 class="card-text">{{ stats.status_stats|length }}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card bg-info text-white">
                            <div class="card-body">
                                <h6 class="card-title">年份范围</h6>
                                <h3 class="card-text">{{ stats.year_stats|length }}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card bg-warning text-white">
                            <div class="card-body">
                                <h6 class="card-title">最近更新</h6>
                                <h3 class="card-text">{{ stats.month_stats|length }}个月</h3>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- 月份统计图表 -->
                <div class="card chart-section">
                    <div class="card-header">
                        近12个月CVE发布趋势
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="monthlyChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- 年份分布图表 -->
                <div class="card chart-section">
                    <div class="card-header">
                        CVE年份分布
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="yearChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- 状态分布图表 -->
                <div class="card chart-section">
                    <div class="card-header">
                        CVE状态分布
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="statusChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 绘制图表
        window.onload = function() {
            // 1. 月份统计图表
            const monthData = {{ stats.month_stats|tojson }};
            const monthLabels = Object.keys(monthData);
            const monthValues = Object.values(monthData);
            
            const monthlyCtx = document.getElementById('monthlyChart').getContext('2d');
            new Chart(monthlyCtx, {
                type: 'line',
                data: {
                    labels: monthLabels,
                    datasets: [{
                        label: 'CVE数量',
                        data: monthValues,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // 2. 年份分布图表
            const yearData = {{ stats.year_stats|tojson }};
            const yearLabels = Object.keys(yearData);
            const yearValues = Object.values(yearData);
            
            const yearCtx = document.getElementById('yearChart').getContext('2d');
            new Chart(yearCtx, {
                type: 'bar',
                data: {
                    labels: yearLabels,
                    datasets: [{
                        label: 'CVE数量',
                        data: yearValues,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // 3. 状态分布图表
            const statusData = {{ stats.status_stats|tojson }};
            const statusLabels = Object.keys(statusData);
            const statusValues = Object.values(statusData);
            
            const statusCtx = document.getElementById('statusChart').getContext('2d');
            new Chart(statusCtx, {
                type: 'pie',
                data: {
                    labels: statusLabels,
                    datasets: [{
                        data: statusValues,
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.6)',
                            'rgba(54, 162, 235, 0.6)',
                            'rgba(255, 206, 86, 0.6)',
                            'rgba(75, 192, 192, 0.6)',
                            'rgba(153, 102, 255, 0.6)',
                            'rgba(255, 159, 64, 0.6)'
                        ],
                        borderColor: [
                            'rgba(255, 99, 132, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)',
                            'rgba(153, 102, 255, 1)',
                            'rgba(255, 159, 64, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        };
    </script>
</body>
</html>'''
        
        # 写入模板文件
        template_files = {
            'index.html': index_html,
            'dashboard.html': dashboard_html,
            'cve_list.html': cve_list_html,
            'cve_detail.html': cve_detail_html,
            'delta_list.html': delta_list_html,
            'asset_list.html': asset_list_html,
            'impact_analysis.html': impact_html
        }
        
        for file_name, content in template_files.items():
            file_path = os.path.join(template_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)


if __name__ == '__main__':
    # 启动Flask应用
    logger.info("启动CVE监控Web界面...")
    logger.info("访问地址: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
