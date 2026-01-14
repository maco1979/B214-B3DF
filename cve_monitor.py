#!/usr/bin/env python3
"""
CVE监控自动化脚本
功能：
1. 定期获取delta.json文件
2. 分析新增和更新的CVE记录
3. 下载完整CVE详情
4. 支持与资产清单关联
5. 实现漏洞情报推送
"""

import os
import json
import time
import random
import logging
import requests
import argparse
from datetime import datetime
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cve_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 修复Windows终端GBK编码问题
try:
    import sys
    if sys.platform == 'win32':
        import win_unicode_console
        win_unicode_console.enable()
except ImportError:
    pass

# 配置常量
DELTA_URL = "https://raw.githubusercontent.com/CVEProject/cvelistV5/main/cves/delta.json"
CVE_BASE_URL = "https://raw.githubusercontent.com/CVEProject/cvelistV5/main/cves/{}/{}/{}.json"

class CVEMonitor:
    def __init__(self, asset_file: str = None, output_dir: str = "cve_data", config_file: str = None):
        self.asset_file = asset_file
        self.output_dir = output_dir
        self.assets = self._load_assets()
        self.config = self._load_config(config_file)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 配置缓存
        self._setup_cache()
    
    def _setup_cache(self):
        """配置HTTP请求缓存，支持分级缓存"""
        cache_config = self.config.get('cache', {})
        if cache_config.get('enabled', True):
            try:
                import requests_cache
                from requests_cache import CachedSession
                from requests_cache.backends.sqlite import SQLiteCache
                
                # 配置缓存路径
                cache_path = os.path.join(self.output_dir, 'http_cache')
                
                # 基础缓存配置
                ttl = cache_config.get('ttl', 3600)
                max_size = cache_config.get('max_size', 1000)
                stale_while_revalidate = cache_config.get('stale_while_revalidate', 300)
                
                # 分级缓存配置
                cache_levels = cache_config.get('cache_levels', 2)
                
                # 配置不同资源类型的缓存策略
                cache_strategies = {
                    'delta': {
                        'ttl': cache_config.get('delta_ttl', 300),  # delta.json频繁更新，缓存时间较短
                        'max_size': cache_config.get('delta_max_size', 100)
                    },
                    'cve_detail': {
                        'ttl': cache_config.get('cve_detail_ttl', 7200),  # CVE详情更新较慢，缓存时间较长
                        'max_size': cache_config.get('cve_detail_max_size', 1000)
                    },
                    'default': {
                        'ttl': ttl,
                        'max_size': max_size
                    }
                }
                
                # 创建缓存适配器
                cache_backend = SQLiteCache(
                    db_path=cache_path + '.sqlite',
                    expire_after=ttl,
                    stale_while_revalidate=stale_while_revalidate,
                    max_size=max_size
                )
                
                # 安装主缓存
                self.session = CachedSession(
                    backend=cache_backend,
                    allowable_codes=[200],
                    allowable_methods=['GET']
                )
                
                # 安装全局缓存
                requests_cache.install_cache(
                    backend=cache_backend,
                    allowable_codes=[200],
                    allowable_methods=['GET']
                )
                
                logger.info(f"缓存已配置，路径: {cache_path}, 过期时间: {ttl}秒, 最大大小: {max_size}, 分级: {cache_levels}")
                logger.info(f"缓存策略: delta={cache_strategies['delta']['ttl']}秒, cve_detail={cache_strategies['cve_detail']['ttl']}秒")
                
                # 执行缓存预热
                if cache_config.get('cache_warmup', False):
                    self._cache_warmup()
            except Exception as e:
                logger.error(f"配置缓存失败: {e}")
        else:
            # 禁用缓存
            try:
                import requests_cache
                requests_cache.uninstall_cache()
            except Exception:
                pass
            logger.info("缓存已禁用")
    
    def _cache_warmup(self):
        """缓存预热功能，预先加载常用数据"""
        logger.info("开始缓存预热...")
        
        try:
            # 预加载delta.json
            logger.info("预加载delta.json...")
            self.fetch_delta()
            
            # 预加载最近的CVE详情（如果有delta数据）
            delta_file = None
            for file_name in os.listdir(self.output_dir):
                if file_name.startswith('delta_') and file_name.endswith('.json'):
                    delta_file = os.path.join(self.output_dir, file_name)
                    break
            
            if delta_file:
                with open(delta_file, 'r', encoding='utf-8') as f:
                    delta_data = json.load(f)
                    
                # 预加载最近的CVE详情（最多预加载20个）
                recent_cves = []
                recent_cves.extend(delta_data.get('new', []))
                recent_cves.extend(delta_data.get('updated', []))
                
                logger.info(f"预加载最近的{min(20, len(recent_cves))}个CVE详情...")
                for i, cve_item in enumerate(recent_cves[:20]):
                    cve_id = cve_item.get('cveId')
                    if cve_id:
                        self.fetch_cve_details(cve_id)
        
        except Exception as e:
            logger.error(f"缓存预热失败: {e}")
        else:
            logger.info("缓存预热完成")
    
    def cache_warmup(self):
        """手动触发缓存预热"""
        self._cache_warmup()
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置值的合法性"""
        # 验证HTTP配置
        http_config = config.get('http', {})
        http_config['timeout'] = max(5, min(60, http_config.get('timeout', 15)))
        http_config['connect_timeout'] = max(1, min(30, http_config.get('connect_timeout', 5)))
        http_config['read_timeout'] = max(1, min(60, http_config.get('read_timeout', 10)))
        http_config['max_retries'] = max(1, min(10, http_config.get('max_retries', 3)))
        http_config['backoff_factor'] = max(0.1, min(5.0, http_config.get('backoff_factor', 0.5)))
        http_config['pool_connections'] = max(1, min(50, http_config.get('pool_connections', 10)))
        http_config['pool_maxsize'] = max(1, min(100, http_config.get('pool_maxsize', 20)))
        http_config['keep_alive'] = max(1, min(3600, http_config.get('keep_alive', 300)))
        http_config['compress'] = bool(http_config.get('compress', True))
        config['http'] = http_config
        
        # 验证性能配置
        perf_config = config.get('performance', {})
        perf_config['concurrent_requests'] = max(1, min(50, perf_config.get('concurrent_requests', 5)))
        perf_config['batch_size'] = max(1, min(100, perf_config.get('batch_size', 10)))
        perf_config['cache_ttl'] = max(300, min(86400, perf_config.get('cache_ttl', 3600)))
        perf_config['parallel_processing'] = bool(perf_config.get('parallel_processing', False))
        perf_config['max_workers'] = max(1, min(20, perf_config.get('max_workers', 4)))
        perf_config['memory_limit_mb'] = max(128, min(8192, perf_config.get('memory_limit_mb', 1024)))
        perf_config['auto_scaling'] = bool(perf_config.get('auto_scaling', True))
        config['performance'] = perf_config
        
        # 验证缓存配置
        cache_config = config.get('cache', {})
        cache_config['ttl'] = max(300, min(86400, cache_config.get('ttl', 3600)))
        cache_config['max_size'] = max(100, min(10000, cache_config.get('max_size', 1000)))
        cache_config['cache_warmup'] = bool(cache_config.get('cache_warmup', False))
        cache_config['cache_levels'] = max(1, min(5, cache_config.get('cache_levels', 2)))
        cache_config['stale_while_revalidate'] = max(0, min(3600, cache_config.get('stale_while_revalidate', 300)))
        cache_config['delta_ttl'] = max(60, min(3600, cache_config.get('delta_ttl', 300)))
        cache_config['delta_max_size'] = max(10, min(1000, cache_config.get('delta_max_size', 100)))
        cache_config['cve_detail_ttl'] = max(3600, min(24*3600, cache_config.get('cve_detail_ttl', 7200)))
        cache_config['cve_detail_max_size'] = max(100, min(5000, cache_config.get('cve_detail_max_size', 1000)))
        config['cache'] = cache_config
        
        # 验证日志级别
        logging_config = config.get('logging', {})
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_level = logging_config.get('level', 'INFO').upper()
        logging_config['level'] = log_level if log_level in valid_levels else 'INFO'
        logging_config['log_to_file'] = bool(logging_config.get('log_to_file', True))
        logging_config['log_to_console'] = bool(logging_config.get('log_to_console', True))
        logging_config['log_file_size_mb'] = max(1, min(100, logging_config.get('log_file_size_mb', 10)))
        logging_config['log_file_backup_count'] = max(1, min(20, logging_config.get('log_file_backup_count', 5)))
        config['logging'] = logging_config
        
        return config
    
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "smtp_user": "user@example.com",
                "smtp_password": "password",
                "sender": "cve-monitor@example.com",
                "recipients": ["admin@example.com"]
            },
            "http": {
                "timeout": 15,
                "connect_timeout": 5,
                "read_timeout": 10,
                "max_retries": 3,
                "backoff_factor": 0.5,
                "pool_connections": 10,
                "pool_maxsize": 20,
                "keep_alive": 300,
                "compress": True,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            "logging": {
                "level": "INFO",
                "file": "cve_monitor.log",
                "log_to_file": True,
                "log_to_console": True,
                "log_file_size_mb": 10,
                "log_file_backup_count": 5
            },
            "monitor": {
                "save_delta": True,
                "save_cve_details": True,
                "check_frequency": 3600,
                "auto_cleanup_old_data": True,
                "data_retention_days": 30
            },
            "performance": {
                "concurrent_requests": 5,
                "batch_size": 10,
                "cache_ttl": 3600,
                "parallel_processing": False,
                "max_workers": 4,
                "memory_limit_mb": 1024,
                "auto_scaling": True
            },
            "cache": {
                "enabled": True,
                "ttl": 3600,
                "max_size": 1000,
                "cache_warmup": False,
                "cache_levels": 2,
                "stale_while_revalidate": 300,
                "delta_ttl": 300,
                "delta_max_size": 100,
                "cve_detail_ttl": 7200,
                "cve_detail_max_size": 1000
            },
            "notifications": {
                "slack": {
                    "enabled": False,
                    "webhook_url": "https://hooks.slack.com/services/your/webhook/url"
                },
                "wechat": {
                    "enabled": False,
                    "webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your-key"
                },
                "webhook": {
                    "enabled": False,
                    "url": "https://example.com/webhook",
                    "method": "POST"
                },
                "dingtalk": {
                    "enabled": False,
                    "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=your-token"
                },
                "telegram": {
                    "enabled": False,
                    "bot_token": "your-telegram-bot-token",
                    "chat_id": "your-chat-id"
                },
                "teams": {
                    "enabled": False,
                    "webhook_url": "https://your-teams-webhook-url"
                },
                "discord": {
                    "enabled": False,
                    "webhook_url": "https://discord.com/api/webhooks/your-webhook-url"
                }
            }
        }
        
        if not config_file or not os.path.exists(config_file):
            logger.info(f"配置文件不存在，使用默认配置: {config_file}")
            return self._validate_config(default_config)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
                # 递归合并配置
                def merge_dict(source, target):
                    for key, value in source.items():
                        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                            merge_dict(value, target[key])
                        else:
                            target[key] = value
                
                merge_dict(user_config, default_config)
                return self._validate_config(default_config)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._validate_config(default_config)
    
    def _load_assets(self) -> List[Dict[str, str]]:
        """加载组织资产清单"""
        if not self.asset_file or not os.path.exists(self.asset_file):
            logger.warning(f"资产清单文件不存在: {self.asset_file}")
            return []
        
        try:
            with open(self.asset_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载资产清单失败: {e}")
            return []
    
    def clear_cache(self):
        """清理HTTP请求缓存"""
        try:
            import requests_cache
            requests_cache.clear()
            logger.info("缓存已清理")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
    
    def show_cache_stats(self):
        """显示缓存统计信息"""
        try:
            import requests_cache
            from requests_cache.backends.sqlite import SQLiteCache
            
            stats = {
                'cache_enabled': True,
                'cache_backend': 'unknown',
                'cache_hits': 0,
                'cache_misses': 0,
                'total_requests': 0,
                'hit_rate': 0.0,
                'cache_size': 0
            }
            
            # 获取缓存对象
            if hasattr(requests_cache, 'get_cache'):
                cache = requests_cache.get_cache()
                stats['cache_backend'] = type(cache).__name__
                
                # 获取缓存统计
                if hasattr(cache, 'stats'):
                    # 处理SQLiteCache等具有stats属性的后端
                    cache_stats = cache.stats
                    stats['cache_hits'] = cache_stats.get('hits', 0)
                    stats['cache_misses'] = cache_stats.get('misses', 0)
                elif hasattr(cache, '_cache'):
                    # 处理其他类型的后端
                    cache_instance = cache._cache
                    if hasattr(cache_instance, 'get_stats'):
                        cache_stats = cache_instance.get_stats()
                        stats['cache_hits'] = cache_stats.get('hits', 0)
                        stats['cache_misses'] = cache_stats.get('misses', 0)
                    elif isinstance(cache_instance, SQLiteCache):
                        # 直接查询SQLite缓存大小
                        import sqlite3
                        cache_path = os.path.join(self.output_dir, 'http_cache.sqlite')
                        if os.path.exists(cache_path):
                            conn = sqlite3.connect(cache_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT COUNT(*) FROM responses")
                            stats['cache_size'] = cursor.fetchone()[0]
                            conn.close()
                
                # 计算总请求数和命中率
                stats['total_requests'] = stats['cache_hits'] + stats['cache_misses']
                if stats['total_requests'] > 0:
                    stats['hit_rate'] = round((stats['cache_hits'] / stats['total_requests']) * 100, 2)
                
                # 尝试获取缓存大小
                cache_path = os.path.join(self.output_dir, 'http_cache')
                if os.path.exists(cache_path + '.sqlite'):
                    stats['cache_size'] = os.path.getsize(cache_path + '.sqlite') // 1024  # KB
                elif os.path.exists(cache_path):
                    stats['cache_size'] = len(os.listdir(cache_path)) if os.path.isdir(cache_path) else 0
            else:
                # 缓存未安装
                stats['cache_enabled'] = False
            
            logger.info(f"缓存统计: {stats}")
            return stats
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {
                'cache_enabled': False,
                'error': str(e)
            }
    
    def _request_with_retry(self, url: str) -> requests.Response:
        """带重试机制的HTTP请求，使用缓存会话"""
        # 从配置中获取HTTP参数
        http_config = self.config.get('http', {})
        max_retries = http_config.get('max_retries', 3)
        timeout = http_config.get('timeout', 15)
        connect_timeout = http_config.get('connect_timeout', 5)
        read_timeout = http_config.get('read_timeout', 10)
        backoff_factor = http_config.get('backoff_factor', 0.5)
        user_agent = http_config.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # 优化重试策略：增加随机延迟，避免同时重试
        for retry in range(max_retries):
            try:
                # 优先使用缓存会话，如果不存在则使用requests库
                if hasattr(self, 'session') and self.session:
                    response = self.session.get(
                        url, 
                        timeout=(connect_timeout, read_timeout),
                        headers={
                            'User-Agent': user_agent
                        }
                    )
                else:
                    response = requests.get(
                        url, 
                        timeout=(connect_timeout, read_timeout),
                        headers={
                            'User-Agent': user_agent
                        }
                    )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if retry < max_retries - 1:
                    # 计算退避时间：backoff_factor * (2 ** retry) + 随机延迟
                    wait_time = backoff_factor * (2 ** retry) + (random.random() * 0.5)
                    logger.warning(f"请求失败，{retry+1}/{max_retries}，{wait_time:.1f}秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    raise
    
    def fetch_delta(self) -> Dict[str, Any]:
        """获取delta.json文件"""
        try:
            logger.info(f"正在获取delta.json: {DELTA_URL}")
            response = self._request_with_retry(DELTA_URL)
            delta_data = response.json()
            
            # 检查是否需要保存delta.json
            monitor_config = self.config.get('monitor', {})
            if monitor_config.get('save_delta', True):
                delta_path = os.path.join(self.output_dir, f"delta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(delta_path, 'w', encoding='utf-8') as f:
                    json.dump(delta_data, f, indent=2, ensure_ascii=False)
                logger.info(f"成功获取delta.json，保存到: {delta_path}")
            else:
                logger.info(f"成功获取delta.json，未保存到本地")
            
            return delta_data
        except Exception as e:
            logger.error(f"获取delta.json失败: {e}")
            return {}
    
    def fetch_cve_details(self, cve_id: str) -> Dict[str, Any]:
        """获取完整CVE详情"""
        # 解析CVE ID格式：CVE-2025-40978 -> 2025/40xxx/CVE-2025-40978.json
        try:
            parts = cve_id.split('-')
            year = parts[1]
            cve_number = parts[2]
            
            # 生成URL路径，例如：2025/40xxx/CVE-2025-40978.json
            if len(cve_number) >= 5:
                prefix = f"{cve_number[:2]}xxx"
            else:
                prefix = f"{cve_number[:1]}xxx"
            
            cve_url = CVE_BASE_URL.format(year, prefix, cve_id)
            logger.info(f"正在获取CVE详情: {cve_url}")
            
            response = self._request_with_retry(cve_url)
            cve_data = response.json()
            
            # 检查是否需要保存CVE详情
            monitor_config = self.config.get('monitor', {})
            if monitor_config.get('save_cve_details', True):
                cve_dir = os.path.join(self.output_dir, year)
                os.makedirs(cve_dir, exist_ok=True)
                
                cve_path = os.path.join(cve_dir, f"{cve_id}.json")
                with open(cve_path, 'w', encoding='utf-8') as f:
                    json.dump(cve_data, f, indent=2, ensure_ascii=False)
                logger.info(f"成功获取CVE详情，保存到: {cve_path}")
            else:
                logger.info(f"成功获取CVE详情，未保存到本地")
            
            return cve_data
        except Exception as e:
            logger.error(f"获取CVE详情失败: {e}")
            return {}
    
    def _version_in_range(self, asset_version: str, vuln_version: str) -> bool:
        """检查资产版本是否在漏洞版本范围内"""
        # 简单情况处理
        if not vuln_version or vuln_version.lower() == 'all versions' or vuln_version == '*':
            return True
        
        # 清理版本字符串
        vuln_version = vuln_version.strip()
        asset_version = asset_version.strip()
        
        # 精确匹配
        if vuln_version == asset_version:
            return True
        
        try:
            import re
            from packaging.version import Version, parse
            from packaging.specifiers import SpecifierSet
            
            # 1. 处理正则表达式匹配（如 /^1\.2\.\d+$/）
            if vuln_version.startswith('/') and vuln_version.endswith('/'):
                regex_pattern = vuln_version[1:-1]
                try:
                    if re.match(regex_pattern, asset_version):
                        return True
                except re.error:
                    logger.debug(f"无效的正则表达式: {regex_pattern}")
            
            # 2. 处理多版本号列表（如 "1.0.0, 1.1.0, 1.2.0"）
            if ',' in vuln_version:
                versions = [v.strip() for v in vuln_version.split(',')]
                for v in versions:
                    if self._version_in_range(asset_version, v):
                        return True
            
            # 解析资产版本
            asset_ver = parse(asset_version)
            
            # 3. 处理简单范围格式
            if vuln_version.startswith(('>=', '<=', '==', '!=', '>', '<', '~=')):
                # 使用packaging.specifiers处理范围
                try:
                    spec = SpecifierSet(vuln_version)
                    return asset_ver in spec
                except Exception:
                    pass
            
            # 4. 处理多条件范围，如 ">=1.0.0 <2.0.0"
            if ' ' in vuln_version:
                try:
                    spec = SpecifierSet(vuln_version)
                    return asset_ver in spec
                except Exception:
                    pass
            
            # 5. 处理方括号范围格式，如 "[1.0.0, 2.0.0)" 或 "(1.0.0, 2.0.0]"
            if (vuln_version.startswith('[') and vuln_version.endswith(']')) or \
               (vuln_version.startswith('(') and vuln_version.endswith(')')) or \
               (vuln_version.startswith('[') and vuln_version.endswith(')')) or \
               (vuln_version.startswith('(') and vuln_version.endswith(']')):
                try:
                    # 转换为packaging.specifiers格式
                    range_str = vuln_version
                    range_str = range_str.replace('[', '').replace(']', '')
                    range_str = range_str.replace('(', '').replace(')', '')
                    if ',' in range_str:
                        min_ver, max_ver = range_str.split(',')
                        min_ver = min_ver.strip()
                        max_ver = max_ver.strip()
                        
                        # 根据括号类型添加相应的比较符
                        if vuln_version.startswith('['):
                            min_op = '>='
                        else:
                            min_op = '>'
                        
                        if vuln_version.endswith(']'):
                            max_op = '<='
                        else:
                            max_op = '<'
                        
                        spec_str = f"{min_op}{min_ver} {max_op}{max_ver}"
                        spec = SpecifierSet(spec_str)
                        return asset_ver in spec
                except Exception:
                    pass
            
            # 6. 处理版本前缀匹配（如 1.2.x, 1.2.*）
            if vuln_version.endswith('.x') or vuln_version.endswith('.*'):
                prefix = vuln_version[:-2]
                try:
                    # 处理主版本前缀（如 1.x 或 1.*）
                    if '.' not in prefix:
                        vuln_spec = f">={prefix}, <{int(prefix) + 1}.0.0"
                    # 处理次版本前缀（如 1.2.x 或 1.2.*）
                    else:
                        major, minor = prefix.split('.')[:2]
                        vuln_spec = f">={prefix}, <{major}.{int(minor) + 1}.0"
                    spec = SpecifierSet(vuln_spec)
                    return asset_ver in spec
                except Exception:
                    pass
            
            # 7. 处理波浪号范围（如 ~1.2）
            if vuln_version.startswith('~'):
                try:
                    spec = SpecifierSet(vuln_version)
                    return asset_ver in spec
                except Exception:
                    pass
            
            # 8. 处理插入符号范围（如 ^1.2.3）
            if vuln_version.startswith('^'):
                try:
                    spec = SpecifierSet(vuln_version)
                    return asset_ver in spec
                except Exception:
                    pass
            
            # 9. 处理包含匹配（如 "1.2" 匹配 "1.2.3"）
            if vuln_version.lower() in asset_version.lower() or asset_version.lower() in vuln_version.lower():
                return True
            
            # 10. 处理带构建元数据的版本（如 1.0.0+build.123）
            # 提取核心版本号（去掉构建元数据）
            asset_core_ver = asset_version.split('+')[0]
            if asset_core_ver != asset_version:
                if self._version_in_range(asset_core_ver, vuln_version):
                    return True
            
            # 11. 尝试解析为语义化版本进行比较
            vuln_ver = parse(vuln_version)
            if isinstance(asset_ver, Version) and isinstance(vuln_ver, Version):
                # 主版本号匹配
                if asset_ver.major == vuln_ver.major:
                    # 次版本号匹配
                    if asset_ver.minor == vuln_ver.minor:
                        return True
                    # 主版本号相同，漏洞版本为0，匹配所有次版本
                    if vuln_ver.minor == 0:
                        return True
                    # 主版本号相同，次版本号小于等于资产版本（适用于漏洞影响所有后续版本）
                    if vuln_ver.minor <= asset_ver.minor:
                        return True
            
        except Exception as e:
            logger.debug(f"版本比较失败，使用备用逻辑: {e}")
            # 备用逻辑：简单字符串匹配和前缀匹配
            vuln_version = vuln_version.lower()
            asset_version = asset_version.lower()
            
            if vuln_version in asset_version or asset_version in vuln_version:
                return True
            
            # 备用前缀匹配：如 "1.2" 匹配 "1.2.3"
            if vuln_version.endswith('.'):
                if asset_version.startswith(vuln_version):
                    return True
            
        return False
    
    def check_asset_impact(self, cve_data: Dict[str, Any]) -> List[str]:
        """检查CVE是否影响组织资产"""
        if not self.assets:
            return []
        
        affected_assets = []
        try:
            # 获取CVE影响的产品和厂商
            cve_affected = []
            if 'containers' in cve_data and 'cna' in cve_data['containers']:
                cna_data = cve_data['containers']['cna']
                if 'affected' in cna_data:
                    for affected in cna_data['affected']:
                        cve_affected.append({
                            'vendor': affected.get('vendor', '').lower(),
                            'product': affected.get('product', '').lower(),
                            'versions': affected.get('versions', []),
                            'defaultStatus': affected.get('defaultStatus', 'unaffected').lower()
                        })
            
            # 检查与资产清单的匹配
            for asset in self.assets:
                asset_name = asset.get('name', 'Unknown')
                asset_vendor = asset.get('vendor', '').lower()
                asset_product = asset.get('product', '').lower()
                asset_version = asset.get('version', '').lower()
                
                for vuln in cve_affected:
                    # 厂商匹配：使用更精确的匹配策略
                    vendor_match = (
                        vuln['vendor'] == asset_vendor or  # 完全匹配优先
                        (vuln['vendor'] and vuln['vendor'] in asset_vendor) or  # 漏洞厂商在资产厂商中
                        (asset_vendor and asset_vendor in vuln['vendor'])  # 资产厂商在漏洞厂商中
                    )
                    
                    # 产品匹配：使用更精确的匹配策略
                    product_match = (
                        vuln['product'] == asset_product or  # 完全匹配优先
                        (vuln['product'] and vuln['product'] in asset_product) or  # 漏洞产品在资产产品中
                        (asset_product and asset_product in vuln['product'])  # 资产产品在漏洞产品中
                    )
                    
                    if vendor_match and product_match:
                        # 检查版本影响
                        version_affected = False
                        
                        # 如果默认状态是受影响，则所有版本都受影响
                        if vuln['defaultStatus'] == 'affected':
                            version_affected = True
                        else:
                            # 检查具体版本范围
                            for version_info in vuln['versions']:
                                status = version_info.get('status', 'unaffected').lower()
                                if status == 'affected':
                                    vuln_version = version_info.get('version', '').lower()
                                    if self._version_in_range(asset_version, vuln_version):
                                        version_affected = True
                                        break
                        
                        if version_affected:
                            affected_assets.append(f"{asset_name} ({asset_vendor} {asset_product} {asset_version})")
                            break
        except Exception as e:
            logger.error(f"资产影响分析失败: {e}")
        
        return affected_assets
    
    def send_email(self, subject: str, body: str):
        """发送邮件通知"""
        email_config = self.config.get('email', {})
        
        if not email_config.get('enabled', False):
            logger.debug("邮件通知已禁用")
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = email_config.get('sender')
            msg['To'] = ', '.join(email_config.get('recipients', []))
            msg['Subject'] = subject
            
            # 添加邮件正文
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(email_config.get('smtp_server'), email_config.get('smtp_port')) as server:
                server.starttls()
                server.login(email_config.get('smtp_user'), email_config.get('smtp_password'))
                server.send_message(msg)
            
            logger.info(f"邮件发送成功，收件人: {', '.join(email_config.get('recipients', []))}")
        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
    
    def send_slack(self, subject: str, body: str):
        """发送Slack通知"""
        slack_config = self.config.get('notifications', {}).get('slack', {})
        
        if not slack_config.get('enabled', False):
            logger.debug("Slack通知已禁用")
            return
        
        try:
            webhook_url = slack_config.get('webhook_url')
            if not webhook_url:
                logger.error("Slack Webhook URL未配置")
                return
            
            payload = {
                "text": f"*{subject}*\n\n{body}"
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info("Slack通知发送成功")
        except Exception as e:
            logger.error(f"发送Slack通知失败: {e}")
    
    def send_wechat(self, subject: str, body: str):
        """发送企业微信通知"""
        wechat_config = self.config.get('notifications', {}).get('wechat', {})
        
        if not wechat_config.get('enabled', False):
            logger.debug("企业微信通知已禁用")
            return
        
        try:
            webhook_url = wechat_config.get('webhook_url')
            if not webhook_url:
                logger.error("企业微信Webhook URL未配置")
                return
            
            payload = {
                "msgtype": "text",
                "text": {
                    "content": f"{subject}\n\n{body}"
                }
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info("企业微信通知发送成功")
        except Exception as e:
            logger.error(f"发送企业微信通知失败: {e}")
    
    def send_webhook(self, subject: str, body: str):
        """发送通用Webhook通知"""
        webhook_config = self.config.get('notifications', {}).get('webhook', {})
        
        if not webhook_config.get('enabled', False):
            logger.debug("Webhook通知已禁用")
            return
        
        try:
            webhook_url = webhook_config.get('url')
            method = webhook_config.get('method', 'POST').upper()
            if not webhook_url:
                logger.error("Webhook URL未配置")
                return
            
            payload = {
                "subject": subject,
                "body": body,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.request(
                method,
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info("Webhook通知发送成功")
        except Exception as e:
            logger.error(f"发送Webhook通知失败: {e}")
    
    def send_dingtalk(self, subject: str, body: str):
        """发送钉钉通知"""
        dingtalk_config = self.config.get('notifications', {}).get('dingtalk', {})
        
        if not dingtalk_config.get('enabled', False):
            logger.debug("钉钉通知已禁用")
            return
        
        try:
            webhook_url = dingtalk_config.get('webhook_url')
            if not webhook_url:
                logger.error("钉钉Webhook URL未配置")
                return
            
            payload = {
                "msgtype": "text",
                "text": {
                    "content": f"{subject}\n\n{body}"
                }
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info("钉钉通知发送成功")
        except Exception as e:
            logger.error(f"发送钉钉通知失败: {e}")
    
    def send_telegram(self, subject: str, body: str):
        """发送Telegram通知"""
        telegram_config = self.config.get('notifications', {}).get('telegram', {})
        
        if not telegram_config.get('enabled', False):
            logger.debug("Telegram通知已禁用")
            return
        
        try:
            bot_token = telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')
            
            if not bot_token or not chat_id:
                logger.error("Telegram Bot Token或Chat ID未配置")
                return
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": f"*{subject}*\n\n{body}",
                "parse_mode": "Markdown"
            }
            
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info("Telegram通知发送成功")
        except Exception as e:
            logger.error(f"发送Telegram通知失败: {e}")
    
    def send_teams(self, subject: str, body: str):
        """发送Microsoft Teams通知"""
        teams_config = self.config.get('notifications', {}).get('teams', {})
        
        if not teams_config.get('enabled', False):
            logger.debug("Microsoft Teams通知已禁用")
            return
        
        try:
            webhook_url = teams_config.get('webhook_url')
            if not webhook_url:
                logger.error("Microsoft Teams Webhook URL未配置")
                return
            
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "0076D7",
                "summary": subject,
                "sections": [{
                    "activityTitle": subject,
                    "text": body
                }]
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info("Microsoft Teams通知发送成功")
        except Exception as e:
            logger.error(f"发送Microsoft Teams通知失败: {e}")
    
    def send_discord(self, subject: str, body: str):
        """发送Discord通知"""
        discord_config = self.config.get('notifications', {}).get('discord', {})
        
        if not discord_config.get('enabled', False):
            logger.debug("Discord通知已禁用")
            return
        
        try:
            webhook_url = discord_config.get('webhook_url')
            if not webhook_url:
                logger.error("Discord Webhook URL未配置")
                return
            
            payload = {
                "content": f"**{subject}**\n\n{body}"
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info("Discord通知发送成功")
        except Exception as e:
            logger.error(f"发送Discord通知失败: {e}")
    
    def send_notification(self, cve_info: Dict[str, Any]):
        """发送漏洞情报通知"""
        cve_id = cve_info.get('cveMetadata', {}).get('cveId', 'Unknown')
        affected_assets = cve_info.get('affected_assets', [])
        
        notification = f"""[CVE漏洞通知]

CVE ID: {cve_id}
状态: {cve_info.get('cveMetadata', {}).get('state', 'Unknown')}
发布日期: {cve_info.get('cveMetadata', {}).get('datePublished', 'Unknown')}

影响评估:
{'影响组织资产: ' + ', '.join(affected_assets) if affected_assets else '未发现组织资产受影响'}

详细信息: https://www.cve.org/CVERecord?id={cve_id}
"""
        
        logger.info(f"发送通知:\n{notification}")
        
        subject = f"[CVE警报] {cve_id} - {'影响组织资产' if affected_assets else '新CVE漏洞'}"
        
        # 发送邮件通知
        self.send_email(subject, notification)
        
        # 发送Slack通知
        self.send_slack(subject, notification)
        
        # 发送企业微信通知
        self.send_wechat(subject, notification)
        
        # 发送通用Webhook通知
        self.send_webhook(subject, notification)
        
        # 发送钉钉通知
        self.send_dingtalk(subject, notification)
        
        # 发送Telegram通知
        self.send_telegram(subject, notification)
        
        # 发送Microsoft Teams通知
        self.send_teams(subject, notification)
        
        # 发送Discord通知
        self.send_discord(subject, notification)
    
    def process_delta(self, delta_data: Dict[str, Any]):
        """处理delta.json数据"""
        if not delta_data:
            return
        
        # 处理新增CVE
        new_cves = delta_data.get('new', [])
        if new_cves:
            logger.info(f"发现{len(new_cves)}个新增CVE")
            for cve in new_cves:
                self._process_cve(cve, "新增")
        
        # 处理更新CVE
        updated_cves = delta_data.get('updated', [])
        if updated_cves:
            logger.info(f"发现{len(updated_cves)}个更新CVE")
            for cve in updated_cves:
                self._process_cve(cve, "更新")
    
    def _process_cve(self, cve_item: Dict[str, str], change_type: str):
        """处理单个CVE记录"""
        cve_id = cve_item.get('cveId')
        if not cve_id:
            logger.warning(f"无效的CVE记录: {cve_item}")
            return
        
        logger.info(f"{change_type} CVE: {cve_id}")
        
        # 获取完整CVE详情
        cve_details = self.fetch_cve_details(cve_id)
        if not cve_details:
            return
        
        # 检查资产影响
        affected_assets = self.check_asset_impact(cve_details)
        cve_details['affected_assets'] = affected_assets
        
        # 发送通知
        if affected_assets or change_type == "新增":
            self.send_notification(cve_details)
    
    def run(self):
        """执行一次监控任务"""
        logger.info("=== 开始CVE监控任务 ===")
        delta_data = self.fetch_delta()
        self.process_delta(delta_data)
        logger.info("=== CVE监控任务完成 ===")
    
    def run_periodically(self, interval_hours: int = 24):
        """定期执行监控任务"""
        logger.info(f"=== 启动CVE定期监控，间隔: {interval_hours}小时 ===")
        while True:
            self.run()
            logger.info(f"等待{interval_hours}小时后执行下一次监控...")
            time.sleep(interval_hours * 3600)


def main():
    parser = argparse.ArgumentParser(description='CVE监控自动化脚本')
    parser.add_argument('--asset-file', '-a', help='组织资产清单JSON文件路径')
    parser.add_argument('--config-file', '-c', help='配置文件路径，用于邮件通知等设置')
    parser.add_argument('--output-dir', '-o', default='cve_data', help='CVE数据保存目录')
    parser.add_argument('--interval', '-i', type=int, help='定期执行间隔（小时），不指定则执行一次')
    parser.add_argument('--clear-cache', action='store_true', help='清理HTTP请求缓存')
    parser.add_argument('--show-cache-stats', action='store_true', help='显示缓存统计信息')
    parser.add_argument('--cache-warmup', action='store_true', help='执行缓存预热')
    parser.add_argument('--web', action='store_true', help='启动Web界面')
    
    args = parser.parse_args()
    
    # 处理Web界面命令
    if args.web:
        try:
            import subprocess
            import sys
            # 启动Web应用
            web_app_path = os.path.join(os.path.dirname(__file__), 'web_app.py')
            subprocess.run([sys.executable, web_app_path], check=True)
            return
        except Exception as e:
            logger.error(f"启动Web界面失败: {e}")
            return
    
    monitor = CVEMonitor(args.asset_file, args.output_dir, args.config_file)
    
    # 处理缓存管理命令
    if args.clear_cache:
        monitor.clear_cache()
        return
    
    if args.show_cache_stats:
        monitor.show_cache_stats()
        return
    
    if args.cache_warmup:
        monitor.cache_warmup()
        return
    
    # 处理监控命令
    if args.interval:
        monitor.run_periodically(args.interval)
    else:
        monitor.run()


if __name__ == "__main__":
    main()
