"""互联网访问服务
提供AI助手的互联网访问能力，包括网页搜索、数据抓取、API调用等功能
"""

from typing import Optional, Dict, List, Any
import requests
from bs4 import BeautifulSoup
import logging
import json
import time
from urllib.parse import urlparse, urljoin
import re

# 配置日志
logger = logging.getLogger(__name__)


class InternetService:
    """互联网访问服务类，提供网页搜索、数据抓取等功能"""
    
    def __init__(self):
        """初始化互联网访问服务"""
        self.session = requests.Session()
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3"
        })
        logger.info("互联网访问服务初始化完成")
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """在互联网上搜索信息
        
        Args:
            query: 搜索关键词
            num_results: 返回结果数量
            
        Returns:
            搜索结果列表，包含标题、URL和摘要
        """
        try:
            logger.info(f"执行网络搜索: {query}")
            
            # 使用DuckDuckGo搜索API（无需API密钥）
            search_url = f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}&kl=cn-zh"
            response = self.session.get(search_url, allow_redirects=True)
            response.raise_for_status()
            
            # 解析搜索结果
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # 查找搜索结果项
            for result in soup.find_all('div', class_='result__body')[:num_results]:
                # 提取标题
                title_elem = result.find('a', class_='result__a')
                if not title_elem:
                    continue
                
                title = title_elem.text.strip()
                url = title_elem['href']
                
                # 提取摘要
                snippet_elem = result.find('a', class_='result__snippet')
                snippet = snippet_elem.text.strip() if snippet_elem else ""
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })
            
            logger.info(f"搜索完成，找到 {len(results)} 个结果")
            return results
            
        except requests.RequestException as e:
            logger.error(f"搜索失败: {str(e)}")
            return [{"title": "搜索失败", "url": "", "snippet": f"无法连接到搜索服务: {str(e)}"}]
        except Exception as e:
            logger.error(f"搜索处理失败: {str(e)}")
            return [{"title": "搜索处理失败", "url": "", "snippet": f"处理搜索结果时出错: {str(e)}"}]
    
    def fetch_webpage(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """获取网页内容
        
        Args:
            url: 网页URL
            timeout: 请求超时时间
            
        Returns:
            网页内容，包含标题、正文和元数据
        """
        try:
            logger.info(f"获取网页内容: {url}")
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            
            # 检测编码
            response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取标题
            title = soup.title.string.strip() if soup.title else ""
            
            # 提取正文（去除脚本和样式）
            for script in soup(['script', 'style', 'nav', 'footer', 'aside']):
                script.decompose()
            
            # 提取正文内容
            paragraphs = soup.find_all('p')
            content = "\n".join(p.text.strip() for p in paragraphs if p.text.strip())
            
            # 提取元数据
            metadata = {}
            for meta in soup.find_all('meta'):
                if 'name' in meta.attrs and 'content' in meta.attrs:
                    metadata[meta['name']] = meta['content']
                elif 'property' in meta.attrs and 'content' in meta.attrs:
                    metadata[meta['property']] = meta['content']
            
            result = {
                "url": url,
                "title": title,
                "content": content,
                "metadata": metadata,
                "status_code": response.status_code
            }
            
            logger.info(f"成功获取网页内容: {url}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"获取网页失败: {str(e)}")
            return {
                "url": url,
                "title": "获取失败",
                "content": f"无法连接到网页: {str(e)}",
                "metadata": {},
                "status_code": getattr(e.response, 'status_code', 500) if hasattr(e, 'response') else 500
            }
        except Exception as e:
            logger.error(f"处理网页内容失败: {str(e)}")
            return {
                "url": url,
                "title": "处理失败",
                "content": f"处理网页内容时出错: {str(e)}",
                "metadata": {},
                "status_code": 500
            }
    
    def summarize_content(self, content: str, max_length: int = 200) -> str:
        """总结网页内容
        
        Args:
            content: 要总结的内容
            max_length: 总结的最大长度
            
        Returns:
            总结后的内容
        """
        try:
            # 简单的总结算法：提取前几个句子
            sentences = re.split(r'(?<=[。！？\.!?])\s+', content)
            summary = "".join(sentences[:3])  # 取前3个句子
            
            # 如果仍然太长，截断并添加省略号
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return summary
        except Exception as e:
            logger.error(f"内容总结失败: {str(e)}")
            return "无法总结内容"
    
    def call_api(self, url: str, method: str = "GET", params: Optional[Dict[str, Any]] = None, 
                 data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """调用外部API
        
        Args:
            url: API端点URL
            method: 请求方法（GET, POST, PUT, DELETE等）
            params: URL参数
            data: 请求体数据
            headers: 自定义请求头
            
        Returns:
            API响应结果
        """
        try:
            logger.info(f"调用API: {method} {url}")
            
            # 准备请求参数
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)
            
            # 发送请求
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers,
                timeout=10
            )
            
            response.raise_for_status()
            
            # 解析响应
            try:
                result = response.json()
            except json.JSONDecodeError:
                result = {
                    "content": response.text,
                    "status_code": response.status_code
                }
            
            logger.info(f"API调用成功: {url}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"API调用失败: {str(e)}")
            return {
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', 500) if hasattr(e, 'response') else 500
            }
        except Exception as e:
            logger.error(f"API处理失败: {str(e)}")
            return {
                "error": str(e),
                "status_code": 500
            }
    
    def get_image_urls(self, url: str, max_images: int = 5) -> List[str]:
        """从网页中提取图片URL
        
        Args:
            url: 网页URL
            max_images: 返回图片数量
            
        Returns:
            图片URL列表
        """
        try:
            logger.info(f"提取网页图片: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            image_urls = []
            
            for img in soup.find_all('img')[:max_images]:
                if 'src' in img.attrs:
                    img_url = img['src']
                    # 处理相对URL
                    if not img_url.startswith(('http://', 'https://')):
                        img_url = urljoin(url, img_url)
                    image_urls.append(img_url)
            
            logger.info(f"提取完成，找到 {len(image_urls)} 张图片")
            return image_urls
            
        except Exception as e:
            logger.error(f"提取图片失败: {str(e)}")
            return []
    
    def download_file(self, url: str, save_path: str) -> Dict[str, Any]:
        """下载文件
        
        Args:
            url: 文件URL
            save_path: 保存路径
            
        Returns:
            下载结果
        """
        try:
            logger.info(f"下载文件: {url} -> {save_path}")
            
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            # 下载文件
            downloaded_size = 0
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
            
            logger.info(f"文件下载完成: {save_path} ({downloaded_size} bytes)")
            return {
                "success": True,
                "file_path": save_path,
                "file_size": downloaded_size,
                "message": "文件下载成功"
            }
            
        except requests.RequestException as e:
            logger.error(f"文件下载失败: {str(e)}")
            return {
                "success": False,
                "file_path": save_path,
                "file_size": 0,
                "message": f"无法下载文件: {str(e)}"
            }
        except Exception as e:
            logger.error(f"文件保存失败: {str(e)}")
            return {
                "success": False,
                "file_path": save_path,
                "file_size": 0,
                "message": f"无法保存文件: {str(e)}"
            }


# 单例模式
internet_service = InternetService()