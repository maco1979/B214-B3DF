import time
from locust import HttpUser, task, between


class APIUser(HttpUser):
    """模拟API用户的并发行为"""
    
    # 设置用户思考时间（请求之间的随机延迟）
    wait_time = between(0.5, 2.0)
    
    # 定义测试任务
    @task(1)
    def health_check(self):
        """测试健康检查端点"""
        self.client.get("/api/system/health")
    
    @task(2)
    def system_info(self):
        """测试系统信息端点"""
        self.client.get("/api/system/info")
    
    @task(2)
    def system_stats(self):
        """测试系统统计端点"""
        self.client.get("/api/system/stats")
    
    @task(1)
    def system_metrics(self):
        """测试系统指标端点"""
        self.client.get("/api/system/metrics")
    
    @task(3)
    def root_endpoint(self):
        """测试根路径端点"""
        self.client.get("/")


if __name__ == "__main__":
    print("Locust压力测试脚本已创建，可通过以下命令运行：")
    print("locust -f load_test.py --host=http://localhost:8001")
    print("")
    print("然后在浏览器中打开 http://localhost:8089 进行压力测试配置")
