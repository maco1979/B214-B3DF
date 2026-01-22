#!/bin/bash
            echo "AI Assistant 安装脚本"
            echo "==================="
            
            echo "正在安装Python依赖..."
            pip install -r requirements.txt
            
            echo "正在安装Node.js依赖..."
            npm install
            
            echo "正在创建配置文件..."
            python -c "from src.core.services.deployment_service import deployment_service; deployment_service._generate_config_file()"
            
            echo "安装完成！"
            echo "您可以使用以下命令启动应用程序："
            echo "python -m uvicorn src.main:app --host 0.0.0.0 --port 8001"
            