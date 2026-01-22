# 修复model_manager.py中的语法错误
import re
import os

file_path = 'd:\\1.5\\backend\\src\\core\\services\\model_manager.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复语法错误
content = content.replace(r'def update_model_metrics\\(self, model_id: str, metrics: dict\\):', 'def update_model_metrics(self, model_id: str, metrics: dict):')

# 保存修改后的文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('已修复语法错误')
