# 替换camera.py文件中所有的FrameResponse为ApiResponse

import re

# 读取文件内容
file_path = "d:/1.6/1.5/backend/src/api/routes/camera.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换所有的FrameResponse为ApiResponse
new_content = re.sub(r'FrameResponse', 'ApiResponse', content)

# 写入修改后的内容
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"替换完成！共替换了 {len(re.findall(r'FrameResponse', content))} 处")