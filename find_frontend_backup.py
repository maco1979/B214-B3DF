import os

# 替换成你的D盘备份根目录路径，比如 D:/项目备份
BACKUP_ROOT = "D:/"

# 前端项目核心特征文件列表
FRONTEND_FLAGS = ["package.json", "src", "index.html", "vite.config.js", "vue.config.js"]

def find_frontend_ui_folders(root_dir):
    frontend_folders = []
    # 遍历D盘备份目录下的所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 统计当前文件夹下的前端特征数量
        flag_count = 0
        # 检查特征文件
        for flag in FRONTEND_FLAGS:
            if flag in filenames or flag in dirnames:
                flag_count += 1
        # 满足2个及以上特征，判定为前端UI项目
        if flag_count >= 2:
            frontend_folders.append(dirpath)
    return frontend_folders

if __name__ == "__main__":
    result = find_frontend_ui_folders(BACKUP_ROOT)
    if result:
        print("找到以下前端UI项目文件夹：")
        for folder in result:
            print(f"✅ {folder}")
    else:
        print("未找到前端UI项目文件夹")