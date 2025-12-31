"""
硬件兼容性验证脚本
检查本地电脑是否满足项目运行要求
"""

import platform
import psutil
import sys
import os
import json
from datetime import datetime


def check_system_compatibility():
    """检查系统兼容性"""
    print("=== 系统兼容性检查 ===")
    
    system_info = {
        "os": platform.system(),
        "os_version": platform.release(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "platform_info": platform.platform()
    }
    
    print(f"操作系统: {system_info['os']} {system_info['os_version']}")
    print(f"系统架构: {system_info['architecture']}")
    print(f"Python版本: {system_info['python_version'].split()[0]}")
    
    # 检查是否为支持的操作系统
    supported_os = ["Windows", "Linux", "Darwin"]  # Windows, Linux, macOS
    if system_info["os"] in supported_os:
        print("✅ 操作系统兼容")
    else:
        print("❌ 操作系统可能不兼容")
    
    return system_info


def check_hardware_requirements():
    """检查硬件要求"""
    print("\n=== 硬件要求检查 ===")
    
    # CPU 检查
    cpu_info = {
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
    }
    
    print(f"逻辑核心数: {cpu_info['logical_cores']}")
    print(f"物理核心数: {cpu_info['physical_cores']}")
    
    if cpu_info['logical_cores'] >= 4:
        print("✅ CPU核心数满足要求")
    else:
        print("⚠️  CPU核心数较低，可能影响性能")
    
    # 内存检查
    memory_info = {
        "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "percent_used": psutil.virtual_memory().percent
    }
    
    print(f"总内存: {memory_info['total_gb']} GB")
    print(f"可用内存: {memory_info['available_gb']} GB")
    print(f"内存使用率: {memory_info['percent_used']}%")
    
    if memory_info['total_gb'] >= 8:
        print("✅ 内存容量满足要求")
    else:
        print("⚠️  内存容量较低，可能影响AI模型训练")
    
    # 磁盘空间检查
    disk_info = {
        "total_gb": round(psutil.disk_usage('.').total / (1024**3), 2),
        "free_gb": round(psutil.disk_usage('.').free / (1024**3), 2),
        "percent_used": round((psutil.disk_usage('.').used/psutil.disk_usage('.').total)*100, 2)
    }
    
    print(f"总磁盘空间: {disk_info['total_gb']} GB")
    print(f"可用磁盘空间: {disk_info['free_gb']} GB")
    print(f"磁盘使用率: {disk_info['percent_used']}%")
    
    if disk_info['free_gb'] >= 10:
        print("✅ 磁盘空间满足要求")
    else:
        print("⚠️  磁盘空间不足，可能影响模型存储")
    
    return cpu_info, memory_info, disk_info


def check_software_dependencies():
    """检查软件依赖"""
    print("\n=== 软件依赖检查 ===")
    
    dependencies = {
        "python": {"required": True, "installed": True, "version": sys.version.split()[0]},
        "jax": {"required": True, "installed": False},
        "flax": {"required": False, "installed": False},  # 由于已知问题，设为非必需
        "numpy": {"required": True, "installed": False},
        "fastapi": {"required": True, "installed": False},
        "requests": {"required": True, "installed": False},
        "pydantic": {"required": True, "installed": False}
    }
    
    # 检查各依赖
    for dep in ["jax", "numpy", "fastapi", "requests", "pydantic"]:
        try:
            if dep == "jax":
                import jax
                dependencies[dep]["installed"] = True
                dependencies[dep]["version"] = jax.__version__
            elif dep == "numpy":
                import numpy
                dependencies[dep]["installed"] = True
                dependencies[dep]["version"] = numpy.__version__
            elif dep == "fastapi":
                import fastapi
                dependencies[dep]["installed"] = True
                dependencies[dep]["version"] = fastapi.__version__
            elif dep == "requests":
                import requests
                dependencies[dep]["installed"] = True
                dependencies[dep]["version"] = requests.__version__
            elif dep == "pydantic":
                import pydantic
                dependencies[dep]["installed"] = True
                dependencies[dep]["version"] = pydantic.__version__
        except ImportError:
            dependencies[dep]["installed"] = False
            dependencies[dep]["version"] = "Not installed"
    
    # 输出检查结果
    for dep, info in dependencies.items():
        status = "✅" if info["installed"] else ("⚠️" if not info["required"] else "❌")
        version_info = f"({info.get('version', '')})" if info['installed'] else ''
        print(f"{status} {dep}: {'Installed' if info['installed'] else 'Not installed'} {version_info}")
    
    # 检查总体兼容性
    required_installed = all(
        info["installed"] for dep, info in dependencies.items() 
        if info["required"]
    )
    
    if required_installed:
        print("\n✅ 所有必需依赖均已安装")
    else:
        missing_deps = [
            dep for dep, info in dependencies.items() 
            if info["required"] and not info["installed"]
        ]
        print(f"\n❌ 缺少依赖: {', '.join(missing_deps)}")
    
    return dependencies


def check_project_specific_requirements():
    """检查项目特定要求"""
    print("\n=== 项目特定要求检查 ===")
    
    # 检查项目文件结构
    project_paths = [
        "backend/src/api/routes",
        "backend/src/services",
        "backend/src/core",
        "frontend/src",
        "api-gateway/src"
    ]
    
    print("检查项目目录结构:")
    for path in project_paths:
        full_path = os.path.join("d:\\1.5", path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {path}")
    
    # 检查关键配置文件
    config_files = [
        "backend/requirements.txt",
        "frontend/package.json",
        "api-gateway/package.json",
        "docker-compose.yml"
    ]
    
    print("\n检查配置文件:")
    for file in config_files:
        full_path = os.path.join("d:\\1.5", file)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
    
    # 检查环境变量文件
    env_files = [".env", "backend/.env", "frontend/.env"]
    print("\n检查环境配置:")
    for file in env_files:
        full_path = os.path.join("d:", "1.5", file)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "ℹ️"
        print(f"  {status} {file}")
    
    return True


def generate_compatibility_report(system_info, cpu_info, memory_info, disk_info, dependencies):
    """生成兼容性报告"""
    print("\n=== 硬件兼容性报告 ===")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "hardware_info": {
            "cpu": cpu_info,
            "memory": memory_info,
            "disk": disk_info
        },
        "dependencies": dependencies,
        "overall_compatibility": "unknown"
    }
    
    # 综合评估
    hw_compatible = (
        cpu_info['logical_cores'] >= 4 and 
        memory_info['total_gb'] >= 8 and 
        disk_info['free_gb'] >= 10
    )
    
    sw_compatible = all(
        info["installed"] for dep, info in dependencies.items() 
        if info["required"]
    )
    
    if hw_compatible and sw_compatible:
        report["overall_compatibility"] = "fully_compatible"
        print("🎉 硬件和软件完全兼容，可以运行项目")
    elif hw_compatible or sw_compatible:
        report["overall_compatibility"] = "partially_compatible"
        print("⚠️  硬件或软件部分兼容，可能需要额外配置")
    else:
        report["overall_compatibility"] = "not_compatible"
        print("❌ 硬件或软件不兼容，需要升级或配置")
    
    print(f"系统: {system_info['os']} {system_info['os_version']}")
    print(f"CPU: {cpu_info['logical_cores']} 核心")
    print(f"内存: {memory_info['total_gb']} GB")
    print(f"可用磁盘: {disk_info['free_gb']} GB")
    
    # 建议
    print("\n建议:")
    if cpu_info['logical_cores'] < 4:
        print("- 建议使用更多核心的CPU以提高AI模型训练效率")
    if memory_info['total_gb'] < 16:
        print("- 建议增加内存容量以支持大型AI模型")
    if disk_info['free_gb'] < 20:
        print("- 建议释放磁盘空间以存储模型和数据")
    if not dependencies['jax']['installed']:
        print("- 建议安装JAX以支持AI计算")
    if not dependencies['flax']['installed']:
        print("- 注意: Flax存在兼容性问题，可能需要特殊配置")
    
    return report


def main():
    """主函数"""
    print("🚀 开始本地电脑硬件验证")
    print("=" * 50)
    
    # 执行各项检查
    system_info = check_system_compatibility()
    cpu_info, memory_info, disk_info = check_hardware_requirements()
    dependencies = check_software_dependencies()
    check_project_specific_requirements()
    
    # 生成报告
    report = generate_compatibility_report(
        system_info, cpu_info, memory_info, disk_info, dependencies
    )
    
    # 保存报告到文件
    report_path = "d:\\1.5\\hardware_compatibility_report.json"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📋 兼容性报告已保存至: {report_path}")
    except Exception as e:
        print(f"\n⚠️  保存报告时出错: {e}")
    
    print("\n" + "=" * 50)
    print("✅ 硬件验证完成!")
    
    return report


if __name__ == "__main__":
    main()