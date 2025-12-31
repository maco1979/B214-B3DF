import sys
import importlib.util
import importlib.abc

# 自定义模块加载器，用于修改Flax的kw_only_dataclasses模块
class FlaxPatchLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        self.target_module = 'flax.linen.kw_only_dataclasses'
        self.original_loader = None
        print("✓ Flax补丁加载器已创建")
    
    def find_spec(self, fullname, path, target=None):
        if fullname == self.target_module:
            print(f"✓ 正在拦截模块导入: {fullname}")
            # 临时移除加载器以避免无限递归
            loader_index = sys.meta_path.index(self)
            sys.meta_path.pop(loader_index)
            try:
                # 获取原始模块的spec
                spec = importlib.util.find_spec(fullname)
            finally:
                # 恢复加载器
                sys.meta_path.insert(loader_index, self)
            
            if spec:
                self.original_loader = spec.loader
                spec.loader = self
                return spec
        return None
    
    def exec_module(self, module):
        if module.__name__ == self.target_module:
            print(f"✓ 正在加载并修补模块: {module.__name__}")
            
            # 加载原始模块的源代码
            if hasattr(self.original_loader, 'get_source'):
                source = self.original_loader.get_source(module.__name__)
                if source:
                    # 应用补丁：1) 解决KeyError问题；2) 确保所有字段都有类型注解
                    patched_source = source.replace('cls_annotations = cls.__dict__[\'__annotations__\']', 'cls_annotations = cls.__annotations__')
                    
                    # 确保Any类型被导入
                    if 'from typing import Any' not in patched_source:
                        # 查找正确的位置添加Any导入
                        import_lines = patched_source.split('\n')
                        typing_import_line = None
                        for i, line in enumerate(import_lines):
                            if line.strip().startswith('from typing import') and 'Any' not in line:
                                typing_import_line = i
                                break
                        
                        if typing_import_line is not None:
                            # 在现有的typing导入中添加Any
                            import_lines[typing_import_line] = import_lines[typing_import_line].replace('from typing import', 'from typing import Any, ')
                            patched_source = '\n'.join(import_lines)
                        elif 'import typing' in patched_source:
                            # 在import typing后添加from typing import Any
                            patched_source = patched_source.replace('import typing', 'import typing\nfrom typing import Any')
                        else:
                            # 在文件开头添加from typing import Any
                            patched_source = 'from typing import Any\n' + patched_source
                    
                    # 修复dataclass字段必须有类型注解的问题
                    patched_source = patched_source.replace(
                        '  # Apply the dataclass transform.\n  transformed_cls: type[M] = dataclasses.dataclass(cls, **kwargs)',
                        '  # Apply the dataclass transform.\n  # 确保所有字段都有类型注解\n  for name, value in list(vars(cls).items()):\n    if isinstance(value, dataclasses.Field) and name not in cls.__annotations__:\n      cls.__annotations__[name] = Any\n  transformed_cls: type[M] = dataclasses.dataclass(cls, **kwargs)'
                    )
                    
                    # 在模块命名空间中执行修补后的代码
                    exec(patched_source, module.__dict__)
                    print("✓ Flax模块修补成功")
                    return
        
        # 如果无法修补，使用原始加载器
        if self.original_loader:
            self.original_loader.exec_module(module)

# 安装导入钩子
sys.meta_path.insert(0, FlaxPatchLoader())
print("✓ Flax导入补丁已安装")

# 尝试导入Flax以验证补丁是否生效
print("✓ 正在验证Flax导入...")
try:
    # 清除可能已缓存的Flax模块
    if 'flax' in sys.modules:
        for key in list(sys.modules.keys()):
            if key.startswith('flax.'):
                del sys.modules[key]
        del sys.modules['flax']
    
    from flax.linen import kw_only_dataclasses
    print("✓ Flax导入成功")
    
    # 验证修复是否有效
    test_class = type('TestClass', (), {})
    test_class.__annotations__ = {}
    print("✓ 补丁验证成功")
except Exception as e:
    print(f"✗ Flax导入失败: {e}")
    import traceback
    traceback.print_exc()
