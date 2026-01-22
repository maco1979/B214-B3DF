import json

def verify_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print(f"✓ {file_path} - JSON syntax is valid")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ {file_path} - JSON syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ {file_path} - Error: {e}")
        return False

# 验证所有zeabur.json文件
files = [
    'd:/1.6/1.5/zeabur.json',
    'd:/1.6/1.5/backend/zeabur.json',
    'd:/1.6/1.5/decision-service/zeabur.json'
]

all_valid = True
for file in files:
    if not verify_json(file):
        all_valid = False

if all_valid:
    print("\n✓ All JSON files are valid")
    exit(0)
else:
    print("\n✗ Some JSON files have errors")
    exit(1)
