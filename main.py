from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # 用于请求体/响应体的数据校验

# 初始化FastAPI应用（自动关联OpenAPI）
app = FastAPI(
    title="用户管理系统",
    description="基于OpenAPI开发的用户CRUD接口",
    version="1.0.0"
)

# --------------------------
# 1. 定义数据模型（接口的输入/输出格式）
# --------------------------
# 请求体模型：创建/更新用户时的参数规范
class UserCreate(BaseModel):
    username: str  # 用户名（必填）
    email: str     # 邮箱（必填）
    age: int = None  # 年龄（可选）

# 响应体模型：返回用户信息时的规范（可隐藏敏感字段）
class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    age: int = None

# 模拟数据库（实际开发替换为MySQL/PostgreSQL）
fake_db = {
    1: {"user_id": 1, "username": "test1", "email": "test1@example.com", "age": 20},
    2: {"user_id": 2, "username": "test2", "email": "test2@example.com"}
}


# --------------------------
# 2. 编写接口（自动同步到OpenAPI）
# --------------------------
# 接口1：获取所有用户（GET请求）
@app.get("/users/", response_model=list[UserResponse], summary="获取所有用户列表")
def get_all_users():
    return list(fake_db.values())

# 接口2：根据ID获取单个用户（GET请求）
@app.get("/users/{user_id}", response_model=UserResponse, summary="根据ID获取用户详情")
def get_user(user_id: int):
    if user_id not in fake_db:
        # 抛出HTTP异常（OpenAPI会自动识别错误响应）
        raise HTTPException(status_code=404, detail="用户不存在")
    return fake_db[user_id]

# 接口3：创建新用户（POST请求）
@app.post("/users/", response_model=UserResponse, status_code=201, summary="创建新用户")
def create_user(user: UserCreate):
    new_user_id = max(fake_db.keys()) + 1
    new_user = {
        "user_id": new_user_id,
        "username": user.username,
        "email": user.email,
        "age": user.age
    }
    fake_db[new_user_id] = new_user
    return new_user

# 接口4：更新用户信息（PUT请求）
@app.put("/users/{user_id}", response_model=UserResponse, summary="更新用户信息")
def update_user(user_id: int, user: UserCreate):
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail="用户不存在")
    fake_db[user_id].update({
        "username": user.username,
        "email": user.email,
        "age": user.age
    })
    return fake_db[user_id]