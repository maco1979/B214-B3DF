"""
简化版API应用
不依赖AI框架，只提供auth和community接口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 直接从对应文件导入路由，避免间接导入AI框架依赖
from .routes.auth import router as auth_router
from .routes.community import router as community_router

# 导入用户和企业路由
try:
    from .routes.user import router as user_router
    from .routes.enterprise import router as enterprise_router
    user_routes_available = True
except ImportError:
    print("WARNING: 用户和企业路由不可用，可能缺少依赖")
    user_routes_available = False

# 创建简化版农业路由
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

# 创建农业路由
agriculture_router = APIRouter(prefix="/agriculture", tags=["agriculture"])

# 简化版光配方请求模型
class LightRecipeRequest(BaseModel):
    crop_type: str
    current_day: int
    target_objective: str  # "最大化产量", "提升甜度", "提升抗性"
    environment: Dict[str, float]  # 温度、湿度等环境数据

# OHSP测试报告模型
class OHSPTestReport(BaseModel):
    product_mark: Optional[str] = None
    model: Optional[str] = None
    manufacture: Optional[str] = None
    temperature: float = 20.0
    humidity: float = 65.0
    tester: Optional[str] = None
    test_date: Optional[str] = None
    par: float = 9.233  # PAR(mW/cm²)
    ppfd: float = 348.067  # PPFD(μmol/m²/s)
    ppfd_uv: float = 19.786  # PPFD_UV(μmol/m²/s)
    ppfd_b: float = 95.239  # PPFD_B(μmol/m²/s)
    ppfd_g: float = 84.514  # PPFD_G(μmol/m²/s)
    ppfd_r: float = 168.314  # PPFD_R(μmol/m²/s)
    ppfd_fr: float = 71.867  # PPFD_Fr(μmol/m²/s)
    yppfd: float = 25.290  # YPPFD(μmol/m²/s)
    kppfd: float = 320.692  # KPPFD(μmol/m²/s)
    eq: float = 6.832  # Eq(mW/cm²)
    eb: float = 2.622  # Eb(mW/cm²)
    ey: float = 1.837  # Ey(mW/cm²)
    er: float = 3.076  # Er(mW/cm²)
    erb_ratio: float = 1.173  # Erb Ratio
    e: float = 13763.24  # E(lx)
    cie_xy: List[float] = [0.3034, 0.2571]  # CIE x,y
    cie_uv: List[float] = [0.2215, 0.2816]  # CIE u,v
    cie_u_prime_v_prime: List[float] = [0.2215, 0.4224]  # CIE u',v'
    cct: float = 9026  # CCT(K)
    duv: float = -0.03317  # Duv
    s_p: float = 2.466  # S/P
    dominant: float = 380.00  # Dominant(nm)
    purity: float = 22.4  # Purity(%)
    half_width: float = 22.4  # HalfWidth(nm)
    peak: float = 664.7  # Peak(nm)
    center: float = 662.1  # Center(nm)
    centroid: float = 58.9  # Centroid(nm)
    color_ratio_rgb: List[float] = [18.7, 6.5, 4.7]  # ColorRatio(RGB)
    cie1931_x: float = 23779.613  # CIE1931 X
    cie1931_y: float = 20151.154  # CIE1931 Y
    cie1931_z: float = 34444.918  # CIE1931 Z
    integral_time: float = 16  # Integral Time(ms)
    ra: float = 62.8  # Ra
    sdcm: float = 35.13  # SDCM

# 模拟OHSP测试报告数据
test_reports = {
    "default": {
        "product_mark": "OHSP-2021",
        "model": "OHSP-TEST-001",
        "manufacture": "OHSP Labs",
        "temperature": 20.0,
        "humidity": 65.0,
        "tester": "admin",
        "test_date": "2021-01-19",
        "par": 9.233,
        "ppfd": 348.067,
        "ppfd_uv": 19.786,
        "ppfd_b": 95.239,
        "ppfd_g": 84.514,
        "ppfd_r": 168.314,
        "ppfd_fr": 71.867,
        "yppfd": 25.290,
        "kppfd": 320.692,
        "eq": 6.832,
        "eb": 2.622,
        "ey": 1.837,
        "er": 3.076,
        "erb_ratio": 1.173,
        "e": 13763.24,
        "cie_xy": [0.3034, 0.2571],
        "cie_uv": [0.2215, 0.2816],
        "cie_u_prime_v_prime": [0.2215, 0.4224],
        "cct": 9026,
        "duv": -0.03317,
        "s_p": 2.466,
        "dominant": 380.00,
        "purity": 22.4,
        "half_width": 22.4,
        "peak": 664.7,
        "center": 662.1,
        "centroid": 58.9,
        "color_ratio_rgb": [18.7, 6.5, 4.7],
        "cie1931_x": 23779.613,
        "cie1931_y": 20151.154,
        "cie1931_z": 34444.918,
        "integral_time": 16,
        "ra": 62.8,
        "sdcm": 35.13
    }
}

# 模拟光谱配置数据
default_spectrum = {
    "uv_380nm": 0.05,
    "far_red_720nm": 0.1,
    "white_light": 0.7,
    "red_660nm": 0.15,
    "white_red_ratio": 4.6667
}

# 模拟作物配置
available_crops = {
    "生菜": {
        "growth_stages": [
            {
                "stage_name": "苗期",
                "duration_days": 15,
                "optimal_temperature": [18, 22],
                "optimal_humidity": [60, 70],
                "light_hours": 14
            },
            {
                "stage_name": "生长期",
                "duration_days": 30,
                "optimal_temperature": [20, 25],
                "optimal_humidity": [55, 65],
                "light_hours": 16
            },
            {
                "stage_name": "收获期",
                "duration_days": 10,
                "optimal_temperature": [18, 23],
                "optimal_humidity": [60, 70],
                "light_hours": 12
            }
        ],
        "target_yield": "高产量",
        "quality_metrics": {
            "维生素C含量": 0.8,
            "叶片厚度": 0.7,
            "抗病虫害能力": 0.9
        }
    },
    "小麦": {
        "growth_stages": [
            {
                "stage_name": "苗期",
                "duration_days": 20,
                "optimal_temperature": [15, 20],
                "optimal_humidity": [50, 60],
                "light_hours": 12
            },
            {
                "stage_name": "分蘖期",
                "duration_days": 30,
                "optimal_temperature": [18, 25],
                "optimal_humidity": [55, 65],
                "light_hours": 14
            },
            {
                "stage_name": "拔节期",
                "duration_days": 25,
                "optimal_temperature": [20, 28],
                "optimal_humidity": [60, 70],
                "light_hours": 16
            },
            {
                "stage_name": "抽穗期",
                "duration_days": 20,
                "optimal_temperature": [22, 30],
                "optimal_humidity": [65, 75],
                "light_hours": 18
            },
            {
                "stage_name": "灌浆期",
                "duration_days": 30,
                "optimal_temperature": [20, 25],
                "optimal_humidity": [60, 70],
                "light_hours": 16
            },
            {
                "stage_name": "收获期",
                "duration_days": 15,
                "optimal_temperature": [18, 25],
                "optimal_humidity": [55, 65],
                "light_hours": 12
            }
        ],
        "target_yield": "高产量",
        "quality_metrics": {
            "蛋白质含量": 0.85,
            "千粒重": 0.9,
            "抗倒伏能力": 0.8
        }
    },
    "玫瑰": {
        "growth_stages": [
            {
                "stage_name": "苗期",
                "duration_days": 25,
                "optimal_temperature": [20, 25],
                "optimal_humidity": [60, 70],
                "light_hours": 12
            },
            {
                "stage_name": "营养生长期",
                "duration_days": 40,
                "optimal_temperature": [22, 28],
                "optimal_humidity": [55, 65],
                "light_hours": 14
            },
            {
                "stage_name": "花芽分化期",
                "duration_days": 20,
                "optimal_temperature": [18, 22],
                "optimal_humidity": [60, 70],
                "light_hours": 16
            },
            {
                "stage_name": "开花期",
                "duration_days": 15,
                "optimal_temperature": [20, 25],
                "optimal_humidity": [55, 65],
                "light_hours": 12
            }
        ],
        "target_yield": "高品质",
        "quality_metrics": {
            "花色鲜艳度": 0.95,
            "花期长度": 0.9,
            "花茎强度": 0.85
        }
    }
}

# 简化版光配方API
@agriculture_router.post("/light-recipe")
async def generate_light_recipe(request: LightRecipeRequest):
    """生成简化版光配方"""
    try:
        return {
            "success": True,
            "data": {
                "recipe": default_spectrum,
                "current_stage": "苗期" if request.current_day < 20 else "生长期",
                "light_hours": 14,
                "recommendations": [
                    "保持31:1的白红配比",
                    "确保PAR值在9.0-10.0 mW/cm²之间",
                    "维持PPFD值在300-400 μmol/m²/s"
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 获取OHSP测试报告
@agriculture_router.get("/ohsp-report/{report_id}")
async def get_ohsp_report(report_id: str = "default"):
    """获取OHSP测试报告"""
    if report_id not in test_reports:
        raise HTTPException(status_code=404, detail="测试报告不存在")
    return {
        "success": True,
        "data": test_reports[report_id]
    }

# 获取可用作物
@agriculture_router.get("/crop-configs")
async def get_available_crops():
    """获取可用的作物配置"""
    return {
        "success": True,
        "data": available_crops
    }

# 获取光谱分析
@agriculture_router.get("/spectrum-analysis")
async def analyze_spectrum():
    """获取光谱分析结果"""
    # 使用OHSP报告数据进行简化分析
    report = test_reports["default"]
    return {
        "success": True,
        "data": {
            "uv_380nm": report["ppfd_uv"] / report["ppfd"],
            "far_red_720nm": report["ppfd_fr"] / report["ppfd"],
            "white_light": report["ppfd_b"] / report["ppfd"],
            "red_660nm": report["ppfd_r"] / report["ppfd"],
            "white_red_ratio": report["ppfd_b"] / report["ppfd_r"],
            "par": report["par"],
            "ppfd": report["ppfd"],
            "ppfd_distribution": {
                "UV": report["ppfd_uv"],
                "Blue": report["ppfd_b"],
                "Green": report["ppfd_g"],
                "Red": report["ppfd_r"],
                "FarRed": report["ppfd_fr"]
            }
        }
    }

def create_simple_app() -> FastAPI:
    """创建简化版FastAPI应用"""
    
    app = FastAPI(
        title="AI项目API服务",
        description="简化版API服务，提供认证和社区功能",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境中应限制来源
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 只注册需要的路由
    app.include_router(auth_router, prefix="/api")
    app.include_router(community_router, prefix="/api")
    
    # 条件性注册用户和企业路由
    if user_routes_available:
        app.include_router(user_router, prefix="/api")
        app.include_router(enterprise_router, prefix="/api")
    
    # 根路径
    @app.get("/")
    async def root():
        return {
            "message": "AI项目API服务",
            "version": "1.0.0",
            "docs": "/docs",
            "mode": "simplified"  # 标记为简化模式
        }
    
    # 健康检查接口
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": "2025-12-27T12:00:00Z",
            "version": "1.0.0"
        }
    
    return app

# 创建应用实例
app = create_simple_app()
