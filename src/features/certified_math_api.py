"""
CertifiedMath API Extension for Open-A.G.I
Exposes CertifiedMath operations through the Open-A.G.I API framework
"""
import sys
import os
import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import numpy as np
from PIL import Image

# Importar modelos de CertifiedMath
from certified_math_models import CertifiedMathOperationRequest, CertifiedMathOperationResponse, CertifiedMathExportRequest, CertifiedMathExportResponse

# Importar componentes de AEGIS
from integration_pipeline import AEGISIntegrationPipeline, PipelineInput, PipelineType
from multimodal_pipelines import MultimodalPipelineManager, MultimodalPipelineConfig, MultimodalPipelineInput, MultimodalPipelineType
from ml_framework_integration import MLFrameworkManager
from advanced_analytics_forecasting import AEGISAdvancedAnalytics
from graph_neural_networks import AEGISGraphNeuralNetworks
from reinforcement_learning import AEGISReinforcementLearning
from anomaly_detection import AEGISAnomalyDetection
from explainable_ai_shap import AEGISExplainableAI
from federated_analytics_privacy import AEGISFederatedAnalytics
from advanced_computer_vision import AEGISAdvancedComputerVision
from natural_language_processing import AEGISNaturalLanguageProcessing
from audio_speech_processing import AEGISAudioSpeechProcessing
from multimodal_fusion import AEGISMultimodalFusion
from tinyml_edge_ai import AEGISTinyML
from generative_ai import AEGISGenerativeAI

# Importar QFS V13 CertifiedMath
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'QFS', 'V13', 'libs'))
from CertifiedMath import CertifiedMath, BigNum128

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== MODELOS DE DATOS API =====

class APIResponse(BaseModel):
    """Respuesta estándar de API"""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    processing_time: Optional[float] = None

class ErrorResponse(BaseModel):
    """Respuesta de error"""
    success: bool = False
    error: str
    error_code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None

class HealthCheckResponse(BaseModel):
    """Respuesta de health check"""
    status: str
    version: str
    uptime: float
    components: Dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ===== AUTENTICACIÓN =====

class AuthManager:
    """Gestor de autenticación"""

    def __init__(self):
        # API keys simples (en producción usar JWT o OAuth)
        self.valid_api_keys = {
            "aegis_admin": "admin_key_2024",
            "aegis_user": "user_key_2024",
            "aegis_demo": "demo_key_2024"
        }

    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validar API key"""
        for user, key in self.valid_api_keys.items():
            if key == api_key:
                return user
        return None

    def get_user_permissions(self, user: str) -> List[str]:
        """Obtener permisos de usuario"""
        permissions = {
            "aegis_admin": ["read", "write", "admin", "unlimited"],
            "aegis_user": ["read", "write"],
            "aegis_demo": ["read"]
        }
        return permissions.get(user, [])

auth_manager = AuthManager()
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependencia para obtener usuario actual"""
    user = auth_manager.validate_api_key(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return user

async def check_permissions(user: str, required_permission: str):
    """Verificar permisos de usuario"""
    permissions = auth_manager.get_user_permissions(user)
    if required_permission not in permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required: {required_permission}"
        )

# ===== RATE LIMITING =====

class RateLimiter:
    """Limitador de tasa simple"""

    def __init__(self):
        self.requests = {}
        self.max_requests_per_minute = 60  # Por defecto

    def is_allowed(self, user: str) -> bool:
        """Verificar si request está permitido"""
        now = datetime.utcnow()
        minute_key = now.strftime("%Y-%m-%d %H:%M")

        if user not in self.requests:
            self.requests[user] = {}

        if minute_key not in self.requests[user]:
            self.requests[user][minute_key] = 0

        # Limpiar requests antiguos
        self._cleanup_old_requests(user, now)

        # Verificar límite
        permissions = auth_manager.get_user_permissions(user)
        max_requests = 1000 if "unlimited" in permissions else 60

        if self.requests[user][minute_key] >= max_requests:
            return False

        self.requests[user][minute_key] += 1
        return True

    def _cleanup_old_requests(self, user: str, now: datetime):
        """Limpiar requests antiguos"""
        cutoff = now - timedelta(minutes=5)
        cutoff_key = cutoff.strftime("%Y-%m-%d %H:%M")

        to_remove = []
        for key in self.requests[user]:
            if key < cutoff_key:
                to_remove.append(key)

        for key in to_remove:
            del self.requests[user][key]

rate_limiter = RateLimiter()

# ===== CERTIFIED MATH API SERVICE =====

class CertifiedMathAPIService:
    """Servicio de API para CertifiedMath"""

    def __init__(self):
        pass

    async def add_operation(self, request: CertifiedMathOperationRequest) -> CertifiedMathOperationResponse:
        """Perform addition operation using CertifiedMath"""
        start_time = time.time()
        try:
            # Convert string operands to BigNum128
            a = CertifiedMath.from_string(request.operand_a)
            b = CertifiedMath.from_string(request.operand_b)
            
            # Perform addition
            result = CertifiedMath.add(a, b, request.pqc_cid)
            
            # Get log hash for audit trail
            log_hash = CertifiedMath.get_log_hash()
            
            return CertifiedMathOperationResponse(
                success=True,
                result=str(result.value),
                log_hash=log_hash,
                pqc_cid=request.pqc_cid,
                operation="add",
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in add_operation: {e}")
            return CertifiedMathOperationResponse(
                success=False,
                error=str(e),
                operation="add",
                timestamp=datetime.utcnow()
            )
    
    async def sub_operation(self, request: CertifiedMathOperationRequest) -> CertifiedMathOperationResponse:
        """Perform subtraction operation using CertifiedMath"""
        start_time = time.time()
        try:
            # Convert string operands to BigNum128
            a = CertifiedMath.from_string(request.operand_a)
            b = CertifiedMath.from_string(request.operand_b)
            
            # Perform subtraction
            result = CertifiedMath.sub(a, b, request.pqc_cid)
            
            # Get log hash for audit trail
            log_hash = CertifiedMath.get_log_hash()
            
            return CertifiedMathOperationResponse(
                success=True,
                result=str(result.value),
                log_hash=log_hash,
                pqc_cid=request.pqc_cid,
                operation="sub",
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in sub_operation: {e}")
            return CertifiedMathOperationResponse(
                success=False,
                error=str(e),
                operation="sub",
                timestamp=datetime.utcnow()
            )
    
    async def mul_operation(self, request: CertifiedMathOperationRequest) -> CertifiedMathOperationResponse:
        """Perform multiplication operation using CertifiedMath"""
        start_time = time.time()
        try:
            # Convert string operands to BigNum128
            a = CertifiedMath.from_string(request.operand_a)
            b = CertifiedMath.from_string(request.operand_b)
            
            # Perform multiplication
            result = CertifiedMath.mul(a, b, request.pqc_cid)
            
            # Get log hash for audit trail
            log_hash = CertifiedMath.get_log_hash()
            
            return CertifiedMathOperationResponse(
                success=True,
                result=str(result.value),
                log_hash=log_hash,
                pqc_cid=request.pqc_cid,
                operation="mul",
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in mul_operation: {e}")
            return CertifiedMathOperationResponse(
                success=False,
                error=str(e),
                operation="mul",
                timestamp=datetime.utcnow()
            )
    
    async def div_operation(self, request: CertifiedMathOperationRequest) -> CertifiedMathOperationResponse:
        """Perform division operation using CertifiedMath"""
        start_time = time.time()
        try:
            # Convert string operands to BigNum128
            a = CertifiedMath.from_string(request.operand_a)
            b = CertifiedMath.from_string(request.operand_b)
            
            # Perform division
            result = CertifiedMath.div(a, b, request.pqc_cid)
            
            # Get log hash for audit trail
            log_hash = CertifiedMath.get_log_hash()
            
            return CertifiedMathOperationResponse(
                success=True,
                result=str(result.value),
                log_hash=log_hash,
                pqc_cid=request.pqc_cid,
                operation="div",
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in div_operation: {e}")
            return CertifiedMathOperationResponse(
                success=False,
                error=str(e),
                operation="div",
                timestamp=datetime.utcnow()
            )
    
    async def sqrt_operation(self, request: CertifiedMathOperationRequest) -> CertifiedMathOperationResponse:
        """Perform square root operation using CertifiedMath"""
        start_time = time.time()
        try:
            # Convert string operand to BigNum128
            a = CertifiedMath.from_string(request.operand_a)
            
            # Perform square root
            result = CertifiedMath.fast_sqrt(a, request.iterations, request.pqc_cid)
            
            # Get log hash for audit trail
            log_hash = CertifiedMath.get_log_hash()
            
            return CertifiedMathOperationResponse(
                success=True,
                result=str(result.value),
                log_hash=log_hash,
                pqc_cid=request.pqc_cid,
                operation="sqrt",
                iterations=request.iterations,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in sqrt_operation: {e}")
            return CertifiedMathOperationResponse(
                success=False,
                error=str(e),
                operation="sqrt",
                timestamp=datetime.utcnow()
            )
    
    async def phi_series_operation(self, request: CertifiedMathOperationRequest) -> CertifiedMathOperationResponse:
        """Perform phi series operation using CertifiedMath"""
        start_time = time.time()
        try:
            # Convert string operand to BigNum128
            a = CertifiedMath.from_string(request.operand_a)
            
            # Perform phi series calculation
            result = CertifiedMath.calculate_phi_series(a, request.iterations, request.pqc_cid)
            
            # Get log hash for audit trail
            log_hash = CertifiedMath.get_log_hash()
            
            return CertifiedMathOperationResponse(
                success=True,
                result=str(result.value),
                log_hash=log_hash,
                pqc_cid=request.pqc_cid,
                operation="phi_series",
                iterations=request.iterations,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in phi_series_operation: {e}")
            return CertifiedMathOperationResponse(
                success=False,
                error=str(e),
                operation="phi_series",
                timestamp=datetime.utcnow()
            )
    
    def export_audit_log(self, request: CertifiedMathExportRequest) -> CertifiedMathExportResponse:
        """Export the audit log to a file"""
        try:
            CertifiedMath.export_log(request.path)
            return CertifiedMathExportResponse(
                success=True,
                message=f"Audit log exported to {request.path}",
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error exporting audit log: {e}")
            return CertifiedMathExportResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Initialize the service
certified_math_service = CertifiedMathAPIService()

# ===== FASTAPI APPLICATION =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager para inicialización y limpieza"""
    yield
    # Limpiar (si es necesario)
    pass

app = FastAPI(
    title="CertifiedMath API Extension",
    description="API extension for CertifiedMath operations in QFS V13",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción especificar orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MIDDLEWARE =====

@app.middleware("http")
async def rate_limiting_middleware(request, call_next):
    """Middleware de rate limiting"""
    # Obtener API key del header
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

    if api_key:
        user = auth_manager.validate_api_key(api_key)
        if user and not rate_limiter.is_allowed(user):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "error_code": "RATE_LIMIT_EXCEEDED"}
            )

    response = await call_next(request)
    return response

@app.middleware("http")
async def logging_middleware(request, call_next):
    """Middleware de logging"""
    start_time = time.time()

    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)

    processing_time = time.time() - start_time
    logger.info(f"Request processed in {processing_time:.3f} seconds")
    return response

# ===== ENDPOINTS =====

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        uptime=time.time(),
        components={"certified_math": "available"}
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CertifiedMath API Extension",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

# ===== CERTIFIED MATH ENDPOINTS =====

@app.post("/api/v1/certified-math/add", response_model=CertifiedMathOperationResponse)
async def certified_math_add(
    request: CertifiedMathOperationRequest,
    current_user: str = Depends(get_current_user)
):
    """Perform addition operation using CertifiedMath"""
    await check_permissions(current_user, "write")
    return await certified_math_service.add_operation(request)

@app.post("/api/v1/certified-math/sub", response_model=CertifiedMathOperationResponse)
async def certified_math_sub(
    request: CertifiedMathOperationRequest,
    current_user: str = Depends(get_current_user)
):
    """Perform subtraction operation using CertifiedMath"""
    await check_permissions(current_user, "write")
    return await certified_math_service.sub_operation(request)

@app.post("/api/v1/certified-math/mul", response_model=CertifiedMathOperationResponse)
async def certified_math_mul(
    request: CertifiedMathOperationRequest,
    current_user: str = Depends(get_current_user)
):
    """Perform multiplication operation using CertifiedMath"""
    await check_permissions(current_user, "write")
    return await certified_math_service.mul_operation(request)

@app.post("/api/v1/certified-math/div", response_model=CertifiedMathOperationResponse)
async def certified_math_div(
    request: CertifiedMathOperationRequest,
    current_user: str = Depends(get_current_user)
):
    """Perform division operation using CertifiedMath"""
    await check_permissions(current_user, "write")
    return await certified_math_service.div_operation(request)

@app.post("/api/v1/certified-math/sqrt", response_model=CertifiedMathOperationResponse)
async def certified_math_sqrt(
    request: CertifiedMathOperationRequest,
    current_user: str = Depends(get_current_user)
):
    """Perform square root operation using CertifiedMath"""
    await check_permissions(current_user, "write")
    return await certified_math_service.sqrt_operation(request)

@app.post("/api/v1/certified-math/phi-series", response_model=CertifiedMathOperationResponse)
async def certified_math_phi_series(
    request: CertifiedMathOperationRequest,
    current_user: str = Depends(get_current_user)
):
    """Perform phi series operation using CertifiedMath"""
    await check_permissions(current_user, "write")
    return await certified_math_service.phi_series_operation(request)

@app.post("/api/v1/certified-math/export-log", response_model=CertifiedMathExportResponse)
async def certified_math_export_log(
    request: CertifiedMathExportRequest,
    current_user: str = Depends(get_current_user)
):
    """Export the audit log to a file"""
    await check_permissions(current_user, "write")
    return certified_math_service.export_audit_log(request)

# ===== ERROR HANDLERS =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejador de excepciones HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code="HTTP_EXCEPTION",
            details={"path": str(request.url.path)}
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejador de excepciones generales"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"path": str(request.url.path)}
        ).dict()
    )

# ===== MAIN =====

if __name__ == "__main__":
    uvicorn.run(
        "certified_math_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )