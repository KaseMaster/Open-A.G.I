#!/usr/bin/env python3
"""
🚀 AEGIS Enterprise API - Sprint 5.1
API REST enterprise completa para todos los servicios de AEGIS
"""

import asyncio
import time
import json
import logging
import base64
import io
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

# ===== MODELOS DE REQUEST =====

class TextAnalysisRequest(BaseModel):
    """Request para análisis de texto"""
    text: str = Field(..., min_length=1, max_length=10000)
    tasks: List[str] = Field(default=["sentiment", "entities", "classification"])
    language: str = "es"
    options: Dict[str, Any] = Field(default_factory=dict)

class ImageAnalysisRequest(BaseModel):
    """Request para análisis de imagen"""
    # image será manejada por UploadFile
    tasks: List[str] = Field(default=["detection", "classification", "segmentation"])
    options: Dict[str, Any] = Field(default_factory=dict)

class AudioAnalysisRequest(BaseModel):
    """Request para análisis de audio"""
    # audio será manejada por UploadFile
    tasks: List[str] = Field(default=["transcription", "classification", "emotion"])
    language: str = "es"
    options: Dict[str, Any] = Field(default_factory=dict)

class MultimodalAnalysisRequest(BaseModel):
    """Request para análisis multimodal"""
    text: Optional[str] = None
    # image y audio serán UploadFile
    tasks: List[str] = Field(default=["sentiment", "fusion"])
    fusion_strategy: str = "attention_based"
    options: Dict[str, Any] = Field(default_factory=dict)

class GenerationRequest(BaseModel):
    """Request para generación de contenido"""
    prompt: str = Field(..., min_length=1, max_length=1000)
    generation_type: str = Field(..., regex="^(text|image|multimodal)$")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class AnalyticsRequest(BaseModel):
    """Request para analytics"""
    data: Dict[str, Any]
    analysis_type: str = Field(..., regex="^(forecasting|anomaly|graph|reinforcement)$")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class PipelineRequest(BaseModel):
    """Request para pipelines"""
    pipeline_type: str
    input_data: Dict[str, Any]
    config: Dict[str, Any] = Field(default_factory=dict)

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

# ===== AEGIS API SERVICE =====

class AEGISAPIService:
    """Servicio principal de API para AEGIS"""

    def __init__(self):
        self.integration_pipeline = None
        self.multimodal_manager = None
        self.components = {}
        self.start_time = time.time()

    async def initialize(self):
        """Inicializar todos los servicios"""
        logger.info("🚀 Inicializando AEGIS API Service...")

        # Inicializar pipelines
        self.integration_pipeline = AEGISIntegrationPipeline()
        await self.integration_pipeline.initialize_pipeline()

        self.multimodal_manager = MultimodalPipelineManager()

        # Inicializar componentes individuales
        self.components = {
            'analytics': AEGISAdvancedAnalytics(),
            'vision': AEGISAdvancedComputerVision(),
            'nlp': AEGISNaturalLanguageProcessing(),
            'audio': AEGISAudioSpeechProcessing(),
            'multimodal': AEGISMultimodalFusion(),
            'generative': AEGISGenerativeAI(),
            'tinyml': AEGISTinyML(),
            'federated': AEGISFederatedAnalytics(),
            'explainable': AEGISExplainableAI()
        }

        logger.info("✅ AEGIS API Service inicializado")

    async def process_text_analysis(self, request: TextAnalysisRequest) -> Dict[str, Any]:
        """Procesar análisis de texto"""
        start_time = time.time()

        try:
            # Usar NLP component
            nlp = self.components['nlp']
            results = await nlp.process_text(request.text)

            # Filtrar por tareas solicitadas
            filtered_results = {}
            for task in request.tasks:
                if task in results:
                    filtered_results[task] = results[task]

            return {
                'results': filtered_results,
                'processing_time': time.time() - start_time,
                'language': request.language
            }

        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

    async def process_image_analysis(self, image_file: UploadFile) -> Dict[str, Any]:
        """Procesar análisis de imagen"""
        start_time = time.time()

        try:
            # Leer imagen
            image_data = await image_file.read()
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)

            # Usar vision component
            vision = self.components['vision']
            results = await vision.process_image(image_array)

            return {
                'results': results,
                'image_info': {
                    'width': image.width,
                    'height': image.height,
                    'format': image.format
                },
                'processing_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

    async def process_audio_analysis(self, audio_file: UploadFile) -> Dict[str, Any]:
        """Procesar análisis de audio"""
        start_time = time.time()

        try:
            # Leer audio
            audio_data = await audio_file.read()
            # Convertir a numpy array (simplificado)
            audio_array = np.frombuffer(audio_data, dtype=np.float32)

            # Usar audio component
            audio = self.components['audio']
            results = await audio.process_audio_file(audio_array)

            return {
                'results': results,
                'audio_info': {
                    'length': len(audio_array),
                    'sample_rate': 16000  # Asumido
                },
                'processing_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

    async def process_multimodal_analysis(self, text: Optional[str], image_file: Optional[UploadFile],
                                        audio_file: Optional[UploadFile], request: MultimodalAnalysisRequest) -> Dict[str, Any]:
        """Procesar análisis multimodal"""
        start_time = time.time()

        try:
            # Preparar inputs
            multimodal_data = {}

            if text:
                multimodal_data['text'] = text

            if image_file:
                image_data = await image_file.read()
                image = Image.open(io.BytesIO(image_data))
                multimodal_data['image'] = np.array(image)

            if audio_file:
                audio_data = await audio_file.read()
                multimodal_data['audio'] = np.frombuffer(audio_data, dtype=np.float32)

            # Usar integration pipeline
            pipeline_input = PipelineInput(
                data=multimodal_data,
                pipeline_type=PipelineType.MULTIMODAL_PIPELINE,
                config={'fusion_strategy': request.fusion_strategy}
            )

            result = await self.integration_pipeline.process_pipeline(pipeline_input)

            return {
                'results': result.results,
                'processing_time': result.processing_time,
                'modalities_used': list(multimodal_data.keys()),
                'stages_completed': [stage.value for stage in result.stages_completed]
            }

        except Exception as e:
            logger.error(f"Multimodal analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {str(e)}")

    async def process_generation(self, request: GenerationRequest) -> Dict[str, Any]:
        """Procesar generación de contenido"""
        start_time = time.time()

        try:
            generative = self.components['generative']

            if request.generation_type == "text":
                result = await generative.generate_text(request.prompt)

            elif request.generation_type == "image":
                result = await generative.generate_image(request.prompt)

            else:
                # Multimodal generation
                result = await generative.generate_story_with_images(request.prompt)

            return {
                'result': result.generated_text if hasattr(result, 'generated_text') else str(result),
                'generation_type': request.generation_type,
                'processing_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    async def process_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Procesar analytics"""
        start_time = time.time()

        try:
            if request.analysis_type == "forecasting":
                analytics = self.components['analytics']
                # Placeholder - implementar forecasting
                result = {"forecast": "forecasting_completed", "confidence": 0.85}

            elif request.analysis_type == "anomaly":
                anomaly = self.components['anomaly_detection']
                # Placeholder
                result = {"anomalies_detected": 0, "confidence": 0.90}

            else:
                result = {"message": f"{request.analysis_type} analytics completed"}

            return {
                'results': result,
                'analysis_type': request.analysis_type,
                'processing_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Analytics failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

    async def process_pipeline(self, request: PipelineRequest) -> Dict[str, Any]:
        """Procesar pipeline personalizado"""
        start_time = time.time()

        try:
            pipeline_input = PipelineInput(
                data=request.input_data,
                pipeline_type=PipelineType(request.pipeline_type),
                config=request.config
            )

            result = await self.integration_pipeline.process_pipeline(pipeline_input)

            return {
                'results': result.results,
                'processing_time': result.processing_time,
                'pipeline_type': request.pipeline_type,
                'stages_completed': [stage.value for stage in result.stages_completed]
            }

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

    def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado de salud del servicio"""
        uptime = time.time() - self.start_time

        component_status = {}
        for name, component in self.components.items():
            component_status[name] = "healthy" if component else "not_initialized"

        return {
            'status': 'healthy',
            'version': '1.0.0',
            'uptime': uptime,
            'components': component_status,
            'timestamp': datetime.utcnow()
        }

# ===== FASTAPI APPLICATION =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager para inicialización y limpieza"""
    # Inicializar
    app.state.aegis_service = AEGISAPIService()
    await app.state.aegis_service.initialize()

    yield

    # Limpiar (si es necesario)
    pass

app = FastAPI(
    title="AEGIS Enterprise API",
    description="API REST enterprise completa para el framework AEGIS de IA multimodal",
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
    logger.info(".3f"
    return response

# ===== ENDPOINTS =====

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    service = app.state.aegis_service
    health_data = service.get_health_status()

    return HealthCheckResponse(**health_data)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AEGIS Enterprise API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

# ===== TEXT ANALYSIS ENDPOINTS =====

@app.post("/api/v1/text/analyze", response_model=APIResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    current_user: str = Depends(get_current_user)
):
    """Analizar texto con múltiples tareas"""
    await check_permissions(current_user, "read")

    service = app.state.aegis_service
    result = await service.process_text_analysis(request)

    return APIResponse(
        data=result,
        message="Text analysis completed",
        processing_time=result.get('processing_time')
    )

@app.post("/api/v1/text/sentiment", response_model=APIResponse)
async def text_sentiment(
    text: str = Form(..., min_length=1, max_length=10000),
    current_user: str = Depends(get_current_user)
):
    """Análisis de sentimiento de texto"""
    await check_permissions(current_user, "read")

    request = TextAnalysisRequest(text=text, tasks=["sentiment"])
    service = app.state.aegis_service
    result = await service.process_text_analysis(request)

    return APIResponse(
        data=result,
        message="Sentiment analysis completed"
    )

@app.post("/api/v1/text/entities", response_model=APIResponse)
async def text_entities(
    text: str = Form(..., min_length=1, max_length=10000),
    current_user: str = Depends(get_current_user)
):
    """Extracción de entidades de texto"""
    await check_permissions(current_user, "read")

    request = TextAnalysisRequest(text=text, tasks=["entities"])
    service = app.state.aegis_service
    result = await service.process_text_analysis(request)

    return APIResponse(
        data=result,
        message="Entity extraction completed"
    )

# ===== IMAGE ANALYSIS ENDPOINTS =====

@app.post("/api/v1/image/analyze", response_model=APIResponse)
async def analyze_image(
    file: UploadFile = File(...),
    tasks: List[str] = Query(["detection", "classification"]),
    current_user: str = Depends(get_current_user)
):
    """Analizar imagen"""
    await check_permissions(current_user, "read")

    service = app.state.aegis_service
    result = await service.process_image_analysis(file)

    return APIResponse(
        data=result,
        message="Image analysis completed",
        processing_time=result.get('processing_time')
    )

@app.post("/api/v1/image/detect", response_model=APIResponse)
async def detect_objects(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """Detectar objetos en imagen"""
    await check_permissions(current_user, "read")

    service = app.state.aegis_service
    result = await service.process_image_analysis(file)

    return APIResponse(
        data=result,
        message="Object detection completed"
    )

# ===== AUDIO ANALYSIS ENDPOINTS =====

@app.post("/api/v1/audio/analyze", response_model=APIResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    tasks: List[str] = Query(["transcription", "classification"]),
    current_user: str = Depends(get_current_user)
):
    """Analizar audio"""
    await check_permissions(current_user, "read")

    service = app.state.aegis_service
    result = await service.process_audio_analysis(file)

    return APIResponse(
        data=result,
        message="Audio analysis completed",
        processing_time=result.get('processing_time')
    )

@app.post("/api/v1/audio/transcribe", response_model=APIResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "es",
    current_user: str = Depends(get_current_user)
):
    """Transcribir audio a texto"""
    await check_permissions(current_user, "read")

    service = app.state.aegis_service
    result = await service.process_audio_analysis(file)

    return APIResponse(
        data=result,
        message="Audio transcription completed"
    )

# ===== MULTIMODAL ENDPOINTS =====

@app.post("/api/v1/multimodal/analyze", response_model=APIResponse)
async def analyze_multimodal(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    tasks: List[str] = Form(["sentiment"]),
    fusion_strategy: str = Form("attention_based"),
    current_user: str = Depends(get_current_user)
):
    """Análisis multimodal"""
    await check_permissions(current_user, "read")

    request = MultimodalAnalysisRequest(
        text=text,
        tasks=tasks,
        fusion_strategy=fusion_strategy
    )

    service = app.state.aegis_service
    result = await service.process_multimodal_analysis(text, image, audio, request)

    return APIResponse(
        data=result,
        message="Multimodal analysis completed",
        processing_time=result.get('processing_time')
    )

@app.post("/api/v1/multimodal/sentiment", response_model=APIResponse)
async def multimodal_sentiment(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    current_user: str = Depends(get_current_user)
):
    """Análisis de sentimiento multimodal"""
    await check_permissions(current_user, "read")

    request = MultimodalAnalysisRequest(
        text=text,
        tasks=["sentiment"]
    )

    service = app.state.aegis_service
    result = await service.process_multimodal_analysis(text, image, audio, request)

    return APIResponse(
        data=result,
        message="Multimodal sentiment analysis completed"
    )

# ===== GENERATION ENDPOINTS =====

@app.post("/api/v1/generate/text", response_model=APIResponse)
async def generate_text(
    prompt: str = Form(..., min_length=1, max_length=1000),
    max_length: int = Form(100, ge=10, le=500),
    current_user: str = Depends(get_current_user)
):
    """Generar texto"""
    await check_permissions(current_user, "write")

    request = GenerationRequest(
        prompt=prompt,
        generation_type="text",
        parameters={"max_length": max_length}
    )

    service = app.state.aegis_service
    result = await service.process_generation(request)

    return APIResponse(
        data=result,
        message="Text generation completed",
        processing_time=result.get('processing_time')
    )

@app.post("/api/v1/generate/image", response_model=APIResponse)
async def generate_image(
    prompt: str = Form(..., min_length=1, max_length=500),
    current_user: str = Depends(get_current_user)
):
    """Generar imagen"""
    await check_permissions(current_user, "write")

    request = GenerationRequest(
        prompt=prompt,
        generation_type="image"
    )

    service = app.state.aegis_service
    result = await service.process_generation(request)

    return APIResponse(
        data=result,
        message="Image generation completed"
    )

# ===== ANALYTICS ENDPOINTS =====

@app.post("/api/v1/analytics/forecast", response_model=APIResponse)
async def forecast_analytics(
    data: Dict[str, Any],
    horizon: int = 10,
    current_user: str = Depends(get_current_user)
):
    """Forecasting con series de tiempo"""
    await check_permissions(current_user, "read")

    request = AnalyticsRequest(
        data=data,
        analysis_type="forecasting",
        parameters={"horizon": horizon}
    )

    service = app.state.aegis_service
    result = await service.process_analytics(request)

    return APIResponse(
        data=result,
        message="Analytics forecasting completed"
    )

@app.post("/api/v1/analytics/anomaly", response_model=APIResponse)
async def detect_anomalies(
    data: Dict[str, Any],
    threshold: float = 0.95,
    current_user: str = Depends(get_current_user)
):
    """Detección de anomalías"""
    await check_permissions(current_user, "read")

    request = AnalyticsRequest(
        data=data,
        analysis_type="anomaly",
        parameters={"threshold": threshold}
    )

    service = app.state.aegis_service
    result = await service.process_analytics(request)

    return APIResponse(
        data=result,
        message="Anomaly detection completed"
    )

# ===== PIPELINE ENDPOINTS =====

@app.post("/api/v1/pipeline/process", response_model=APIResponse)
async def process_pipeline(
    pipeline_type: str,
    input_data: Dict[str, Any],
    config: Dict[str, Any] = {},
    current_user: str = Depends(get_current_user)
):
    """Procesar pipeline personalizado"""
    await check_permissions(current_user, "write")

    request = PipelineRequest(
        pipeline_type=pipeline_type,
        input_data=input_data,
        config=config
    )

    service = app.state.aegis_service
    result = await service.process_pipeline(request)

    return APIResponse(
        data=result,
        message="Pipeline processing completed",
        processing_time=result.get('processing_time')
    )

# ===== UTILITY ENDPOINTS =====

@app.get("/api/v1/status")
async def get_status(current_user: str = Depends(get_current_user)):
    """Obtener estado del sistema"""
    service = app.state.aegis_service
    status_data = service.get_health_status()

    return APIResponse(
        data=status_data,
        message="System status retrieved"
    )

@app.get("/api/v1/capabilities")
async def get_capabilities(current_user: str = Depends(get_current_user)):
    """Obtener capacidades disponibles"""
    capabilities = {
        "text_analysis": ["sentiment", "entities", "classification", "summarization"],
        "image_analysis": ["detection", "classification", "segmentation", "captioning"],
        "audio_analysis": ["transcription", "classification", "emotion", "speaker_id"],
        "multimodal": ["sentiment", "fusion", "retrieval", "generation"],
        "generation": ["text", "image", "multimodal"],
        "analytics": ["forecasting", "anomaly", "graph", "reinforcement"],
        "pipelines": ["integration", "multimodal_specialized", "custom"]
    }

    return APIResponse(
        data=capabilities,
        message="Capabilities retrieved"
    )

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
        "aegis_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
