#!/usr/bin/env python3
"""
AEGIS API Server
Servidor REST API completo para exposición de servicios AEGIS

Características:
- API REST completa con FastAPI
- Autenticación JWT y OAuth2
- Documentación automática con Swagger/OpenAPI
- Rate limiting y throttling
- Middleware de seguridad
- Endpoints para todos los componentes AEGIS
- WebSocket para tiempo real
- Métricas y monitoreo
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import hashlib
import secrets

try:
    from fastapi import FastAPI, HTTPException, Depends, status, Request, WebSocket, WebSocketDisconnect
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Fallback classes
    class BaseModel:
        pass
    class FastAPI:
        pass

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from dataclasses import dataclass, asdict
from enum import Enum


# Configuración JWT
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class APIStatus(Enum):
    """Estados de la API"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class UserRole(Enum):
    """Roles de usuario"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"


# Modelos Pydantic
class APIResponse(BaseModel):
    """Respuesta estándar de la API"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class UserCreate(BaseModel):
    """Modelo para crear usuario"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.USER


class UserLogin(BaseModel):
    """Modelo para login"""
    username: str
    password: str


class Token(BaseModel):
    """Token de acceso"""
    access_token: str
    token_type: str
    expires_in: int


class NodeStatus(BaseModel):
    """Estado del nodo AEGIS"""
    node_id: str
    status: str
    uptime: float
    components: Dict[str, bool]
    metrics: Dict[str, Any]
    last_update: datetime


class ConfigUpdate(BaseModel):
    """Actualización de configuración"""
    key: str
    value: Any
    level: str = "runtime"
    reason: str = ""


class MetricQuery(BaseModel):
    """Consulta de métricas"""
    metric_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=100, le=1000)


@dataclass
class RateLimitRule:
    """Regla de rate limiting"""
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int


class RateLimiter:
    """Sistema de rate limiting"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.rules: Dict[str, RateLimitRule] = {
            "default": RateLimitRule(60, 1000, 10),
            "admin": RateLimitRule(120, 2000, 20),
            "service": RateLimitRule(300, 5000, 50)
        }
    
    def is_allowed(self, client_id: str, role: str = "default") -> tuple[bool, Dict[str, Any]]:
        """Verifica si una solicitud está permitida"""
        now = time.time()
        rule = self.rules.get(role, self.rules["default"])
        
        # Limpiar solicitudes antiguas
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Filtrar solicitudes de la última hora
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < 3600
        ]
        
        # Verificar límites
        recent_requests = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < 60
        ]
        
        if len(recent_requests) >= rule.requests_per_minute:
            return False, {
                "error": "Rate limit exceeded",
                "limit": rule.requests_per_minute,
                "window": "1 minute",
                "retry_after": 60 - (now - min(recent_requests))
            }
        
        if len(self.requests[client_id]) >= rule.requests_per_hour:
            return False, {
                "error": "Hourly limit exceeded",
                "limit": rule.requests_per_hour,
                "window": "1 hour",
                "retry_after": 3600 - (now - min(self.requests[client_id]))
            }
        
        # Registrar solicitud
        self.requests[client_id].append(now)
        
        return True, {
            "remaining_minute": rule.requests_per_minute - len(recent_requests) - 1,
            "remaining_hour": rule.requests_per_hour - len(self.requests[client_id])
        }


class AuthManager:
    """Gestor de autenticación"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.users_db: Dict[str, Dict[str, Any]] = {}
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Crear usuario admin por defecto
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Crea usuario admin por defecto"""
        admin_password = secrets.token_urlsafe(16)
        self.users_db["admin"] = {
            "username": "admin",
            "email": "admin@aegis.local",
            "hashed_password": self.pwd_context.hash(admin_password),
            "role": UserRole.ADMIN.value,
            "created_at": datetime.now(),
            "active": True
        }
        logger.info(f"🔐 Usuario admin creado con contraseña: {admin_password}")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifica contraseña"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Genera hash de contraseña"""
        return self.pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Autentica usuario"""
        user = self.users_db.get(username)
        if not user or not user.get("active", True):
            return None
        
        if not self.verify_password(password, user["hashed_password"]):
            return None
        
        return user
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Crea token de acceso"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Registrar token activo
        self.active_tokens[token] = {
            "user": data.get("sub"),
            "role": data.get("role"),
            "created_at": datetime.utcnow(),
            "expires_at": expire
        }
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                return None
            
            # Verificar si el token está activo
            if token not in self.active_tokens:
                return None
            
            return payload
        except JWTError:
            return None
    
    def revoke_token(self, token: str):
        """Revoca token"""
        if token in self.active_tokens:
            del self.active_tokens[token]
    
    def create_user(self, user_data: UserCreate) -> bool:
        """Crea nuevo usuario"""
        if user_data.username in self.users_db:
            return False
        
        self.users_db[user_data.username] = {
            "username": user_data.username,
            "email": user_data.email,
            "hashed_password": self.get_password_hash(user_data.password),
            "role": user_data.role.value,
            "created_at": datetime.now(),
            "active": True
        }
        return True


class WebSocketManager:
    """Gestor de conexiones WebSocket"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # client_id -> [topics]
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Conecta cliente WebSocket"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = []
        logger.info(f"🔌 Cliente WebSocket conectado: {client_id}")
    
    def disconnect(self, client_id: str):
        """Desconecta cliente"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"🔌 Cliente WebSocket desconectado: {client_id}")
    
    def subscribe(self, client_id: str, topic: str):
        """Suscribe cliente a un tópico"""
        if client_id in self.subscriptions:
            if topic not in self.subscriptions[client_id]:
                self.subscriptions[client_id].append(topic)
    
    def unsubscribe(self, client_id: str, topic: str):
        """Desuscribe cliente de un tópico"""
        if client_id in self.subscriptions and topic in self.subscriptions[client_id]:
            self.subscriptions[client_id].remove(topic)
    
    async def send_personal_message(self, message: str, client_id: str):
        """Envía mensaje personal"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error enviando mensaje a {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_topic(self, message: Dict[str, Any], topic: str):
        """Difunde mensaje a suscriptores de un tópico"""
        message_str = json.dumps(message)
        disconnected = []
        
        for client_id, topics in self.subscriptions.items():
            if topic in topics and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_text(message_str)
                except Exception as e:
                    logger.error(f"Error enviando a {client_id}: {e}")
                    disconnected.append(client_id)
        
        # Limpiar conexiones desconectadas
        for client_id in disconnected:
            self.disconnect(client_id)


class AEGISAPIServer:
    """Servidor API REST de AEGIS"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI no está disponible. Instalar con: pip install fastapi uvicorn")
        
        self.config = config or {}
        self.status = APIStatus.STOPPED
        self.start_time = time.time()
        
        # Componentes
        self.rate_limiter = RateLimiter()
        self.auth_manager = AuthManager()
        self.websocket_manager = WebSocketManager()
        
        # FastAPI app
        self.app = FastAPI(
            title="AEGIS API",
            description="API REST para el framework AEGIS de IA distribuida",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Configurar middleware
        self._setup_middleware()
        
        # Configurar rutas
        self._setup_routes()
        
        # Estado interno
        self.node_status = {
            "node_id": f"aegis_node_{uuid.uuid4().hex[:8]}",
            "components": {},
            "metrics": {},
            "last_update": datetime.now()
        }
        
        logger.info("🚀 AEGIS API Server inicializado")
    
    def _setup_middleware(self):
        """Configura middleware de seguridad"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted hosts
        if "trusted_hosts" in self.config:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config["trusted_hosts"]
            )
        
        # Rate limiting middleware
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            client_ip = request.client.host
            user_agent = request.headers.get("user-agent", "unknown")
            client_id = hashlib.md5(f"{client_ip}:{user_agent}".encode()).hexdigest()
            
            # Obtener rol del usuario si está autenticado
            role = "default"
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                payload = self.auth_manager.verify_token(token)
                if payload:
                    role = payload.get("role", "default")
            
            allowed, info = self.rate_limiter.is_allowed(client_id, role)
            
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "message": "Rate limit exceeded",
                        "error": info
                    },
                    headers={"Retry-After": str(int(info.get("retry_after", 60)))}
                )
            
            response = await call_next(request)
            
            # Añadir headers de rate limiting
            response.headers["X-RateLimit-Remaining-Minute"] = str(info.get("remaining_minute", 0))
            response.headers["X-RateLimit-Remaining-Hour"] = str(info.get("remaining_hour", 0))
            
            return response
    
    def _setup_routes(self):
        """Configura todas las rutas de la API"""
        
        # Dependencias de autenticación
        security = HTTPBearer()
        oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
        
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Obtiene usuario actual desde token"""
            token = credentials.credentials
            payload = self.auth_manager.verify_token(token)
            if payload is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return payload
        
        async def get_admin_user(current_user: dict = Depends(get_current_user)):
            """Verifica que el usuario sea admin"""
            if current_user.get("role") != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            return current_user
        
        # Rutas de salud y estado
        @self.app.get("/", response_model=APIResponse)
        async def root():
            """Endpoint raíz"""
            return APIResponse(
                success=True,
                message="AEGIS API Server running",
                data={
                    "version": "2.0.0",
                    "status": self.status.value,
                    "uptime": time.time() - self.start_time,
                    "node_id": self.node_status["node_id"]
                }
            )
        
        @self.app.get("/health", response_model=APIResponse)
        async def health_check():
            """Verificación de salud"""
            return APIResponse(
                success=True,
                message="API Server healthy",
                data={
                    "status": self.status.value,
                    "uptime": time.time() - self.start_time,
                    "components": self.node_status["components"],
                    "active_connections": len(self.websocket_manager.active_connections)
                }
            )
        
        @self.app.get("/status", response_model=NodeStatus)
        async def get_status():
            """Estado completo del nodo"""
            return NodeStatus(
                node_id=self.node_status["node_id"],
                status=self.status.value,
                uptime=time.time() - self.start_time,
                components=self.node_status["components"],
                metrics=self.node_status["metrics"],
                last_update=datetime.now()
            )
        
        # Rutas de autenticación
        @self.app.post("/auth/login", response_model=Token)
        async def login(user_data: UserLogin):
            """Login de usuario"""
            user = self.auth_manager.authenticate_user(user_data.username, user_data.password)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password"
                )
            
            access_token = self.auth_manager.create_access_token(
                data={"sub": user["username"], "role": user["role"]}
            )
            
            return Token(
                access_token=access_token,
                token_type="bearer",
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
        
        @self.app.post("/auth/register", response_model=APIResponse)
        async def register(user_data: UserCreate, current_user: dict = Depends(get_admin_user)):
            """Registro de usuario (solo admin)"""
            if self.auth_manager.create_user(user_data):
                return APIResponse(
                    success=True,
                    message=f"User {user_data.username} created successfully"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists"
                )
        
        @self.app.post("/auth/logout", response_model=APIResponse)
        async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Logout de usuario"""
            token = credentials.credentials
            self.auth_manager.revoke_token(token)
            return APIResponse(success=True, message="Logged out successfully")
        
        @self.app.get("/auth/me", response_model=APIResponse)
        async def get_current_user_info(current_user: dict = Depends(get_current_user)):
            """Información del usuario actual"""
            return APIResponse(
                success=True,
                message="User information",
                data={
                    "username": current_user["sub"],
                    "role": current_user["role"]
                }
            )
        
        # Rutas de configuración
        @self.app.get("/config", response_model=APIResponse)
        async def get_config(current_user: dict = Depends(get_current_user)):
            """Obtiene configuración actual"""
            try:
                # Importar config_manager si está disponible
                from config_manager import get_config_manager
                config_mgr = get_config_manager()
                config_data = config_mgr.get_all()
                
                return APIResponse(
                    success=True,
                    message="Configuration retrieved",
                    data=config_data
                )
            except Exception as e:
                logger.error(f"Error obteniendo configuración: {e}")
                return APIResponse(
                    success=False,
                    message=f"Error retrieving configuration: {str(e)}"
                )
        
        @self.app.post("/config", response_model=APIResponse)
        async def update_config(config_update: ConfigUpdate, current_user: dict = Depends(get_admin_user)):
            """Actualiza configuración"""
            try:
                from config_manager import get_config_manager
                config_mgr = get_config_manager()
                
                success = config_mgr.set(
                    config_update.key,
                    config_update.value,
                    level=config_update.level,
                    user=current_user["sub"],
                    reason=config_update.reason
                )
                
                if success:
                    return APIResponse(
                        success=True,
                        message=f"Configuration {config_update.key} updated"
                    )
                else:
                    return APIResponse(
                        success=False,
                        message="Configuration update failed"
                    )
            except Exception as e:
                logger.error(f"Error actualizando configuración: {e}")
                return APIResponse(
                    success=False,
                    message=f"Error updating configuration: {str(e)}"
                )
        
        # Rutas de métricas
        @self.app.get("/metrics", response_model=APIResponse)
        async def get_metrics(query: MetricQuery = Depends(), current_user: dict = Depends(get_current_user)):
            """Obtiene métricas del sistema"""
            try:
                # Simular métricas por ahora
                metrics_data = {
                    "system": {
                        "cpu_usage": 45.2,
                        "memory_usage": 67.8,
                        "disk_usage": 23.1,
                        "network_io": {"in": 1024, "out": 2048}
                    },
                    "aegis": {
                        "active_nodes": 3,
                        "consensus_rounds": 156,
                        "crypto_operations": 89,
                        "p2p_connections": 12
                    },
                    "performance": {
                        "batch_operations": 45,
                        "compression_ratio": 0.73,
                        "optimization_score": 8.7
                    }
                }
                
                return APIResponse(
                    success=True,
                    message="Metrics retrieved",
                    data=metrics_data
                )
            except Exception as e:
                logger.error(f"Error obteniendo métricas: {e}")
                return APIResponse(
                    success=False,
                    message=f"Error retrieving metrics: {str(e)}"
                )
        
        # Rutas de componentes AEGIS
        @self.app.get("/components", response_model=APIResponse)
        async def get_components(current_user: dict = Depends(get_current_user)):
            """Estado de componentes AEGIS"""
            return APIResponse(
                success=True,
                message="Components status",
                data=self.node_status["components"]
            )
        
        @self.app.post("/components/{component}/start", response_model=APIResponse)
        async def start_component(component: str, current_user: dict = Depends(get_admin_user)):
            """Inicia un componente"""
            # Implementar lógica de inicio de componentes
            self.node_status["components"][component] = True
            return APIResponse(
                success=True,
                message=f"Component {component} started"
            )
        
        @self.app.post("/components/{component}/stop", response_model=APIResponse)
        async def stop_component(component: str, current_user: dict = Depends(get_admin_user)):
            """Detiene un componente"""
            # Implementar lógica de parada de componentes
            self.node_status["components"][component] = False
            return APIResponse(
                success=True,
                message=f"Component {component} stopped"
            )
        
        # WebSocket endpoint
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """Endpoint WebSocket para tiempo real"""
            await self.websocket_manager.connect(websocket, client_id)
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Manejar diferentes tipos de mensajes
                    if message.get("type") == "subscribe":
                        topic = message.get("topic")
                        if topic:
                            self.websocket_manager.subscribe(client_id, topic)
                            await websocket.send_text(json.dumps({
                                "type": "subscribed",
                                "topic": topic,
                                "status": "success"
                            }))
                    
                    elif message.get("type") == "unsubscribe":
                        topic = message.get("topic")
                        if topic:
                            self.websocket_manager.unsubscribe(client_id, topic)
                            await websocket.send_text(json.dumps({
                                "type": "unsubscribed",
                                "topic": topic,
                                "status": "success"
                            }))
                    
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(client_id)
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Inicia el servidor API"""
        self.status = APIStatus.STARTING
        
        try:
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
            server = uvicorn.Server(config)
            
            self.status = APIStatus.RUNNING
            logger.success(f"🚀 AEGIS API Server iniciado en http://{host}:{port}")
            logger.info(f"📚 Documentación disponible en http://{host}:{port}/docs")
            
            await server.serve()
            
        except Exception as e:
            self.status = APIStatus.ERROR
            logger.error(f"Error iniciando servidor API: {e}")
            raise
    
    def update_component_status(self, component: str, status: bool):
        """Actualiza estado de un componente"""
        self.node_status["components"][component] = status
        self.node_status["last_update"] = datetime.now()
        
        # Notificar via WebSocket
        asyncio.create_task(self.websocket_manager.broadcast_to_topic({
            "type": "component_update",
            "component": component,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }, "components"))
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Actualiza métricas del sistema"""
        self.node_status["metrics"].update(metrics)
        self.node_status["last_update"] = datetime.now()
        
        # Notificar via WebSocket
        asyncio.create_task(self.websocket_manager.broadcast_to_topic({
            "type": "metrics_update",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }, "metrics"))
    
    def shutdown(self):
        """Cierra el servidor"""
        self.status = APIStatus.STOPPING
        logger.info("🛑 Cerrando AEGIS API Server...")
        self.status = APIStatus.STOPPED


# Instancia global
api_server = None


def get_api_server() -> AEGISAPIServer:
    """Obtiene la instancia global del servidor API"""
    global api_server
    if api_server is None:
        api_server = AEGISAPIServer()
    return api_server


async def start_api_server(config: Dict[str, Any] = None):
    """Inicia el servidor API AEGIS"""
    try:
        server = get_api_server()
        
        # Configurar desde parámetros
        if config:
            server.config.update(config)
        
        host = config.get("host", "0.0.0.0") if config else "0.0.0.0"
        port = config.get("port", 8000) if config else 8000
        
        await server.start_server(host, port)
        
    except Exception as e:
        logger.error(f"Error iniciando servidor API: {e}")
        raise


if __name__ == "__main__":
    # Demostración del servidor API
    async def demo():
        print("🌐 Demostración del Servidor API AEGIS")
        
        config = {
            "host": "127.0.0.1",
            "port": 8080,
            "cors_origins": ["http://localhost:3000"],
            "trusted_hosts": ["localhost", "127.0.0.1"]
        }
        
        try:
            await start_api_server(config)
        except KeyboardInterrupt:
            print("\n🛑 Servidor detenido por el usuario")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    asyncio.run(demo())