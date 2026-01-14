#!/usr/bin/env python3
"""
Configuración de Seguridad para AEGIS Open AGI
Implementa medidas de seguridad para entorno de producción
"""

import os
import ssl
import secrets
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path

try:
    from logging_config import get_logger
    logger = get_logger("Security")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Configuración de seguridad"""
    # Configuración SSL/TLS
    ssl_enabled: bool = True
    ssl_cert_path: str = "certs/server.crt"
    ssl_key_path: str = "certs/server.key"
    ssl_min_version: str = "TLSv1.2"
    ssl_ciphers: str = "HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA"
    
    # Configuración de CORS
    cors_enabled: bool = True
    cors_origins: List[str] = None
    cors_methods: List[str] = None
    cors_headers: List[str] = None
    cors_max_age: int = 86400
    
    # Configuración de rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hora
    rate_limit_storage: str = "memory"
    
    # Configuración de autenticación
    jwt_secret: str = None
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hora
    jwt_refresh_expiration: int = 86400  # 24 horas
    
    # Configuración de IP filtering
    ip_filtering_enabled: bool = True
    allowed_ips: List[str] = None
    blocked_ips: List[str] = None
    trusted_proxies: List[str] = None
    
    # Configuración de headers de seguridad
    security_headers: Dict[str, str] = None
    
    # Configuración de auditoría
    audit_enabled: bool = True
    audit_log_path: str = "logs/audit.log"
    audit_max_size: int = 100 * 1024 * 1024  # 100MB
    audit_backup_count: int = 10
    
    # Configuración de contraseñas
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    password_history_count: int = 5
    password_expiration_days: int = 90
    
    # Configuración de sesiones
    session_timeout: int = 1800  # 30 minutos
    session_cleanup_interval: int = 3600  # 1 hora
    session_storage: str = "memory"
    
    # Configuración de validación de entrada
    input_validation_enabled: bool = True
    input_max_length: int = 10000
    input_allowed_chars: str = None
    input_blocked_patterns: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["https://localhost:8080", "https://127.0.0.1:8080"]
        
        if self.cors_methods is None:
            self.cors_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        
        if self.cors_headers is None:
            self.cors_headers = ["Content-Type", "Authorization", "X-Requested-With"]
        
        if self.jwt_secret is None:
            self.jwt_secret = secrets.token_urlsafe(64)
        
        if self.allowed_ips is None:
            self.allowed_ips = ["127.0.0.1", "::1", "192.168.1.0/24", "10.0.0.0/8"]
        
        if self.blocked_ips is None:
            self.blocked_ips = []
        
        if self.trusted_proxies is None:
            self.trusted_proxies = ["127.0.0.1", "::1"]
        
        if self.security_headers is None:
            self.security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; media-src 'self'; object-src 'none'; child-src 'none'; form-action 'self'; base-uri 'self'; frame-ancestors 'none';",
                'Referrer-Policy': 'strict-origin-when-cross-origin',
                'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
            }
        
        if self.input_blocked_patterns is None:
            self.input_blocked_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'vbscript:',
                r'onload\s*=',
                r'onerror\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>',
                r'<form[^>]*>',
                r'union\s+select',
                r'drop\s+table',
                r'insert\s+into',
                r'delete\s+from',
                r'update\s+.*\s+set',
                r'exec\s+\(',
                r'xp_cmdshell',
                r'sp_executesql'
            ]

class SecurityManager:
    """Gestor de seguridad para AEGIS"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.audit_logger = self._setup_audit_logging()
        self.rate_limiter = self._setup_rate_limiter()
        self.session_manager = self._setup_session_manager()
        
        logger.info("🔐 Security Manager inicializado")
    
    def _setup_audit_logging(self):
        """Configura el logging de auditoría"""
        if not self.config.audit_enabled:
            return None
        
        audit_logger = logging.getLogger('audit')
        audit_handler = logging.handlers.RotatingFileHandler(
            self.config.audit_log_path,
            maxBytes=self.config.audit_max_size,
            backupCount=self.config.audit_backup_count
        )
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        
        return audit_logger
    
    def _setup_rate_limiter(self):
        """Configura el rate limiter"""
        if not self.config.rate_limit_enabled:
            return None
        
        # Implementación simple de rate limiter
        from collections import defaultdict
        import time
        
        class SimpleRateLimiter:
            def __init__(self, max_requests=100, window=3600):
                self.max_requests = max_requests
                self.window = window
                self.requests = defaultdict(list)
            
            def is_allowed(self, identifier):
                now = time.time()
                # Limpiar requests antiguos
                self.requests[identifier] = [
                    req_time for req_time in self.requests[identifier]
                    if now - req_time < self.window
                ]
                
                # Verificar límite
                if len(self.requests[identifier]) >= self.max_requests:
                    return False
                
                # Agregar request actual
                self.requests[identifier].append(now)
                return True
        
        return SimpleRateLimiter(self.config.rate_limit_requests, self.config.rate_limit_window)
    
    def _setup_session_manager(self):
        """Configura el gestor de sesiones"""
        # Implementación simple de gestor de sesiones
        class SimpleSessionManager:
            def __init__(self, timeout=1800):
                self.sessions = {}
                self.timeout = timeout
            
            def create_session(self, user_id):
                session_id = secrets.token_urlsafe(32)
                self.sessions[session_id] = {
                    'user_id': user_id,
                    'created_at': datetime.now(),
                    'last_activity': datetime.now()
                }
                return session_id
            
            def validate_session(self, session_id):
                if session_id not in self.sessions:
                    return None
                
                session = self.sessions[session_id]
                if datetime.now() - session['last_activity'] > timedelta(seconds=self.timeout):
                    del self.sessions[session_id]
                    return None
                
                session['last_activity'] = datetime.now()
                return session['user_id']
            
            def cleanup_expired_sessions(self):
                expired = []
                for session_id, session in self.sessions.items():
                    if datetime.now() - session['last_activity'] > timedelta(seconds=self.timeout):
                        expired.append(session_id)
                
                for session_id in expired:
                    del self.sessions[session_id]
                
                return len(expired)
        
        return SimpleSessionManager(self.config.session_timeout)
    
    def validate_ip_address(self, ip_address: str) -> bool:
        """Valida si una dirección IP está permitida"""
        if not self.config.ip_filtering_enabled:
            return True
        
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Verificar IPs bloqueadas
            for blocked_ip in self.config.blocked_ips:
                if ip == ipaddress.ip_address(blocked_ip):
                    self.log_security_event('blocked_ip', {'ip': ip_address})
                    return False
            
            # Verificar IPs permitidas
            for allowed_ip in self.config.allowed_ips:
                if '/' in allowed_ip:
                    network = ipaddress.ip_network(allowed_ip, strict=False)
                    if ip in network:
                        return True
                elif ip == ipaddress.ip_address(allowed_ip):
                    return True
            
            self.log_security_event('unauthorized_ip', {'ip': ip_address})
            return False
            
        except ValueError:
            logger.error(f"❌ Dirección IP inválida: {ip_address}")
            return False
    
    def validate_rate_limit(self, identifier: str) -> bool:
        """Valida el límite de tasa"""
        if not self.config.rate_limit_enabled or not self.rate_limiter:
            return True
        
        is_allowed = self.rate_limiter.is_allowed(identifier)
        
        if not is_allowed:
            self.log_security_event('rate_limit_exceeded', {'identifier': identifier})
        
        return is_allowed
    
    def validate_input(self, data: str) -> bool:
        """Valida la entrada de datos"""
        if not self.config.input_validation_enabled:
            return True
        
        # Verificar longitud máxima
        if len(data) > self.config.input_max_length:
            self.log_security_event('input_too_long', {'length': len(data)})
            return False
        
        # Verificar patrones bloqueados
        import re
        for pattern in self.config.input_blocked_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                self.log_security_event('malicious_input', {'pattern': pattern, 'data': data[:100]})
                return False
        
        return True
    
    def validate_password(self, password: str) -> Dict[str, any]:
        """Valida la fortaleza de una contraseña"""
        result = {'valid': True, 'errors': []}
        
        if len(password) < self.config.password_min_length:
            result['errors'].append(f"Mínimo {self.config.password_min_length} caracteres")
            result['valid'] = False
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            result['errors'].append("Debe contener mayúsculas")
            result['valid'] = False
        
        if self.config.password_require_lowercase and not any(c.islower() for c in password):
            result['errors'].append("Debe contener minúsculas")
            result['valid'] = False
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            result['errors'].append("Debe contener números")
            result['valid'] = False
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            result['errors'].append("Debe contener caracteres especiales")
            result['valid'] = False
        
        return result
    
    def hash_password(self, password: str) -> str:
        """Genera hash seguro de contraseña"""
        salt = secrets.token_bytes(32)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt.hex() + key.hex()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verifica una contraseña contra su hash"""
        try:
            salt = bytes.fromhex(password_hash[:64])
            key = bytes.fromhex(password_hash[64:])
            
            new_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return new_key == key
        except Exception:
            return False
    
    def generate_jwt_token(self, payload: Dict) -> str:
        """Genera un token JWT"""
        import jwt
        
        payload.update({
            'exp': datetime.utcnow() + timedelta(seconds=self.config.jwt_expiration),
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)
        })
        
        token = jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
        self.log_security_event('jwt_created', {'jti': payload['jti']})
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verifica un token JWT"""
        import jwt
        
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            self.log_security_event('jwt_verified', {'jti': payload.get('jti', 'unknown')})
            return payload
        except jwt.ExpiredSignatureError:
            self.log_security_event('jwt_expired', {'token': token[:20]})
            return None
        except jwt.InvalidTokenError:
            self.log_security_event('jwt_invalid', {'token': token[:20]})
            return None
    
    def get_security_headers(self) -> Dict[str, str]:
        """Obtiene headers de seguridad"""
        return self.config.security_headers.copy()
    
    def log_security_event(self, event_type: str, details: Dict):
        """Registra evento de seguridad"""
        if self.audit_logger:
            self.audit_logger.info(f"SECURITY_EVENT: {event_type} - {details}")
        
        logger.info(f"🔒 Evento de seguridad: {event_type} - {details}")
    
    def get_security_status(self) -> Dict[str, any]:
        """Obtiene el estado de seguridad del sistema"""
        return {
            'ssl_enabled': self.config.ssl_enabled,
            'cors_enabled': self.config.cors_enabled,
            'rate_limiting_enabled': self.config.rate_limit_enabled,
            'ip_filtering_enabled': self.config.ip_filtering_enabled,
            'input_validation_enabled': self.config.input_validation_enabled,
            'audit_enabled': self.config.audit_enabled,
            'session_manager_active': self.session_manager is not None,
            'rate_limiter_active': self.rate_limiter is not None,
            'audit_logger_active': self.audit_logger is not None,
            'security_headers_count': len(self.config.security_headers),
            'allowed_ips_count': len(self.config.allowed_ips),
            'blocked_ips_count': len(self.config.blocked_ips),
            'timestamp': datetime.now().isoformat()
        }

# Instancia global
_security_manager = None

def get_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Obtiene la instancia global del SecurityManager"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager(config)
    return _security_manager

def validate_ip_address(ip_address: str) -> bool:
    """Valida una dirección IP"""
    manager = get_security_manager()
    return manager.validate_ip_address(ip_address)

def validate_rate_limit(identifier: str) -> bool:
    """Valida el límite de tasa"""
    manager = get_security_manager()
    return manager.validate_rate_limit(identifier)

def validate_input(data: str) -> bool:
    """Valida entrada de datos"""
    manager = get_security_manager()
    return manager.validate_input(data)

def get_security_headers() -> Dict[str, str]:
    """Obtiene headers de seguridad"""
    manager = get_security_manager()
    return manager.get_security_headers()

def log_security_event(event_type: str, details: Dict):
    """Registra evento de seguridad"""
    manager = get_security_manager()
    manager.log_security_event(event_type, details)