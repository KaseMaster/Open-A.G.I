"""
AEGIS Security Middleware
Rate limiting, input validation, and security hardening
"""

import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    max_requests: int
    window_seconds: int
    burst_allowance: int = 0


@dataclass
class ClientRecord:
    """Record of client activity"""
    client_id: str
    request_timestamps: deque = field(default_factory=deque)
    total_requests: int = 0
    blocked_until: Optional[float] = None
    violation_count: int = 0


class RateLimiter:
    """Token bucket rate limiter with burst support"""
    
    def __init__(self, default_rule: Optional[RateLimitRule] = None):
        self.default_rule = default_rule or RateLimitRule(
            max_requests=100,
            window_seconds=60,
            burst_allowance=20
        )
        self.rules: Dict[str, RateLimitRule] = {}
        self.clients: Dict[str, ClientRecord] = {}
        
        self.total_requests = 0
        self.total_blocked = 0
    
    def set_rule(self, endpoint: str, rule: RateLimitRule):
        """Set rate limit rule for specific endpoint"""
        self.rules[endpoint] = rule
    
    def check_rate_limit(
        self,
        client_id: str,
        endpoint: str = "default"
    ) -> tuple[bool, Optional[str]]:
        """Check if request is allowed under rate limit"""
        self.total_requests += 1
        
        if client_id not in self.clients:
            self.clients[client_id] = ClientRecord(client_id=client_id)
        
        client = self.clients[client_id]
        
        if client.blocked_until and time.time() < client.blocked_until:
            remaining = int(client.blocked_until - time.time())
            self.total_blocked += 1
            return False, f"Client blocked for {remaining} more seconds"
        
        client.blocked_until = None
        
        rule = self.rules.get(endpoint, self.default_rule)
        current_time = time.time()
        cutoff_time = current_time - rule.window_seconds
        
        while client.request_timestamps and client.request_timestamps[0] < cutoff_time:
            client.request_timestamps.popleft()
        
        request_count = len(client.request_timestamps)
        
        effective_limit = rule.max_requests + rule.burst_allowance
        
        if request_count >= effective_limit:
            client.violation_count += 1
            
            if client.violation_count >= 3:
                client.blocked_until = current_time + 300
                logger.warning(f"Client {client_id} blocked for repeated violations")
            
            self.total_blocked += 1
            return False, f"Rate limit exceeded: {request_count}/{effective_limit} requests in {rule.window_seconds}s"
        
        client.request_timestamps.append(current_time)
        client.total_requests += 1
        
        return True, None
    
    def reset_client(self, client_id: str):
        """Reset rate limit for a client"""
        if client_id in self.clients:
            del self.clients[client_id]
    
    def get_client_stats(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a client"""
        if client_id not in self.clients:
            return None
        
        client = self.clients[client_id]
        
        return {
            "client_id": client_id,
            "total_requests": client.total_requests,
            "current_window_requests": len(client.request_timestamps),
            "violation_count": client.violation_count,
            "is_blocked": client.blocked_until is not None and time.time() < client.blocked_until,
            "blocked_until": client.blocked_until
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall rate limiter statistics"""
        active_clients = len(self.clients)
        blocked_clients = sum(
            1 for c in self.clients.values()
            if c.blocked_until and time.time() < c.blocked_until
        )
        
        block_rate = self.total_blocked / self.total_requests if self.total_requests > 0 else 0.0
        
        return {
            "total_requests": self.total_requests,
            "total_blocked": self.total_blocked,
            "block_rate": block_rate,
            "active_clients": active_clients,
            "blocked_clients": blocked_clients,
            "rules_count": len(self.rules)
        }


class InputValidator:
    """Input validation and sanitization"""
    
    def __init__(self):
        self.patterns = {
            "alphanumeric": re.compile(r'^[a-zA-Z0-9_-]+$'),
            "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            "hex": re.compile(r'^[0-9a-fA-F]+$'),
            "ip_address": re.compile(r'^(\d{1,3}\.){3}\d{1,3}$'),
            "safe_string": re.compile(r'^[a-zA-Z0-9\s\-_.,:;!?()]+$')
        }
        
        self.max_lengths = {
            "username": 50,
            "email": 100,
            "password": 128,
            "text": 1000,
            "description": 5000,
            "hash": 128
        }
        
        self.dangerous_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'(union|select|insert|update|delete|drop)\s+', re.IGNORECASE),
            re.compile(r'[;\'"].*?(--|\#|/\*)')
        ]
    
    def validate_string(
        self,
        value: str,
        field_type: str = "safe_string",
        min_length: int = 0,
        max_length: Optional[int] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate string input"""
        if not isinstance(value, str):
            return False, "Value must be a string"
        
        if len(value) < min_length:
            return False, f"Value too short (min: {min_length})"
        
        if max_length is None:
            max_length = self.max_lengths.get(field_type, 1000)
        
        if len(value) > max_length:
            return False, f"Value too long (max: {max_length})"
        
        for pattern in self.dangerous_patterns:
            if pattern.search(value):
                return False, "Potentially dangerous content detected"
        
        if field_type in self.patterns:
            if not self.patterns[field_type].match(value):
                return False, f"Invalid format for {field_type}"
        
        return True, None
    
    def validate_integer(
        self,
        value: Any,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate integer input"""
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            return False, "Value must be an integer"
        
        if min_value is not None and int_value < min_value:
            return False, f"Value too small (min: {min_value})"
        
        if max_value is not None and int_value > max_value:
            return False, f"Value too large (max: {max_value})"
        
        return True, None
    
    def validate_float(
        self,
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate float input"""
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            return False, "Value must be a number"
        
        if min_value is not None and float_value < min_value:
            return False, f"Value too small (min: {min_value})"
        
        if max_value is not None and float_value > max_value:
            return False, f"Value too large (max: {max_value})"
        
        return True, None
    
    def validate_dict(
        self,
        value: Any,
        required_keys: Optional[List[str]] = None,
        allowed_keys: Optional[List[str]] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate dictionary input"""
        if not isinstance(value, dict):
            return False, "Value must be a dictionary"
        
        if required_keys:
            missing_keys = set(required_keys) - set(value.keys())
            if missing_keys:
                return False, f"Missing required keys: {missing_keys}"
        
        if allowed_keys:
            extra_keys = set(value.keys()) - set(allowed_keys)
            if extra_keys:
                return False, f"Unexpected keys: {extra_keys}"
        
        return True, None
    
    def sanitize_string(self, value: str) -> str:
        """Sanitize string by removing dangerous content"""
        sanitized = value
        
        for pattern in self.dangerous_patterns:
            sanitized = pattern.sub('', sanitized)
        
        sanitized = sanitized.strip()
        
        return sanitized
    
    def validate_hash(self, value: str, algorithm: str = "sha256") -> tuple[bool, Optional[str]]:
        """Validate cryptographic hash"""
        expected_lengths = {
            "md5": 32,
            "sha1": 40,
            "sha256": 64,
            "sha512": 128
        }
        
        if algorithm not in expected_lengths:
            return False, f"Unknown hash algorithm: {algorithm}"
        
        expected_length = expected_lengths[algorithm]
        
        if len(value) != expected_length:
            return False, f"Invalid {algorithm} hash length"
        
        if not self.patterns["hex"].match(value):
            return False, "Hash must be hexadecimal"
        
        return True, None


class SecurityMiddleware:
    """Comprehensive security middleware"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.input_validator = InputValidator()
        
        self.blocked_ips: set = set()
        self.suspicious_activity: Dict[str, List[float]] = {}
        
        self.security_events: List[Dict[str, Any]] = []
        self.max_events = 1000
    
    def check_request_security(
        self,
        client_id: str,
        endpoint: str,
        params: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Comprehensive security check for request"""
        if client_id in self.blocked_ips:
            self._log_security_event("blocked_ip", client_id, endpoint)
            return False, "Client IP is blocked"
        
        allowed, message = self.rate_limiter.check_rate_limit(client_id, endpoint)
        if not allowed:
            self._log_security_event("rate_limit", client_id, endpoint)
            return False, message
        
        for key, value in params.items():
            if isinstance(value, str):
                is_valid, error = self.input_validator.validate_string(value, "safe_string")
                if not is_valid:
                    self._log_security_event("invalid_input", client_id, endpoint, {
                        "field": key,
                        "error": error
                    })
                    return False, f"Invalid input for {key}: {error}"
        
        return True, None
    
    def block_client(self, client_id: str, duration_seconds: int = 3600):
        """Block a client for specified duration"""
        self.blocked_ips.add(client_id)
        self._log_security_event("client_blocked", client_id, "", {
            "duration": duration_seconds
        })
        logger.warning(f"Blocked client: {client_id} for {duration_seconds}s")
    
    def unblock_client(self, client_id: str):
        """Unblock a client"""
        if client_id in self.blocked_ips:
            self.blocked_ips.remove(client_id)
            self._log_security_event("client_unblocked", client_id, "")
    
    def detect_suspicious_activity(
        self,
        client_id: str,
        threshold: int = 10,
        window_seconds: int = 60
    ) -> bool:
        """Detect suspicious activity patterns"""
        current_time = time.time()
        
        if client_id not in self.suspicious_activity:
            self.suspicious_activity[client_id] = []
        
        activity_log = self.suspicious_activity[client_id]
        
        activity_log = [t for t in activity_log if current_time - t < window_seconds]
        self.suspicious_activity[client_id] = activity_log
        
        if len(activity_log) >= threshold:
            self._log_security_event("suspicious_activity", client_id, "", {
                "event_count": len(activity_log),
                "window_seconds": window_seconds
            })
            return True
        
        activity_log.append(current_time)
        return False
    
    def _log_security_event(
        self,
        event_type: str,
        client_id: str,
        endpoint: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log security event"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "client_id": client_id,
            "endpoint": endpoint,
            "metadata": metadata or {}
        }
        
        self.security_events.append(event)
        
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        recent_events = self.security_events[-100:]
        
        event_counts = {}
        for event in recent_events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.security_events),
            "recent_event_counts": event_counts,
            "blocked_clients": len(self.blocked_ips),
            "rate_limiter_stats": self.rate_limiter.get_stats()
        }
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent security events"""
        return self.security_events[-limit:]
