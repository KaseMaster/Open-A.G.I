"""
Sistema de Alertas Inteligentes AEGIS
Alertas predictivas con IA, múltiples canales y análisis de patrones
"""

import asyncio
import json
import time
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from email.mime.text import MIMEText as MimeText
    from email.mime.multipart import MIMEMultipart as MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logger.warning("Módulos de email no disponibles - notificaciones por email deshabilitadas")

class AlertSeverity(Enum):
    """Niveles de severidad de alertas"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertCategory(Enum):
    """Categorías de alertas"""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    NETWORK = "network"
    APPLICATION = "application"
    PREDICTION = "prediction"

class AlertStatus(Enum):
    """Estados de alertas"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    """Estructura de una alerta"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    source: str
    timestamp: float
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = None
    predicted: bool = False
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AlertRule:
    """Regla de alerta"""
    id: str
    name: str
    condition: str
    severity: AlertSeverity
    category: AlertCategory
    enabled: bool = True
    cooldown_minutes: int = 5
    threshold_count: int = 1
    time_window_minutes: int = 1
    
class NotificationChannel:
    """Canal de notificación base"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
    
    async def send(self, alert: Alert) -> bool:
        """Enviar notificación"""
        raise NotImplementedError

class EmailChannel(NotificationChannel):
    """Canal de notificación por email"""
    
    async def send(self, alert: Alert) -> bool:
        try:
            if not EMAIL_AVAILABLE:
                logger.warning("Módulos de email no disponibles - saltando notificación por email")
                return False
                
            smtp_server = self.config.get("smtp_server", "localhost")
            smtp_port = self.config.get("smtp_port", 587)
            username = self.config.get("username")
            password = self.config.get("password")
            from_email = self.config.get("from_email")
            to_emails = self.config.get("to_emails", [])
            
            if not all([username, password, from_email, to_emails]):
                logger.warning("Configuración de email incompleta")
                return False
            
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ", ".join(to_emails)
            msg['Subject'] = f"[AEGIS {alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alerta AEGIS

Título: {alert.title}
Descripción: {alert.description}
Severidad: {alert.severity.value.upper()}
Categoría: {alert.category.value}
Fuente: {alert.source}
Tiempo: {datetime.fromtimestamp(alert.timestamp)}
Predicción: {'Sí' if alert.predicted else 'No'}
Confianza: {alert.confidence:.2%}

Metadatos:
{json.dumps(alert.metadata, indent=2)}
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error enviando email: {e}")
            return False

class WebhookChannel(NotificationChannel):
    """Canal de notificación por webhook"""
    
    async def send(self, alert: Alert) -> bool:
        try:
            url = self.config.get("url")
            headers = self.config.get("headers", {})
            timeout = self.config.get("timeout", 10)
            
            if not url:
                logger.warning("URL de webhook no configurada")
                return False
            
            payload = {
                "alert": asdict(alert),
                "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat(),
                "aegis_node": self.config.get("node_id", "unknown")
            }
            
            response = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=timeout
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error enviando webhook: {e}")
            return False

class SlackChannel(NotificationChannel):
    """Canal de notificación para Slack"""
    
    async def send(self, alert: Alert) -> bool:
        try:
            webhook_url = self.config.get("webhook_url")
            
            if not webhook_url:
                logger.warning("Webhook URL de Slack no configurada")
                return False
            
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500", 
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.EMERGENCY: "#8b0000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "title": f"AEGIS Alert: {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Severidad", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Categoría", "value": alert.category.value, "short": True},
                        {"title": "Fuente", "value": alert.source, "short": True},
                        {"title": "Predicción", "value": "Sí" if alert.predicted else "No", "short": True}
                    ],
                    "timestamp": int(alert.timestamp)
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error enviando a Slack: {e}")
            return False

class PatternAnalyzer:
    """Analizador de patrones para predicciones"""
    
    def __init__(self):
        self.patterns = {}
        self.learning_window = 3600  # 1 hora
        
    def analyze_pattern(self, alerts: List[Alert]) -> Dict[str, float]:
        """Analiza patrones en alertas históricas"""
        patterns = {}
        
        # Análisis temporal
        time_patterns = self._analyze_time_patterns(alerts)
        patterns.update(time_patterns)
        
        # Análisis de secuencias
        sequence_patterns = self._analyze_sequences(alerts)
        patterns.update(sequence_patterns)
        
        # Análisis de correlaciones
        correlation_patterns = self._analyze_correlations(alerts)
        patterns.update(correlation_patterns)
        
        return patterns
    
    def _analyze_time_patterns(self, alerts: List[Alert]) -> Dict[str, float]:
        """Analiza patrones temporales"""
        patterns = {}
        
        # Agrupar por hora del día
        hourly_counts = {}
        for alert in alerts:
            hour = datetime.fromtimestamp(alert.timestamp).hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        # Encontrar horas pico
        if hourly_counts:
            max_count = max(hourly_counts.values())
            for hour, count in hourly_counts.items():
                if count > max_count * 0.7:  # 70% del máximo
                    patterns[f"peak_hour_{hour}"] = count / max_count
        
        return patterns
    
    def _analyze_sequences(self, alerts: List[Alert]) -> Dict[str, float]:
        """Analiza secuencias de alertas"""
        patterns = {}
        
        # Ordenar por timestamp
        sorted_alerts = sorted(alerts, key=lambda x: x.timestamp)
        
        # Buscar secuencias comunes
        sequences = []
        for i in range(len(sorted_alerts) - 1):
            current = sorted_alerts[i]
            next_alert = sorted_alerts[i + 1]
            
            # Si las alertas están dentro de 5 minutos
            if next_alert.timestamp - current.timestamp <= 300:
                sequence = f"{current.category.value}->{next_alert.category.value}"
                sequences.append(sequence)
        
        # Calcular frecuencias
        sequence_counts = {}
        for seq in sequences:
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
        
        total_sequences = len(sequences)
        if total_sequences > 0:
            for seq, count in sequence_counts.items():
                if count >= 2:  # Al menos 2 ocurrencias
                    patterns[f"sequence_{seq}"] = count / total_sequences
        
        return patterns
    
    def _analyze_correlations(self, alerts: List[Alert]) -> Dict[str, float]:
        """Analiza correlaciones entre tipos de alertas"""
        patterns = {}
        
        # Agrupar alertas por ventanas de tiempo
        time_windows = {}
        window_size = 300  # 5 minutos
        
        for alert in alerts:
            window = int(alert.timestamp // window_size)
            if window not in time_windows:
                time_windows[window] = []
            time_windows[window].append(alert)
        
        # Buscar correlaciones
        correlations = {}
        for window_alerts in time_windows.values():
            if len(window_alerts) >= 2:
                categories = [alert.category.value for alert in window_alerts]
                unique_categories = list(set(categories))
                
                if len(unique_categories) >= 2:
                    correlation_key = "+".join(sorted(unique_categories))
                    correlations[correlation_key] = correlations.get(correlation_key, 0) + 1
        
        # Normalizar correlaciones
        total_windows = len(time_windows)
        if total_windows > 0:
            for corr, count in correlations.items():
                if count >= 2:
                    patterns[f"correlation_{corr}"] = count / total_windows
        
        return patterns

class PredictiveEngine:
    """Motor predictivo de alertas"""
    
    def __init__(self):
        self.analyzer = PatternAnalyzer()
        self.prediction_threshold = 0.7
        self.models = {}
        
    def train(self, historical_alerts: List[Alert]):
        """Entrena el motor con alertas históricas"""
        logger.info(f"Entrenando motor predictivo con {len(historical_alerts)} alertas")
        
        patterns = self.analyzer.analyze_pattern(historical_alerts)
        self.models = patterns
        
        logger.info(f"Patrones identificados: {len(patterns)}")
        
    def predict(self, current_context: Dict[str, Any]) -> List[Alert]:
        """Predice posibles alertas futuras"""
        predictions = []
        
        current_time = time.time()
        current_hour = datetime.fromtimestamp(current_time).hour
        
        # Predicciones basadas en patrones temporales
        peak_pattern = f"peak_hour_{current_hour}"
        if peak_pattern in self.models:
            confidence = self.models[peak_pattern]
            if confidence >= self.prediction_threshold:
                prediction = Alert(
                    id=f"pred_{int(current_time)}",
                    title="Posible incremento de alertas",
                    description=f"Patrón histórico indica alta probabilidad de alertas a las {current_hour}:00",
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.PREDICTION,
                    source="predictive_engine",
                    timestamp=current_time,
                    predicted=True,
                    confidence=confidence
                )
                predictions.append(prediction)
        
        # Predicciones basadas en métricas actuales
        if "cpu_usage" in current_context:
            cpu_usage = current_context["cpu_usage"]
            if cpu_usage > 70:  # CPU alta
                confidence = min((cpu_usage - 70) / 30, 1.0)
                if confidence >= 0.5:
                    prediction = Alert(
                        id=f"pred_cpu_{int(current_time)}",
                        title="Posible sobrecarga de CPU",
                        description=f"CPU al {cpu_usage}%, posible alerta crítica en 5-10 minutos",
                        severity=AlertSeverity.WARNING,
                        category=AlertCategory.PREDICTION,
                        source="predictive_engine",
                        timestamp=current_time,
                        predicted=True,
                        confidence=confidence,
                        metadata={"current_cpu": cpu_usage}
                    )
                    predictions.append(prediction)
        
        return predictions

class AlertStorage:
    """Almacenamiento de alertas"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Inicializa la base de datos"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp REAL NOT NULL,
                status TEXT NOT NULL,
                predicted INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                metadata TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts(timestamp);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON alerts(category);
        """)
        
        conn.commit()
        conn.close()
    
    def store_alert(self, alert: Alert):
        """Almacena una alerta"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO alerts 
            (id, title, description, severity, category, source, timestamp, 
             status, predicted, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.id, alert.title, alert.description,
            alert.severity.value, alert.category.value, alert.source,
            alert.timestamp, alert.status.value,
            1 if alert.predicted else 0, alert.confidence,
            json.dumps(alert.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def get_alerts(self, 
                   limit: int = 100,
                   category: Optional[AlertCategory] = None,
                   severity: Optional[AlertSeverity] = None,
                   since: Optional[float] = None) -> List[Alert]:
        """Obtiene alertas con filtros"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category.value)
            
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
            
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in rows:
            alert = Alert(
                id=row[0],
                title=row[1],
                description=row[2],
                severity=AlertSeverity(row[3]),
                category=AlertCategory(row[4]),
                source=row[5],
                timestamp=row[6],
                status=AlertStatus(row[7]),
                predicted=bool(row[8]),
                confidence=row[9],
                metadata=json.loads(row[10]) if row[10] else {}
            )
            alerts.append(alert)
        
        return alerts

class AEGISAlertSystem:
    """Sistema principal de alertas AEGIS"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage = AlertStorage(config.get("storage_path", "aegis_alerts.db"))
        self.predictive_engine = PredictiveEngine()
        self.channels = {}
        self.rules = {}
        self.running = False
        self.alert_queue = asyncio.Queue()
        self.rule_cooldowns = {}
        
        self._setup_channels()
        self._setup_rules()
        
    def _setup_channels(self):
        """Configura canales de notificación"""
        channels_config = self.config.get("channels", {})
        
        # Si channels es una lista vacía, convertir a dict vacío
        if isinstance(channels_config, list):
            channels_config = {}
        
        for name, channel_config in channels_config.items():
            channel_type = channel_config.get("type")
            
            if channel_type == "email":
                self.channels[name] = EmailChannel(name, channel_config)
            elif channel_type == "webhook":
                self.channels[name] = WebhookChannel(name, channel_config)
            elif channel_type == "slack":
                self.channels[name] = SlackChannel(name, channel_config)
            else:
                logger.warning(f"Tipo de canal desconocido: {channel_type}")
    
    def _setup_rules(self):
        """Configura reglas de alertas"""
        rules_config = self.config.get("rules", [])
        
        for rule_config in rules_config:
            rule = AlertRule(**rule_config)
            self.rules[rule.id] = rule
    
    async def start(self):
        """Inicia el sistema de alertas"""
        logger.info("🚀 Iniciando sistema de alertas inteligentes...")
        
        self.running = True
        
        # Entrenar motor predictivo
        await self._train_predictive_engine()
        
        # Iniciar procesamiento de alertas
        asyncio.create_task(self._process_alerts())
        
        # Iniciar predicciones periódicas
        asyncio.create_task(self._periodic_predictions())
        
        logger.info("✅ Sistema de alertas iniciado")
    
    async def stop(self):
        """Detiene el sistema de alertas"""
        logger.info("🛑 Deteniendo sistema de alertas...")
        self.running = False
        logger.info("✅ Sistema de alertas detenido")
    
    async def _train_predictive_engine(self):
        """Entrena el motor predictivo"""
        try:
            # Obtener alertas de los últimos 30 días
            since = time.time() - (30 * 24 * 3600)
            historical_alerts = self.storage.get_alerts(limit=10000, since=since)
            
            if len(historical_alerts) >= 10:
                self.predictive_engine.train(historical_alerts)
            else:
                logger.info("Insuficientes datos históricos para entrenamiento")
                
        except Exception as e:
            logger.error(f"Error entrenando motor predictivo: {e}")
    
    async def _process_alerts(self):
        """Procesa alertas de la cola"""
        while self.running:
            try:
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # Almacenar alerta
                self.storage.store_alert(alert)
                
                # Enviar notificaciones
                await self._send_notifications(alert)
                
                logger.info(f"Alerta procesada: {alert.title} [{alert.severity.value}]")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error procesando alerta: {e}")
    
    async def _send_notifications(self, alert: Alert):
        """Envía notificaciones por todos los canales habilitados"""
        for channel_name, channel in self.channels.items():
            if not channel.enabled:
                continue
                
            try:
                success = await channel.send(alert)
                if success:
                    logger.debug(f"Notificación enviada por {channel_name}")
                else:
                    logger.warning(f"Fallo enviando por {channel_name}")
                    
            except Exception as e:
                logger.error(f"Error en canal {channel_name}: {e}")
    
    async def _periodic_predictions(self):
        """Ejecuta predicciones periódicas"""
        while self.running:
            try:
                # Obtener contexto actual (esto se conectaría con el sistema de métricas)
                current_context = await self._get_current_context()
                
                # Generar predicciones
                predictions = self.predictive_engine.predict(current_context)
                
                # Procesar predicciones
                for prediction in predictions:
                    await self.create_alert(
                        prediction.title,
                        prediction.description,
                        prediction.severity,
                        prediction.category,
                        prediction.source,
                        metadata=prediction.metadata,
                        predicted=True,
                        confidence=prediction.confidence
                    )
                
                # Esperar antes de la siguiente predicción
                await asyncio.sleep(self.config.get("prediction_interval", 300))  # 5 minutos
                
            except Exception as e:
                logger.error(f"Error en predicciones periódicas: {e}")
                await asyncio.sleep(60)
    
    async def _get_current_context(self) -> Dict[str, Any]:
        """Obtiene el contexto actual del sistema"""
        # Aquí se conectaría con el sistema de métricas
        # Por ahora retornamos datos simulados
        return {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "disk_usage": 30.0,
            "network_connections": 150
        }
    
    async def create_alert(self,
                          title: str,
                          description: str,
                          severity: AlertSeverity,
                          category: AlertCategory,
                          source: str,
                          metadata: Optional[Dict[str, Any]] = None,
                          predicted: bool = False,
                          confidence: float = 0.0) -> str:
        """Crea una nueva alerta"""
        
        alert_id = f"alert_{int(time.time() * 1000)}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            category=category,
            source=source,
            timestamp=time.time(),
            metadata=metadata or {},
            predicted=predicted,
            confidence=confidence
        )
        
        # Añadir a la cola de procesamiento
        await self.alert_queue.put(alert)
        
        return alert_id
    
    def evaluate_rule(self, rule_id: str, context: Dict[str, Any]) -> bool:
        """Evalúa una regla de alerta"""
        if rule_id not in self.rules:
            return False
            
        rule = self.rules[rule_id]
        
        if not rule.enabled:
            return False
        
        # Verificar cooldown
        cooldown_key = f"{rule_id}_{rule.condition}"
        if cooldown_key in self.rule_cooldowns:
            last_trigger = self.rule_cooldowns[cooldown_key]
            if time.time() - last_trigger < rule.cooldown_minutes * 60:
                return False
        
        # Evaluar condición (implementación básica)
        try:
            # Aquí se implementaría un evaluador de expresiones más sofisticado
            result = eval(rule.condition, {"__builtins__": {}}, context)
            
            if result:
                self.rule_cooldowns[cooldown_key] = time.time()
                return True
                
        except Exception as e:
            logger.error(f"Error evaluando regla {rule_id}: {e}")
        
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de alertas"""
        try:
            # Alertas de las últimas 24 horas
            since_24h = time.time() - (24 * 3600)
            recent_alerts = self.storage.get_alerts(limit=10000, since=since_24h)
            
            stats = {
                "total_24h": len(recent_alerts),
                "by_severity": {},
                "by_category": {},
                "predictions": 0,
                "accuracy": 0.0
            }
            
            # Contar por severidad y categoría
            for alert in recent_alerts:
                severity = alert.severity.value
                category = alert.category.value
                
                stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
                stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
                
                if alert.predicted:
                    stats["predictions"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}

# Instancia global del sistema de alertas
_alert_system: Optional[AEGISAlertSystem] = None

async def start_alert_system(config: Dict[str, Any]):
    """Inicia el sistema de alertas"""
    global _alert_system
    
    _alert_system = AEGISAlertSystem(config)
    await _alert_system.start()

async def stop_alert_system():
    """Detiene el sistema de alertas"""
    global _alert_system
    
    if _alert_system:
        await _alert_system.stop()
        _alert_system = None

def get_alert_system() -> Optional[AEGISAlertSystem]:
    """Obtiene la instancia del sistema de alertas"""
    return _alert_system

# Decorador para alertas automáticas
def alert_on_error(severity: AlertSeverity = AlertSeverity.WARNING,
                   category: AlertCategory = AlertCategory.APPLICATION):
    """Decorador que crea alertas automáticamente en caso de error"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if _alert_system:
                    await _alert_system.create_alert(
                        title=f"Error en {func.__name__}",
                        description=f"Excepción: {str(e)}",
                        severity=severity,
                        category=category,
                        source=f"{func.__module__}.{func.__name__}",
                        metadata={"exception_type": type(e).__name__}
                    )
                raise
        return wrapper
    return decorator

if __name__ == "__main__":
    # Configuración de ejemplo
    config = {
        "storage_path": "aegis_alerts.db",
        "prediction_interval": 300,
        "channels": {
            "email": {
                "type": "email",
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "alerts@aegis.com",
                "password": "password",
                "from_email": "alerts@aegis.com",
                "to_emails": ["admin@aegis.com"]
            },
            "slack": {
                "type": "slack",
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/services/..."
            }
        },
        "rules": [
            {
                "id": "high_cpu",
                "name": "CPU Alto",
                "condition": "cpu_usage > 90",
                "severity": "critical",
                "category": "performance",
                "cooldown_minutes": 5
            }
        ]
    }
    
    async def demo():
        await start_alert_system(config)
        
        # Crear alerta de prueba
        system = get_alert_system()
        if system:
            await system.create_alert(
                "Sistema iniciado",
                "El sistema de alertas AEGIS ha sido iniciado correctamente",
                AlertSeverity.INFO,
                AlertCategory.SYSTEM,
                "alert_system"
            )
        
        # Esperar un poco
        await asyncio.sleep(5)
        
        await stop_alert_system()
    
    asyncio.run(demo())