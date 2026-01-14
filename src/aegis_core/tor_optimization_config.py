#!/usr/bin/env python3
"""
Configuración optimizada para TOR Integration
Mejoras de rendimiento, seguridad y estabilidad
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TorOptimizationLevel(Enum):
    """Niveles de optimización TOR"""
    PERFORMANCE = "performance"  # Máximo rendimiento
    BALANCED = "balanced"       # Balance rendimiento/seguridad
    SECURITY = "security"       # Máxima seguridad

@dataclass
class TorOptimizationConfig:
    """Configuración optimizada para TOR"""
    
    # Configuración básica
    optimization_level: TorOptimizationLevel = TorOptimizationLevel.BALANCED
    control_port: int = 9051
    socks_port: int = 9050
    
    # Optimizaciones de circuito
    circuit_build_timeout: int = 60  # segundos
    circuit_idle_timeout: int = 300  # 5 minutos
    max_circuits_pending: int = 32
    max_client_circuits_pending: int = 16
    
    # Optimizaciones de conexión
    connection_padding: bool = True
    reduced_connection_padding: bool = False
    dormant_client_timeout: int = 1800  # 30 minutos
    
    # Configuración de nodos
    use_entry_guards: bool = True
    num_entry_guards: int = 3
    guard_lifetime: int = 86400 * 30  # 30 días
    
    # Configuración de diversidad geográfica
    enforce_distinct_subnets: bool = True
    client_use_ipv6: bool = True
    prefer_ipv6: bool = False
    
    # Configuración de rendimiento
    bandwidth_rate: Optional[int] = None  # KB/s, None = sin límite
    bandwidth_burst: Optional[int] = None  # KB/s, None = sin límite
    relay_bandwidth_rate: Optional[int] = None
    
    # Configuración de seguridad
    safe_socks: bool = False  # Para compatibilidad con aplicaciones
    test_socks: bool = False
    warn_unsafe_socks: bool = True
    
    # Configuración de logs
    log_level: str = "notice"
    log_to_file: bool = True
    log_file_path: str = "G:\\OpenAGI_logs\\tor_optimized.log"
    
    # Configuración de directorios
    data_directory: str = "G:\\Open A.G.I\\tor_data"
    cache_directory: str = "G:\\Open A.G.I\\tor_cache"
    
    def get_torrc_config(self) -> str:
        """Genera configuración torrc optimizada"""
        
        config_lines = [
            "# Configuración TOR optimizada para AEGIS Framework",
            f"# Nivel de optimización: {self.optimization_level.value}",
            "",
            "# Configuración básica",
            f"SocksPort 127.0.0.1:{self.socks_port} IsolateDestAddr IsolateSOCKSAuth",
            f"ControlPort 127.0.0.1:{self.control_port}",
            "CookieAuthentication 1",
            "HashedControlPassword 16:872860B76453A77D60CA2BB8C1A7042072093276A3D701AD684053EC4C",
            "",
            "# Directorios",
            f'DataDirectory "{self.data_directory}"',
            f'CacheDirectory "{self.cache_directory}"',
            "",
            "# Configuración de cliente",
            "ClientOnly 1",
            "AvoidDiskWrites 1",
            "",
        ]
        
        # Configuración según nivel de optimización
        if self.optimization_level == TorOptimizationLevel.PERFORMANCE:
            config_lines.extend([
                "# Optimizaciones de rendimiento",
                f"CircuitBuildTimeout {self.circuit_build_timeout}",
                f"CircuitIdleTimeout {self.circuit_idle_timeout}",
                f"MaxClientCircuitsPending {self.max_client_circuits_pending}",
                "LearnCircuitBuildTimeout 1",
                "UseEntryGuards 1",
                f"NumEntryGuards {self.num_entry_guards}",
                "ConnectionPadding 0",  # Desactivar para rendimiento
                "ReducedConnectionPadding 1",
                "",
            ])
        
        elif self.optimization_level == TorOptimizationLevel.BALANCED:
            config_lines.extend([
                "# Configuración balanceada",
                f"CircuitBuildTimeout {self.circuit_build_timeout}",
                f"CircuitIdleTimeout {self.circuit_idle_timeout}",
                f"MaxClientCircuitsPending {self.max_client_circuits_pending}",
                "UseEntryGuards 1",
                f"NumEntryGuards {self.num_entry_guards}",
                "ConnectionPadding 1",
                "ReducedConnectionPadding 0",
                "",
            ])
        
        elif self.optimization_level == TorOptimizationLevel.SECURITY:
            config_lines.extend([
                "# Configuración de máxima seguridad",
                f"CircuitBuildTimeout {self.circuit_build_timeout + 30}",  # Más tiempo para seguridad
                f"CircuitIdleTimeout {self.circuit_idle_timeout // 2}",    # Rotación más frecuente
                f"MaxClientCircuitsPending {self.max_client_circuits_pending // 2}",
                "UseEntryGuards 1",
                f"NumEntryGuards {self.num_entry_guards + 2}",  # Más guards
                "ConnectionPadding 1",
                "ReducedConnectionPadding 0",
                "EnforceDistinctSubnets 1",
                "StrictNodes 1",
                "",
            ])
        
        # Configuración de red
        config_lines.extend([
            "# Configuración de red",
            f"EnforceDistinctSubnets {1 if self.enforce_distinct_subnets else 0}",
            f"ClientUseIPv6 {1 if self.client_use_ipv6 else 0}",
            f"ClientPreferIPv6ORPort {1 if self.prefer_ipv6 else 0}",
            "",
        ])
        
        # Configuración de ancho de banda
        if self.bandwidth_rate:
            config_lines.append(f"BandwidthRate {self.bandwidth_rate} KB")
        if self.bandwidth_burst:
            config_lines.append(f"BandwidthBurst {self.bandwidth_burst} KB")
        
        # Configuración de seguridad
        config_lines.extend([
            "",
            "# Configuración de seguridad",
            f"SafeSocks {1 if self.safe_socks else 0}",
            f"TestSocks {1 if self.test_socks else 0}",
            f"WarnUnsafeSocks {1 if self.warn_unsafe_socks else 0}",
            "SocksPolicy accept *",
            "",
        ])
        
        # Configuración de logs
        if self.log_to_file:
            config_lines.extend([
                "# Configuración de logs",
                f"Log {self.log_level} file {self.log_file_path}",
                "",
            ])
        
        # Configuración de servicio onion
        config_lines.extend([
            "# Servicio onion optimizado",
            'HiddenServiceDir "G:\\Open A.G.I\\onion_service"',
            "HiddenServiceVersion 3",
            "HiddenServicePort 80 127.0.0.1:8090",
            "HiddenServiceMaxStreams 10",
            "HiddenServiceMaxStreamsCloseCircuit 1",
            "",
            "# Autenticación de cliente onion",
            'ClientOnionAuthDir "G:\\Open A.G.I\\client_onion_auth"',
            "",
        ])
        
        # Configuración de GeoIP
        config_lines.extend([
            "# Bases de datos GeoIP",
            'GeoIPFile "C:\\ProgramData\\chocolatey\\lib\\tor\\tools\\data\\geoip"',
            'GeoIPv6File "C:\\ProgramData\\chocolatey\\lib\\tor\\tools\\data\\geoip6"',
        ])
        
        return "\n".join(config_lines)
    
    def save_optimized_torrc(self, filepath: str = None) -> str:
        """Guarda la configuración optimizada en un archivo torrc"""
        if filepath is None:
            filepath = "G:\\Open A.G.I\\config\\torrc_optimized"
        
        config_content = self.get_torrc_config()
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"Configuración TOR optimizada guardada en: {filepath}")
        return filepath

class TorPerformanceMonitor:
    """Monitor de rendimiento para TOR"""
    
    def __init__(self):
        self.circuit_build_times: List[float] = []
        self.connection_success_rate: float = 0.0
        self.bandwidth_usage: Dict[str, float] = {}
        
    async def monitor_circuit_performance(self, tor_gateway) -> Dict[str, float]:
        """Monitorea el rendimiento de los circuitos"""
        try:
            if not tor_gateway.controller:
                return {}
            
            # Obtener estadísticas de circuitos
            circuits = tor_gateway.controller.get_circuits()
            
            active_circuits = [c for c in circuits if c.status == 'BUILT']
            failed_circuits = [c for c in circuits if c.status == 'FAILED']
            
            success_rate = len(active_circuits) / max(len(circuits), 1)
            
            return {
                'active_circuits': len(active_circuits),
                'failed_circuits': len(failed_circuits),
                'success_rate': success_rate,
                'total_circuits': len(circuits)
            }
            
        except Exception as e:
            logger.error(f"Error monitoreando rendimiento TOR: {e}")
            return {}
    
    async def get_bandwidth_stats(self, tor_gateway) -> Dict[str, int]:
        """Obtiene estadísticas de ancho de banda"""
        try:
            if not tor_gateway.controller:
                return {}
            
            # Obtener estadísticas de ancho de banda
            info = tor_gateway.controller.get_info(['traffic/read', 'traffic/written'])
            
            return {
                'bytes_read': int(info.get('traffic/read', 0)),
                'bytes_written': int(info.get('traffic/written', 0))
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de ancho de banda: {e}")
            return {}

def create_optimized_config(level: TorOptimizationLevel = TorOptimizationLevel.BALANCED) -> TorOptimizationConfig:
    """Crea una configuración optimizada según el nivel especificado"""
    
    config = TorOptimizationConfig(optimization_level=level)
    
    # Ajustes específicos por nivel
    if level == TorOptimizationLevel.PERFORMANCE:
        config.circuit_build_timeout = 45
        config.circuit_idle_timeout = 600  # 10 minutos
        config.max_client_circuits_pending = 32
        config.connection_padding = False
        config.reduced_connection_padding = True
        
    elif level == TorOptimizationLevel.SECURITY:
        config.circuit_build_timeout = 90
        config.circuit_idle_timeout = 180  # 3 minutos
        config.max_client_circuits_pending = 8
        config.num_entry_guards = 5
        config.safe_socks = True
        config.test_socks = True
    
    return config

if __name__ == "__main__":
    # Crear configuración optimizada balanceada
    config = create_optimized_config(TorOptimizationLevel.BALANCED)
    
    # Guardar configuración
    config_path = config.save_optimized_torrc()
    print(f"Configuración TOR optimizada creada en: {config_path}")
    
    # Mostrar configuración
    print("\nConfiguración generada:")
    print("=" * 50)
    print(config.get_torrc_config())