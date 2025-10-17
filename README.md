# 🤖 IA Distribuida y Colaborativa

**Programador Principal:** Jose Gómez alias KaseMaster  
**Contacto:** kasemaster@protonmail.com
**Versión:** 2.0.0  
**Licencia:** MIT  

## ⚠️ AVISO LEGAL Y ÉTICO

**Este proyecto está diseñado exclusivamente para investigación académica y desarrollo ético de sistemas de inteligencia artificial distribuida. El uso de este código para actividades maliciosas, ilegales o que violen la privacidad está estrictamente prohibido.**

### 🛡️ Principios de Seguridad AEGIS

- **Transparencia**: Todo el código es auditable y documentado
- **Privacidad**: Protección de datos mediante cifrado de extremo a extremo
- **Consenso**: Decisiones distribuidas sin puntos únicos de fallo
- **Responsabilidad**: Trazabilidad de todas las acciones en la red

---

## 🏗️ Arquitectura del Sistema

### Componentes Principales

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TOR Gateway   │◄──►│  P2P Network    │◄──►│ Knowledge Base  │
│                 │    │   Manager       │    │   Distribuida   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Crypto Engine   │    │ Resource Pool   │    │ Consensus Core  │
│                 │    │   Manager       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Características de Seguridad

- **🔐 Cifrado de Extremo a Extremo**: ChaCha20-Poly1305 + Double Ratchet
- **🌐 Comunicaciones Anónimas**: Integración completa con red TOR
- **🤝 Consenso Bizantino**: Tolerancia a fallos con PBFT + Proof of Computation
- **🔑 Identidades Criptográficas**: Ed25519 para firmas digitales
- **🛡️ Resistencia a Ataques**: Protección contra Sybil, Eclipse y envenenamiento

---

## 🚀 Instalación y Configuración

### Prerrequisitos

1. **Python 3.9+**
2. **TOR Browser o Daemon** (para comunicaciones anónimas)
3. **4GB+ RAM** (para operaciones de ML)
4. **Conexión a Internet estable**

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/ia-distribuida.git
cd ia-distribuida

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar TOR (Ubuntu/Debian)
sudo apt-get install tor
sudo systemctl start tor
sudo systemctl enable tor
```

### Configuración de TOR

```bash
# Editar configuración de TOR
sudo nano /etc/tor/torrc

# Añadir las siguientes líneas:
ControlPort 9051
CookieAuthentication 1
CookieAuthFileGroupReadable 1
```

### Variables de Entorno

```bash
# Crear archivo .env
cat > .env << EOF
# Configuración de Red
TOR_CONTROL_PORT=9051
TOR_SOCKS_PORT=9050
P2P_PORT=8080

# Configuración de Seguridad
SECURITY_LEVEL=HIGH  # STANDARD, HIGH, PARANOID
MIN_COMPUTATION_SCORE=50.0
BYZANTINE_THRESHOLD_RATIO=0.33

# Configuración de Consenso
POC_INTERVAL=300  # segundos entre desafíos
PBFT_TIMEOUT=30   # timeout para consenso PBFT

# Logging
LOG_LEVEL=INFO
LOG_FILE=distributed_ai.log
EOF
```

---

## 🔧 Uso del Sistema

### Inicialización de Nodo

```python
import asyncio
from tor_integration import create_secure_tor_gateway, SecurityLevel
from consensus_protocol import HybridConsensus
from cryptography.hazmat.primitives.asymmetric import ed25519

async def initialize_node():
    # Generar identidad criptográfica
    private_key = ed25519.Ed25519PrivateKey.generate()
    node_id = secrets.token_hex(16)
    
    # Inicializar TOR Gateway
    tor_gateway = await create_secure_tor_gateway(SecurityLevel.HIGH)
    
    # Crear servicio onion
    onion_address = await tor_gateway.create_onion_service(8080)
    print(f"Nodo disponible en: {onion_address}")
    
    # Inicializar consenso
    consensus = HybridConsensus(node_id, private_key)
    
    # Unirse a la red (descubrir otros nodos)
    await discover_and_connect_peers(consensus, tor_gateway)
    
    return tor_gateway, consensus

# Ejecutar
asyncio.run(initialize_node())
```

### Contribuir Conocimiento

```python
async def contribute_knowledge(consensus, knowledge_data):
    """Contribuye conocimiento a la red distribuida"""
    
    # Validar y procesar datos
    processed_data = await process_knowledge(knowledge_data)
    
    # Crear propuesta de cambio
    change_proposal = {
        "type": "knowledge_update",
        "content_hash": hashlib.sha256(processed_data).hexdigest(),
        "source_node": consensus.node_id,
        "timestamp": time.time(),
        "data": processed_data
    }
    
    # Proponer cambio a la red
    success = await consensus.pbft.propose_change(change_proposal)
    
    if success:
        print("Conocimiento propuesto exitosamente")
    else:
        print("Error proponiendo conocimiento")
```

### Consultar Red

```python
async def query_network(consensus, query):
    """Consulta la base de conocimiento distribuida"""
    
    # Crear consulta distribuida
    query_message = {
        "type": "knowledge_query",
        "query": query,
        "requester": consensus.node_id,
        "timestamp": time.time()
    }
    
    # Enviar a nodos relevantes
    responses = await broadcast_query(query_message)
    
    # Agregar y validar respuestas
    validated_responses = []
    for response in responses:
        if await validate_response(response):
            validated_responses.append(response)
    
    return aggregate_responses(validated_responses)
```

---

## 📁 Repository Structure

- config/ — JSON configuration and templates (app_config.json, torrc, project/task configs)
- scripts/ — helper scripts for starting/stopping Archon, Tor utilities, and generated command scripts
- reports/ — generated reports and analysis outputs (integration_report.json, task_security_analysis.json)
- docs/ — project documentation
- tests/ — integration tests

---

## 🔒 Consideraciones de Seguridad

### Amenazas Mitigadas

1. **Ataques de Sybil**
   - Proof of Computation para validar identidades
   - Sistema de reputación basado en contribuciones

2. **Ataques de Eclipse**
   - Diversificación geográfica de conexiones TOR
   - Rotación automática de circuitos

3. **Envenenamiento de Datos**
   - Consenso bizantino para validación
   - Firmas criptográficas en todas las contribuciones

4. **Análisis de Tráfico**
   - Comunicaciones exclusivamente a través de TOR
   - Padding temporal y ruido sintético

### Mejores Prácticas

- **Nunca** ejecutar como usuario root
- **Siempre** validar certificados TOR
- **Rotar** claves regularmente (cada 24h)
- **Monitorear** logs de seguridad
- **Actualizar** dependencias frecuentemente

---

## 📊 Monitoreo y Métricas

### Métricas de Red

```python
# Obtener estadísticas de la red
stats = consensus.get_network_stats()
print(f"Nodos activos: {stats['active_nodes']}")
print(f"Umbral bizantino: {stats['byzantine_threshold']}")
print(f"Puntaje promedio: {stats['avg_computation_score']:.2f}")
```

### Métricas de TOR

```python
# Estado de la red TOR
tor_status = await tor_gateway.get_network_status()
print(f"Circuitos activos: {tor_status['active_circuits']}")
print(f"Nodos disponibles: {tor_status['available_nodes']}")
```

### Logs de Seguridad

```bash
# Monitorear logs en tiempo real
tail -f distributed_ai.log | grep -E "(WARNING|ERROR|SECURITY)"

# Analizar patrones de ataque
grep "SECURITY" distributed_ai.log | awk '{print $1, $2, $NF}' | sort | uniq -c
```

---

## 🧪 Testing y Validación

### Tests de Seguridad

```bash
# Ejecutar suite completa de tests
python -m pytest tests/ -v --cov=.

# Tests específicos de seguridad
python -m pytest tests/test_security.py -v

# Tests de consenso
python -m pytest tests/test_consensus.py -v

# Tests de TOR
python -m pytest tests/test_tor_integration.py -v
```

### Simulación de Ataques

```bash
# Simular ataque Sybil
python tests/simulate_sybil_attack.py --nodes 100 --malicious 30

# Simular ataque Eclipse
python tests/simulate_eclipse_attack.py --target node_123

# Test de resistencia bizantina
python tests/test_byzantine_resistance.py --byzantine_ratio 0.25
```

---

## 🤝 Contribuciones

### Código de Conducta

- **Uso Ético**: Solo para investigación y desarrollo legítimo
- **Transparencia**: Documentar todos los cambios de seguridad
- **Responsabilidad**: Reportar vulnerabilidades de forma responsable
- **Colaboración**: Respetar la diversidad y inclusión

### Proceso de Contribución

1. **Fork** del repositorio
2. **Crear** rama para la característica (`git checkout -b feature/nueva-caracteristica`)
3. **Implementar** con tests de seguridad
4. **Documentar** cambios y consideraciones de seguridad
5. **Enviar** Pull Request con descripción detallada

### Reporte de Vulnerabilidades

**NO** reportar vulnerabilidades públicamente. Usar:
- Email: security@proyecto-ia-distribuida.org
- PGP Key: [Clave PGP para comunicación segura]

---

## 📚 Documentación Adicional

- [Guía de Arquitectura Detallada](docs/architecture.md)
- [Manual de Seguridad](docs/security_manual.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)

---

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia MIT con Cláusulas de Uso Ético**.

### Restricciones Adicionales

- **Prohibido** el uso para actividades ilegales
- **Prohibido** el uso para vigilancia no autorizada
- **Prohibido** el uso para manipulación de información
- **Requerido** el cumplimiento de leyes locales de privacidad

---

## 🙏 Reconocimientos

- **TOR Project** por la infraestructura de anonimato
- **Cryptography.io** por las primitivas criptográficas
- **Comunidad de Seguridad** por las mejores prácticas
- **Investigadores en IA Distribuida** por los fundamentos teóricos

---

**⚠️ RECORDATORIO FINAL: Este software es una herramienta de investigación. El usuario es completamente responsable de su uso ético y legal. Los desarrolladores no se hacen responsables del mal uso de este código.**

---

*Desarrollado por AEGIS - Analista Experto en Gestión de Información y Seguridad*  
*Versión 1.0 - Para uso ético únicamente*
<p align="center">
  <a href="https://github.com/KaseMaster/Open-A.G.I/actions/workflows/ci.yml">
    <img src="https://github.com/KaseMaster/Open-A.G.I/actions/workflows/ci.yml/badge.svg" alt="CI Status" />
  </a>
</p>
