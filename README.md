# 🤖 AEGIS Framework - IA Distribuida y Colaborativa

<p align="center">
  <a href="https://github.com/KaseMaster/Open-A.G.I/actions/workflows/ci.yml">
    <img src="https://github.com/KaseMaster/Open-A.G.I/actions/workflows/ci.yml/badge.svg" alt="CI Status" />
  </a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python Version" />
  <img src="https://img.shields.io/badge/node-20%2B-green" alt="Node Version" />
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="License" />
</p>

**Programador Principal:** Jose Gómez alias KaseMaster  
**Contacto:** kasemaster@protonmail.com  
**Versión:** 2.1.0  
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
2. **Node.js 20+** (para DApps)
3. **TOR Browser o Daemon** (para comunicaciones anónimas)
4. **4GB+ RAM** (para operaciones de ML)

### Instalación del Núcleo (Python)

```bash
# Clonar el repositorio
git clone https://github.com/KaseMaster/Open-A.G.I.git
cd Open-A.G.I

# Instalar dependencias del núcleo (Editable mode)
pip install -e .

# Configurar TOR (Ubuntu/Debian)
sudo apt-get install tor
sudo systemctl start tor
```

### Instalación de DApps (Node.js)

```bash
# Instalar dependencias de Smart Contracts (Token)
cd dapps/aegis-token
npm install

# Instalar dependencias de Secure Chat
cd ../secure-chat
npm install

# Instalar dependencias del UI
cd ui
npm install
```

### Variables de Entorno

Crear un archivo `.env` en la raíz:

```bash
# Configuración de Red
TOR_CONTROL_PORT=9051
TOR_SOCKS_PORT=9050
P2P_PORT=8080

# Configuración de Seguridad
SECURITY_LEVEL=HIGH  # STANDARD, HIGH, PARANOID
MIN_COMPUTATION_SCORE=50.0
BYZANTINE_THRESHOLD_RATIO=0.33

# Logging
LOG_LEVEL=INFO
```

---

## 🔧 Uso del Sistema

### Iniciar Nodo Completo

```bash
# Iniciar nodo con configuración por defecto
python main.py start-node

# Iniciar solo dashboard de monitoreo
python main.py start-dashboard --type monitoring
```

### Desarrollo de DApps

```bash
# Ejecutar tests de contratos (Aegis Token)
cd dapps/aegis-token
npx hardhat test

# Iniciar UI de chat seguro
cd dapps/secure-chat/ui
npm run dev
```

---

## 📁 Estructura del Repositorio

- **src/aegis_core/**: Núcleo del framework (Python). Contiene módulos de P2P, Crypto, Consenso, TOR.
- **dapps/**: Aplicaciones Descentralizadas (Smart Contracts + UI).
  - `aegis-token/`: Token de gobernanza.
  - `secure-chat/`: Sistema de mensajería segura.
- **config/**: Archivos de configuración y templates.
- **scripts/**: Scripts de utilidad y despliegue.
- **tests/**: Tests de integración y unitarios (Python).
- **docs/**: Documentación del proyecto.

---

## 🧪 Testing y Validación

### Tests del Núcleo (Python)

```bash
# Ejecutar suite completa
pytest tests/
```

### Tests de DApps (Node.js)

```bash
# Tests de contratos inteligentes
npm test --prefix dapps/aegis-token
npm test --prefix dapps/secure-chat
```

---

## 🤝 Contribuciones

### Código de Conducta

- **Uso Ético**: Solo para investigación y desarrollo legítimo
- **Transparencia**: Documentar todos los cambios de seguridad
- **Responsabilidad**: Reportar vulnerabilidades de forma responsable

### Proceso de Contribución

1. **Fork** del repositorio
2. **Crear** rama (`git checkout -b feature/nueva-caracteristica`)
3. **Implementar** con tests
4. **Enviar** Pull Request

---

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia MIT con Cláusulas de Uso Ético**.

**⚠️ RECORDATORIO FINAL: Este software es una herramienta de investigación. El usuario es completamente responsable de su uso ético y legal.**

---

*Desarrollado por AEGIS - Analista Experto en Gestión de Información y Seguridad*
