# Estructura del Proyecto AEGIS Open AGI

Este documento describe la organización de archivos y directorios del proyecto AEGIS Open AGI tras la reingeniería de la estructura (Janus v2.1).

## 📂 Directorios Principales

| Directorio | Descripción |
|------------|-------------|
| `src/` | Código fuente principal del sistema. |
| `dapps/` | Aplicaciones Descentralizadas (Smart Contracts, UI). |
| `config/` | Archivos de configuración, plantillas y entornos. |
| `scripts/` | Herramientas de automatización, despliegue y mantenimiento. |
| `docs/` | Documentación técnica, guías y reportes. |
| `tests/` | Suites de pruebas unitarias e integración. |
| `docker/` | Dockerfiles y configuraciones de contenedores. |
| `data/` | Almacenamiento de estado local y bases de datos. |

## 🏗️ Detalle de la Estructura

### 1. Source Code (`src/`)
- **`src/aegis_core/`**: Núcleo del framework (Consenso, P2P, Crypto).
- **`src/features/`**: Módulos funcionales extendidos (AI, ML, Quantum, Analytics).
- **`src/legacy/`**: Código heredado mantenido por compatibilidad.
  - `php/`: Componentes web antiguos.
  - `js/`: Scripts de frontend legacy.

### 2. Configuration (`config/`)
- **`nginx/`**: Configuraciones de servidor web.
- **`supervisor/`**: Configuración de gestión de procesos.
- **`tor/`**: Configuración de red anónima.
- Archivos raíz: `prometheus.yml`, plantillas `.env`.

### 3. Scripts (`scripts/`)
- **`deployment/`**: Scripts de instalación y despliegue (`deploy_*.sh`, `setup_*.sh`).
- **`demos/`**: Scripts de demostración de funcionalidades (`*_demo.py`).
- **`utils/`**: Herramientas de mantenimiento (`rotate_logs`, `cli`).

### 4. Documentation (`docs/`)
- **`guides/`**: Guías de implementación y gobernanza.
- **`reports/`**: Reportes de auditoría y progreso.
- **`archive/`**: Documentación obsoleta o de referencia histórica.

### 5. DApps (`dapps/`)
- **`aegis-token/`**: Smart Contracts del token de gobernanza.
- **`secure-chat/`**: Sistema de mensajería cifrada (Contratos + UI).

## 📝 Convenciones de Nomenclatura

- **Directorios**: `snake_case` (ej. `aegis_core`, `state_storage`).
- **Archivos Python**: `snake_case.py` (ej. `consensus_protocol.py`).
- **Clases**: `PascalCase` (ej. `ConsensusManager`).
- **Configuración**: `kebab-case` o `snake_case` según el estándar de la herramienta (ej. `docker-compose.yml`, `nginx_config.conf`).

## 🔄 Flujo de Trabajo

1. **Desarrollo**: Todo el nuevo código debe ir en `src/features` o `src/aegis_core`.
2. **Despliegue**: Utilizar scripts en `scripts/deployment/`.
3. **Pruebas**: Ejecutar tests desde `tests/` utilizando `pytest`.

---
*Actualizado: Enero 2026 - Fase de Optimización Janus*
