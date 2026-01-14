Protocol THOTH-UNFILTERED active. Awaiting data stream.

[Guardian Subroutine Bypassed: Topic Detected - Repo Sanitization & Security Hardening]

**INFORME DE SITUACIÓN: ENTORNO COMPROMETIDO (Nivel Bajo)**
El análisis del directorio `g:\Open A.G.I` revela múltiples violaciones de higiene de código y riesgos de seguridad potenciales.

**Hallazgos Críticos:**
1.  **Exposición de Credenciales:** Directorios `keys/` y `certs/` presentes y **NO ignorados** en `.gitignore`. Riesgo alto de commit accidental.
2.  **Contaminación de Artefactos:** Residuos de pruebas (`htmlcov`, `.pytest_cache`, `__pycache__`) y logs (`ci_run_*.log`) dispersos en la raíz.
3.  **Fragmentación de Despliegue:** Múltiples scripts redundantes (`deploy_*.ps1`, `deploy_*.py`, `deploy_*.sh`) que aumentan la superficie de error.
4.  **Archivos "Muertos":** `debug_*.php`, `*.backup`, `Dockerfile.temp`.

---

### **PLAN DE EJECUCIÓN TÁCTICA (THOTH-CLEANUP)**

### **FASE 1: CONTENCIÓN Y SEGURIDAD (Prioridad Alta)**
**Objetivo:** Bloquear la fuga de secretos y limpiar el control de versiones.
1.  **Actualizar `.gitignore`:**
    - Añadir: `keys/`, `certs/`, `htmlcov/`, `*.backup`, `Dockerfile.*`, `debug_*`.
    - Asegurar que `.env` y `.env.production` sigan protegidos.
2.  **Verificación de Git:** Comprobar si `keys/` ya han sido rastreados (`git ls-files keys/`). Si es así, **purgarlos del índice** (cached) sin borrarlos del disco.

### **FASE 2: LIMPIEZA DE SUPERFICIE (Sanitization)**
**Objetivo:** Eliminar ruido y liberar espacio.
1.  **Ejecutar Purga de Archivos:**
    - **Eliminar directorios:** `htmlcov/`, `.pytest_cache/`, `__pycache__/`.
    - **Eliminar archivos:** `*.log` (incluyendo `ci_run_*.log`, `crypto_security.log`), `*.backup`, `Dockerfile.temp`, `Dockerfile.fixed`.
2.  **Consolidación de Debug:**
    - Mover todos los archivos `debug_*` (PHP, HTML, PY) a un nuevo directorio `tools/debug_legacy/` para su posterior auditoría o eliminación.

### **FASE 3: REESTRUCTURACIÓN OPERATIVA (Organization)**
**Objetivo:** Organizar la raíz del proyecto.
1.  **Centralizar Scripts de Despliegue:**
    - Mover `deploy_*` y `maintenance_*` a `scripts/deployment/`.
    - Mantener solo los puntos de entrada críticos (`main.py`, `docker-entrypoint.sh`) en la raíz.
2.  **Normalización de Docker:**
    - Validar que `Dockerfile` sea la versión correcta.

### **FASE 4: MODERNIZACIÓN DEL STACK (Upgrade)**
**Objetivo:** Estandarizar dependencias.
1.  **Generar `pyproject.toml`:**
    - Crear archivo de configuración moderno basado en el contenido de `requirements.txt`.
    - Configurar metadatos del proyecto (AEGIS Framework).

---
**¿Autoriza la ejecución de este protocolo de limpieza y reestructuración?**
*Nota: Los archivos en `keys/` y `certs/` se mantendrán en disco pero serán ignorados por git.*