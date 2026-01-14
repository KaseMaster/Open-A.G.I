# Remediación de Alertas Dependabot / Security Alerts

Fecha: 2026-01-14

## Alcance

Se replicaron alertas tipo Dependabot usando auditorías locales:

- Python: `pip-audit` sobre manifests del repo.
- Node.js: `npm audit` sobre cada `package.json` del proyecto.

Los reportes JSON de auditoría quedan en `reports/`.

## Resultados (Python)

- `pip-audit` no reportó vulnerabilidades en:
  - `requirements.txt`
  - `requirements-test.txt`
  - `requirements-dev.txt`
  - `docker/requirements-docker.txt`
  - `market-pulse-agi/backend/requirements.txt`

### Correcciones aplicadas

- Se corrigió `requirements-dev.txt` (contenía bytes nulos y era inválido para herramientas automáticas).  
  Referencia: [requirements-dev.txt](file:///g:/Open%20A.G.I/requirements-dev.txt)
- Se limpió `requirements.txt` eliminando dependencias de desarrollo y una duplicidad de `torch`.  
  Referencia: [requirements.txt](file:///g:/Open%20A.G.I/requirements.txt)
- Se eliminó `safety` de `requirements-test.txt` (se usa auditoría moderna con `pip-audit`).  
  Referencia: [requirements-test.txt](file:///g:/Open%20A.G.I/requirements-test.txt)
- Se eliminó la duplicidad de `httpx` en `market-pulse-agi/backend/requirements.txt`.  
  Referencia: [requirements.txt](file:///g:/Open%20A.G.I/market-pulse-agi/backend/requirements.txt)

## Resultados (Node.js)

### Paquetes sin vulnerabilidades (audit-level=low)

- `dashboard` (tras migración)
- `market-pulse-agi/frontend`
- `dapps/secure-chat/ui`

### Paquetes con vulnerabilidades residuales (solo LOW, sin fix disponible)

- `dapps/aegis-token`
- `dapps/secure-chat`

Las vulnerabilidades residuales provienen del toolchain de Hardhat (transitivas) y aparecen con “No fix available” en `npm audit`:

- `cookie <0.7.0` vía `@sentry/node` (dependencia transitiva de Hardhat)
- `elliptic` vía dependencias antiguas de `@ethersproject/*`
- `tmp <=0.2.3` vía `solc`

Mitigación aplicada: se corrigieron vulnerabilidades HIGH con `glob` usando overrides, dejando únicamente LOW no remediables sin cambiar de toolchain.

## Cambios aplicados (Node.js)

### 1) Eliminación de `ipfs-http-client` (vulnerable) y reemplazo por `fetch`

- Se eliminó `ipfs-http-client` y se implementó acceso a IPFS vía HTTP API usando `fetch`:
  - [App.jsx](file:///g:/Open%20A.G.I/dapps/secure-chat/ui/src/App.jsx)
  - [ipfs.js](file:///g:/Open%20A.G.I/dapps/secure-chat/ui/src/ipfs.js)
- Se actualizó Vite a rama segura (incluye fix de esbuild) y se añadió override para `js-yaml`:
  - [package.json](file:///g:/Open%20A.G.I/dapps/secure-chat/ui/package.json)

### 2) Remediación de HIGH en dapps (Hardhat)

- Se añadieron `overrides` para forzar versiones seguras en transitivas:
  - [aegis-token package.json](file:///g:/Open%20A.G.I/dapps/aegis-token/package.json)
  - [secure-chat package.json](file:///g:/Open%20A.G.I/dapps/secure-chat/package.json)

### 3) Migración de `dashboard` de CRA a Vite (para eliminar `react-scripts`)

- Se reemplazó `react-scripts` (vulnerabilidades sin remediación en CRA) por Vite moderno:
  - [dashboard package.json](file:///g:/Open%20A.G.I/dashboard/package.json)
  - [index.html](file:///g:/Open%20A.G.I/dashboard/index.html)
  - [vite.config.js](file:///g:/Open%20A.G.I/dashboard/vite.config.js)

## Verificación

- `pip-audit` ejecutado y reportes guardados en `reports/`.
- `npm audit --audit-level=low`:
  - 0 vulnerabilidades en `dashboard`, `market-pulse-agi/frontend`, `dapps/secure-chat/ui`.
  - Solo LOW sin fix disponible en `dapps/aegis-token` y `dapps/secure-chat`.
- Pruebas y builds relevantes:
  - `pytest` (subset): token rules, ledger API y consensus (OK).
  - `npm run build`: dashboard y secure-chat UI (OK).

## Recomendación para cerrar LOW residuales (Hardhat)

Para eliminar también las LOW sin fix disponible, la opción realista es migrar el toolchain (por ejemplo, a Foundry) o a un stack Hardhat que no arrastre esas dependencias transitivas.
