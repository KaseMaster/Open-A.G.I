# Contribuir a Open-A.G.I

¡Gracias por tu interés en contribuir!

## Requisitos previos

- Python 3.11+
- Docker 24+
- Git y GitHub CLI (`gh`) opcional para flujos avanzados

## Flujo de trabajo

1. Crea una rama desde `main`.
2. Implementa cambios y añade tests.
3. Ejecuta localmente:
   - `flake8` (lint)
   - `pytest -q` (tests)
   - `bandit -r .` y `pip-audit` (seguridad)
   - `docker build` (smoke test)
4. Abre un Pull Request usando la plantilla y enlaza el issue.

## CI/CD

- Lint y tests en Windows y Ubuntu.
- Build Docker con caché; SBOM y firma Cosign en `main`/release.
- Publicación multi-arch a GHCR con tags semver en releases.

## Estilo y commits

- Mensajes claros (tipo: contexto breve).
- Evita secretos en código/commits.

## Código de conducta

Consulta `CODE_OF_CONDUCT.md`.