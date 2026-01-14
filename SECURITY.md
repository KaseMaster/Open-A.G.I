# Política de Seguridad

Gracias por ayudar a mantener la seguridad de Open-A.G.I.

## Versiones soportadas

- Rama `main`: mantenida de forma continua.
- Versiones etiquetadas: se recomienda usar la última release estable.

## Reporte de vulnerabilidades

Please report vulnerabilities via email to `kasemaster@protonmail.com`
- No publiques exploits ni POC sin coordinación previa.

## Buenas prácticas internas

- Escaneo con `bandit` y `pip-audit` en CI.
- Imagenes Docker firmadas con Cosign (keyless) y SBOM generado (SPDX).
- Dependabot para dependencias de `pip` y `github-actions`.