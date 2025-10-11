# Política de Seguridad

Gracias por ayudar a mantener la seguridad de Open-A.G.I.

## Versiones soportadas

- Rama `main`: mantenida de forma continua.
- Versiones etiquetadas: se recomienda usar la última release estable.

## Reporte de vulnerabilidades

- Usa GitHub Security Advisories para reportar de forma privada.
- Alternativamente, abre un issue marcado como `security` con información limitada y solicita contacto privado.
- No publiques exploits ni POC sin coordinación previa.

## Buenas prácticas internas

- Escaneo con `bandit` y `pip-audit` en CI.
- Imagenes Docker firmadas con Cosign (keyless) y SBOM generado (SPDX).
- Dependabot para dependencias de `pip` y `github-actions`.