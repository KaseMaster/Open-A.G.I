# AEGIS-Storage (Fase 1: AEGIS-Storage-ID)

Módulo DApp del ecosistema AEGIS para gestión de identidades extendidas, roles y permisos de almacenamiento.

## Objetivo (Fase 1)

- Extender la identidad criptográfica AEGIS (Ed25519) con roles de almacenamiento.
- Permitir políticas de multifirma (m-of-n) para operaciones sensibles (atributos/permisos).
- Emitir “transacciones” firmadas por Ed25519 compatibles con el handshake P2P del core.

## Estructura

- `src/`: cliente Python (gestión de identidad/permisos) e integración con consenso.
- `contracts/`: contrato Solidity equivalente (esqueleto para despliegue EVM).

