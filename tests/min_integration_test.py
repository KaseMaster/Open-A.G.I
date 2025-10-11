#!/usr/bin/env python3
"""
Prueba mínima de integración del framework AEGIS.
- Verifica importación de módulos clave
- Ejecuta main.start_node en modo dry-run
"""

import asyncio
import os
import sys

# Asegurar que el directorio del proyecto esté en PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_health_summary_keys():
    from main import health_summary
    summary = health_summary()
    assert "python" in summary
    assert "modules" in summary
    # Verificar algunas dependencias clave declaradas en main
    for m in ["asyncio", "cryptography", "websockets"]:
        assert m in summary["modules"], f"Módulo {m} no está en el resumen"


def test_start_node_dry_run_executes():
    from main import start_node
    # Debe ejecutarse sin levantar servicios pesados
    asyncio.run(start_node(dry_run=True))