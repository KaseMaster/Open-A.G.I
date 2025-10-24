"""
AEGIS Storage Module

Sistema de almacenamiento y persistencia de datos incluyendo respaldos
y base de conocimiento.
"""

from aegis.storage import backup_system
from aegis.storage import knowledge_base

__all__ = [
    "backup_system",
    "knowledge_base",
]
