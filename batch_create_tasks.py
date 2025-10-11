#!/usr/bin/env python3
"""
Script para crear todas las tareas del proyecto AEGIS Framework en Archon MCP
usando la interfaz web directamente.
"""

import json
import time
from pathlib import Path

# ID del proyecto creado
# Actualizado para el proyecto solicitado por el usuario
PROJECT_ID = "3be97fb2-933c-4284-9990-eff927ea6a06"

# Tareas completadas
completed_tasks = [
    {
        "title": "Análisis de Seguridad P2P - Vectores de Ataque",
        "description": "Análisis exhaustivo de vectores de ataque en arquitecturas P2P distribuidas: Sybil, Eclipse, Data Poisoning, Timing attacks. Implementación de mitigaciones específicas.",
        "feature": "Seguridad y Criptografía",
        "task_order": 10,
        "estimated_hours": 8,
        "status": "done"
    },
    {
        "title": "Framework Criptográfico - Autenticación y Cifrado",
        "description": "Implementación completa del sistema criptográfico: Ed25519 para autenticación, ChaCha20-Poly1305 para cifrado, X25519 para intercambio de claves, Double Ratchet para forward secrecy.",
        "feature": "Seguridad y Criptografía",
        "task_order": 10,
        "estimated_hours": 12,
        "status": "done"
    },
    {
        "title": "Integración TOR - Gateway Anónimo",
        "description": "Implementación del gateway TOR para comunicaciones anónimas: gestión de circuitos, selección de nodos, rotación automática, resistencia a ataques de correlación.",
        "feature": "Red TOR y Anonimato",
        "task_order": 10,
        "estimated_hours": 10,
        "status": "done"
    },
    {
        "title": "Protocolo de Consenso Bizantino - PBFT Optimizado",
        "description": "Implementación del algoritmo PBFT con optimizaciones para redes P2P: tolerancia a fallos bizantinos, validación de transacciones, sincronización de estado distribuido.",
        "feature": "Consenso Distribuido",
        "task_order": 10,
        "estimated_hours": 15,
        "status": "done"
    },
    {
        "title": "Sistema de Asignación Dinámica de Recursos",
        "description": "Gestor inteligente de recursos computacionales: balanceador de carga adaptativo, monitoreo en tiempo real, asignación basada en capacidades y demanda.",
        "feature": "Gestión de Recursos",
        "task_order": 10,
        "estimated_hours": 15,
        "status": "done"
    }
]

# Tarea en progreso
current_task = {
    "title": "Mecanismos de Tolerancia a Fallos y Recuperación de Nodos",
    "description": "Sistema robusto de detección y recuperación de fallos: heartbeat distribuido, detección de nodos caídos, replicación de datos, recuperación automática, balanceador de carga resiliente.",
    "feature": "Tolerancia a Fallos",
    "task_order": 9,
    "estimated_hours": 18,
    # Marcar como completada según solicitud
    "status": "done"
}

# Tareas pendientes
pending_tasks = [
    {
        "title": "Optimización de Algoritmos de Consenso",
        "description": "Mejoras de rendimiento en PBFT: reducción de latencia, optimización de mensajes, paralelización de validaciones, técnicas de batching.",
        "feature": "Consenso Distribuido",
        "task_order": 8,
        "estimated_hours": 12,
        # Marcar como completada según solicitud
        "status": "done"
    },
    {
        "title": "Sistema de Métricas y Monitoreo Distribuido",
        "description": "Plataforma de observabilidad: métricas de rendimiento, trazabilidad distribuida, alertas inteligentes, dashboards en tiempo real.",
        "feature": "Monitoreo y Métricas",
        "task_order": 7,
        "estimated_hours": 14,
        # Marcar como completada según solicitud
        "status": "done"
    },
    {
        "title": "Pruebas de Carga y Estrés Distribuidas",
        "description": "Suite de testing: simulación de redes P2P masivas, pruebas de resistencia a ataques, benchmarks de rendimiento, análisis de escalabilidad.",
        "feature": "Testing y Validación",
        "task_order": 6,
        "estimated_hours": 16,
        # Marcar como completada según solicitud
        "status": "done"
    },
    {
        "title": "Documentación Técnica y Arquitectural",
        "description": "Documentación completa: especificaciones de protocolos, guías de implementación, diagramas de arquitectura, manuales de operación.",
        "feature": "Testing y Validación",
        "task_order": 5,
        "estimated_hours": 10,
        # Marcar como completada según solicitud
        "status": "done"
    }
]

def create_task_commands():
    """Genera comandos curl para crear todas las tareas"""
    all_tasks = completed_tasks + [current_task] + pending_tasks
    
    commands = []
    for i, task in enumerate(all_tasks, 1):
        task_with_project = {
            "project_id": PROJECT_ID,
            **task
        }
        
        json_str = json.dumps(task_with_project, ensure_ascii=False)
        command = f'curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d \'{json_str}\''
        commands.append(f"# Tarea {i}: {task['title']}")
        commands.append(command)
        commands.append("")
    
    return commands

def main():
    """Función principal"""
    print("🚀 Generando comandos para crear tareas en Archon MCP...")
    
    commands = create_task_commands()
    
    # Guardar comandos en archivo
    commands_file = Path("archon_task_commands.sh")
    with open(commands_file, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Comandos para crear todas las tareas del proyecto AEGIS Framework\n\n")
        f.write(f"PROJECT_ID=\"{PROJECT_ID}\"\n\n")
        f.write("\n".join(commands))
    
    print(f"✅ Comandos guardados en: {commands_file}")
    print(f"📊 Total de tareas a crear: {len(completed_tasks + [current_task] + pending_tasks)}")
    print(f"   - Completadas: {len(completed_tasks)}")
    print(f"   - En progreso: 1")
    print(f"   - Pendientes: {len(pending_tasks)}")
    
    # Crear resumen del proyecto
    summary = {
        "project_id": PROJECT_ID,
        "project_title": "IA Distribuida y Colaborativa - AEGIS Framework",
        "total_tasks": len(completed_tasks + [current_task] + pending_tasks),
        "completed_tasks": len(completed_tasks),
        "in_progress_tasks": 1,
        "pending_tasks": len(pending_tasks),
        "total_hours": sum(t["estimated_hours"] for t in completed_tasks + [current_task] + pending_tasks),
        "completed_hours": sum(t["estimated_hours"] for t in completed_tasks),
        "features": {
            "Seguridad y Criptografía": 2,
            "Red TOR y Anonimato": 1,
            "Consenso Distribuido": 2,
            "Gestión de Recursos": 1,
            "Tolerancia a Fallos": 1,
            "Monitoreo y Métricas": 1,
            "Testing y Validación": 2
        }
    }
    
    with open("archon_project_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Resumen del proyecto guardado en: archon_project_summary.json")
    
    return commands

if __name__ == "__main__":
    main()