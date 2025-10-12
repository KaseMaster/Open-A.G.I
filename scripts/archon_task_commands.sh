#!/bin/bash
# Comandos para crear todas las tareas del proyecto AEGIS Framework

PROJECT_ID="3be97fb2-933c-4284-9990-eff927ea6a06"

# Tarea 1: Análisis de Seguridad P2P - Vectores de Ataque
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Análisis de Seguridad P2P - Vectores de Ataque", "description": "Análisis exhaustivo de vectores de ataque en arquitecturas P2P distribuidas: Sybil, Eclipse, Data Poisoning, Timing attacks. Implementación de mitigaciones específicas.", "feature": "Seguridad y Criptografía", "task_order": 10, "estimated_hours": 8, "status": "done"}'

# Tarea 2: Framework Criptográfico - Autenticación y Cifrado
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Framework Criptográfico - Autenticación y Cifrado", "description": "Implementación completa del sistema criptográfico: Ed25519 para autenticación, ChaCha20-Poly1305 para cifrado, X25519 para intercambio de claves, Double Ratchet para forward secrecy.", "feature": "Seguridad y Criptografía", "task_order": 10, "estimated_hours": 12, "status": "done"}'

# Tarea 3: Integración TOR - Gateway Anónimo
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Integración TOR - Gateway Anónimo", "description": "Implementación del gateway TOR para comunicaciones anónimas: gestión de circuitos, selección de nodos, rotación automática, resistencia a ataques de correlación.", "feature": "Red TOR y Anonimato", "task_order": 10, "estimated_hours": 10, "status": "done"}'

# Tarea 4: Protocolo de Consenso Bizantino - PBFT Optimizado
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Protocolo de Consenso Bizantino - PBFT Optimizado", "description": "Implementación del algoritmo PBFT con optimizaciones para redes P2P: tolerancia a fallos bizantinos, validación de transacciones, sincronización de estado distribuido.", "feature": "Consenso Distribuido", "task_order": 10, "estimated_hours": 15, "status": "done"}'

# Tarea 5: Sistema de Asignación Dinámica de Recursos
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Sistema de Asignación Dinámica de Recursos", "description": "Gestor inteligente de recursos computacionales: balanceador de carga adaptativo, monitoreo en tiempo real, asignación basada en capacidades y demanda.", "feature": "Gestión de Recursos", "task_order": 10, "estimated_hours": 15, "status": "done"}'

# Tarea 6: Mecanismos de Tolerancia a Fallos y Recuperación de Nodos
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Mecanismos de Tolerancia a Fallos y Recuperación de Nodos", "description": "Sistema robusto de detección y recuperación de fallos: heartbeat distribuido, detección de nodos caídos, replicación de datos, recuperación automática, balanceador de carga resiliente.", "feature": "Tolerancia a Fallos", "task_order": 9, "estimated_hours": 18, "status": "done"}'

# Tarea 7: Optimización de Algoritmos de Consenso
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Optimización de Algoritmos de Consenso", "description": "Mejoras de rendimiento en PBFT: reducción de latencia, optimización de mensajes, paralelización de validaciones, técnicas de batching.", "feature": "Consenso Distribuido", "task_order": 8, "estimated_hours": 12, "status": "done"}'

# Tarea 8: Sistema de Métricas y Monitoreo Distribuido
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Sistema de Métricas y Monitoreo Distribuido", "description": "Plataforma de observabilidad: métricas de rendimiento, trazabilidad distribuida, alertas inteligentes, dashboards en tiempo real.", "feature": "Monitoreo y Métricas", "task_order": 7, "estimated_hours": 14, "status": "done"}'

# Tarea 9: Pruebas de Carga y Estrés Distribuidas
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Pruebas de Carga y Estrés Distribuidas", "description": "Suite de testing: simulación de redes P2P masivas, pruebas de resistencia a ataques, benchmarks de rendimiento, análisis de escalabilidad.", "feature": "Testing y Validación", "task_order": 6, "estimated_hours": 16, "status": "done"}'

# Tarea 10: Documentación Técnica y Arquitectural
curl -X POST "http://localhost:8181/api/tasks" -H "Content-Type: application/json" -d '{"project_id": "3be97fb2-933c-4284-9990-eff927ea6a06", "title": "Documentación Técnica y Arquitectural", "description": "Documentación completa: especificaciones de protocolos, guías de implementación, diagramas de arquitectura, manuales de operación.", "feature": "Testing y Validación", "task_order": 5, "estimated_hours": 10, "status": "done"}'
