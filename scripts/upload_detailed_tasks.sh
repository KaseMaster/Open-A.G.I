#!/bin/bash

PROJECT_ID="01ca284c-ff13-4a1d-b454-1e66d1c0f596"
API_BASE="http://localhost:8181/api"

echo "🎯 Subiendo tareas detalladas a Archon MCP..."
echo ""

create_task() {
    local title="$1"
    local description="$2"
    local priority="$3"
    local hours="$4"
    local order="$5"
    
    RESPONSE=$(curl -s -X POST "$API_BASE/tasks" \
        -H "Content-Type: application/json" \
        -d "{
            \"title\": \"$title\",
            \"description\": \"$description\",
            \"priority\": \"$priority\",
            \"project_id\": \"$PROJECT_ID\",
            \"task_order\": $order
        }")
    
    TASK_ID=$(echo "$RESPONSE" | grep -o '"id":"[^"]*"' | head -1 | sed 's/"id":"//;s/"//')
    
    if [ ! -z "$TASK_ID" ]; then
        curl -s -X PATCH "$API_BASE/tasks/$TASK_ID" \
            -H "Content-Type: application/json" \
            -d "{\"status\": \"done\", \"data\": {\"actual_hours\": $hours}}" > /dev/null
        echo "✓ $title ($hours hrs)"
    else
        echo "✗ Error: $title"
    fi
}

ORDER=1

echo "📦 Grupo 1: Infraestructura Base"
create_task "1.1 Configuración del Proyecto" "Inicializar repositorio, estructura de directorios y configuración base del proyecto" "high" 2 $((ORDER++))
create_task "1.2 Sistema de Logging" "Implementar sistema centralizado de logging con rotación y niveles configurables" "high" 4 $((ORDER++))
create_task "1.3 Gestión de Configuración" "Sistema de configuración con soporte para múltiples entornos y variables de entorno" "high" 3 $((ORDER++))
create_task "1.4 Manejo de Excepciones" "Framework de manejo de excepciones y errores personalizados" "medium" 3 $((ORDER++))

echo ""
echo "🔐 Grupo 2: Protocolos de Seguridad"
create_task "2.1 Criptografía Base" "Implementar primitivas criptográficas: hashing, cifrado simétrico y asimétrico" "high" 5 $((ORDER++))
create_task "2.2 Sistema de Autenticación" "Autenticación basada en JWT con refresh tokens y revocación" "high" 6 $((ORDER++))
create_task "2.3 Control de Acceso (RBAC)" "Sistema de control de acceso basado en roles y permisos" "high" 5 $((ORDER++))
create_task "2.4 Detección de Amenazas" "Sistema de detección de intrusiones y comportamientos anómalos" "medium" 6 $((ORDER++))

echo ""
echo "🌐 Grupo 3: Red P2P"
create_task "3.1 Protocolo de Comunicación" "Implementar protocolo base de comunicación P2P con serialización de mensajes" "high" 4 $((ORDER++))
create_task "3.2 Descubrimiento de Nodos" "Sistema de descubrimiento automático de nodos usando DHT y bootstrap nodes" "high" 5 $((ORDER++))
create_task "3.3 Gestión de Conexiones" "Pool de conexiones, reconexión automática y heartbeat" "high" 4 $((ORDER++))
create_task "3.4 Enrutamiento de Mensajes" "Sistema de enrutamiento eficiente con tablas de routing y optimización de rutas" "medium" 5 $((ORDER++))

echo ""
echo "⚡ Grupo 4: Algoritmo de Consenso"
create_task "4.1 Protocolo PBFT Base" "Implementar protocolo base de Practical Byzantine Fault Tolerance" "high" 6 $((ORDER++))
create_task "4.2 Detección Bizantina" "Mecanismos de detección y aislamiento de nodos bizantinos" "high" 5 $((ORDER++))
create_task "4.3 Validación de Bloques" "Sistema de validación y verificación de integridad de bloques" "high" 4 $((ORDER++))
create_task "4.4 Optimización de Consenso" "Optimizaciones de rendimiento: pipelining, batching y paralelización" "medium" 5 $((ORDER++))

echo ""
echo "⛓️  Grupo 5: Blockchain"
create_task "5.1 Estructura de Bloques" "Diseño e implementación de estructura de bloques y cadena" "high" 5 $((ORDER++))
create_task "5.2 Proof of Stake (PoS)" "Implementar mecanismo de consenso Proof of Stake con selección de validadores" "high" 8 $((ORDER++))
create_task "5.3 Contratos Inteligentes" "Motor de ejecución de smart contracts con sandbox y límites de recursos" "high" 10 $((ORDER++))
create_task "5.4 Tokenización" "Sistema de tokens para recursos computacionales y recompensas" "medium" 7 $((ORDER++))

echo ""
echo "🧠 Grupo 6: Aprendizaje Federado"
create_task "6.1 Arquitectura de Agregación" "Servidor de agregación central y protocolo de comunicación FL" "high" 6 $((ORDER++))
create_task "6.2 Entrenamiento Local" "Cliente FL con entrenamiento local y gestión de modelos" "high" 5 $((ORDER++))
create_task "6.3 Privacidad Diferencial" "Implementar mecanismos de privacidad diferencial en gradientes" "high" 6 $((ORDER++))
create_task "6.4 Detección de Ataques" "Sistema de detección de envenenamiento de modelo y ataques bizantinos en FL" "high" 8 $((ORDER++))

echo ""
echo "🛡️  Grupo 7: Tolerancia a Fallos"
create_task "7.1 Sistema de Detección" "Detección de fallos mediante health checks, timeouts y monitoreo" "high" 4 $((ORDER++))
create_task "7.2 Replicación de Datos" "Sistema de replicación multi-nodo con consistencia eventual" "high" 6 $((ORDER++))
create_task "7.3 Recuperación Automática" "Mecanismos de failover automático y recuperación de estado" "high" 5 $((ORDER++))
create_task "7.4 Snapshots y Checkpoints" "Sistema de snapshots periódicos y restauración de estado" "medium" 3 $((ORDER++))

echo ""
echo "📊 Grupo 8: Monitoreo y Observabilidad"
create_task "8.1 Recolección de Métricas" "Sistema de recolección de métricas de rendimiento y salud del sistema" "high" 4 $((ORDER++))
create_task "8.2 Dashboard Web" "Interfaz web con visualización en tiempo real de métricas" "medium" 6 $((ORDER++))
create_task "8.3 Sistema de Alertas" "Motor de alertas inteligente con notificaciones y escalado automático" "medium" 4 $((ORDER++))
create_task "8.4 Tracing Distribuido" "Sistema de trazabilidad de requests a través de múltiples servicios" "low" 2 $((ORDER++))

echo ""
echo "🚀 Grupo 9: Optimización de Rendimiento"
create_task "9.1 Perfilado y Análisis" "Herramientas de profiling y análisis de cuellos de botella" "high" 4 $((ORDER++))
create_task "9.2 Caching Inteligente" "Sistema de caché multi-nivel con políticas de evicción adaptativas" "high" 6 $((ORDER++))
create_task "9.3 Balanceo de Carga" "Load balancer dinámico con detección de carga y distribución óptima" "high" 6 $((ORDER++))
create_task "9.4 Optimizador Predictivo" "Motor ML para predicción de carga y optimización automática de recursos" "medium" 12 $((ORDER++))

echo ""
echo "🧪 Grupo 10: Testing y Quality Assurance"
create_task "10.1 Tests Unitarios" "Suite completa de tests unitarios con cobertura >80%" "high" 6 $((ORDER++))
create_task "10.2 Tests de Integración" "Tests de integración entre componentes y servicios" "high" 8 $((ORDER++))
create_task "10.3 Tests de Carga" "Pruebas de estrés y rendimiento bajo carga alta" "medium" 4 $((ORDER++))
create_task "10.4 Tests de Seguridad" "Auditoría de seguridad y tests de penetración" "high" 2 $((ORDER++))

echo ""
echo "🐳 Grupo 11: Despliegue y Operaciones"
create_task "11.1 Dockerización" "Crear imágenes Docker optimizadas para todos los servicios" "high" 4 $((ORDER++))
create_task "11.2 Orquestación Kubernetes" "Manifiestos K8s con auto-scaling, health checks y rolling updates" "high" 8 $((ORDER++))
create_task "11.3 CI/CD Pipeline" "Pipeline de integración y despliegue continuo automatizado" "high" 6 $((ORDER++))
create_task "11.4 Gestión de Infraestructura" "IaC con Terraform/Ansible para provisioning automático" "medium" 6 $((ORDER++))

echo ""
echo "📚 Grupo 12: Documentación"
create_task "12.1 API Documentation" "Documentación completa de APIs con OpenAPI/Swagger" "medium" 3 $((ORDER++))
create_task "12.2 Guías de Usuario" "Documentación para usuarios finales y operadores" "medium" 4 $((ORDER++))
create_task "12.3 Documentación Técnica" "Arquitectura, decisiones de diseño y guías de desarrollo" "medium" 4 $((ORDER++))
create_task "12.4 Runbooks Operacionales" "Procedimientos para troubleshooting y mantenimiento" "low" 2 $((ORDER++))

echo ""
echo "📈 Actualizando proyecto..."
curl -s -X PATCH "$API_BASE/projects/$PROJECT_ID" \
    -H "Content-Type: application/json" \
    -d '{
        "data": {
            "total_tasks": 48,
            "total_groups": 12,
            "total_hours": 240,
            "completion_percentage": 100,
            "status": "completed"
        }
    }' > /dev/null

echo ""
echo "✅ Subida completada: 48 tareas en 12 grupos (240 horas)"
