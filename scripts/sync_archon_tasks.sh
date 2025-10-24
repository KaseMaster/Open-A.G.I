#!/bin/bash

PROJECT_ID="01ca284c-ff13-4a1d-b454-1e66d1c0f596"
API_BASE="http://localhost:8181/api"

echo "🔄 Actualizando estado de tareas en Archon MCP..."
echo ""

# Obtener todas las tareas
echo "📥 Obteniendo lista de tareas..."
TASKS_JSON=$(curl -s "$API_BASE/tasks?project_id=$PROJECT_ID&per_page=100")

# Contar tareas
TOTAL_TASKS=$(echo "$TASKS_JSON" | grep -o '"total":[0-9]*' | grep -o '[0-9]*' | head -1)
echo "📊 Total de tareas: $TOTAL_TASKS"
echo ""

# Extraer IDs de las primeras 48 tareas (originales)
echo "🔄 Marcando primeras 48 tareas como completadas..."

# Crear script Python para procesar JSON y actualizar tareas
python3 << 'PYTHON_SCRIPT'
import json
import requests
import sys

API_BASE = "http://localhost:8181/api"
PROJECT_ID = "01ca284c-ff13-4a1d-b454-1e66d1c0f596"

try:
    # Obtener todas las tareas
    response = requests.get(f"{API_BASE}/tasks?project_id={PROJECT_ID}&per_page=100")
    data = response.json()
    
    tasks = data.get('tasks', [])
    
    # Actualizar las primeras 48 tareas como completadas
    updated = 0
    errors = 0
    
    for i, task in enumerate(tasks[:48], 1):
        task_id = task['id']
        title = task['title']
        
        try:
            # Actualizar tarea a estado 'done'
            update_response = requests.patch(
                f"{API_BASE}/tasks/{task_id}",
                json={"status": "done"},
                headers={"Content-Type": "application/json"}
            )
            
            if update_response.status_code in [200, 204]:
                updated += 1
                if i % 10 == 0:
                    print(f"  ✓ {i}/48 tareas actualizadas...")
            else:
                errors += 1
                
        except Exception as e:
            errors += 1
            print(f"  ✗ Error en tarea {i}: {str(e)[:50]}", file=sys.stderr)
    
    print(f"\n✅ Actualización completada:")
    print(f"   - Tareas actualizadas: {updated}")
    print(f"   - Errores: {errors}")
    print(f"   - Total procesadas: {updated + errors}")
    
except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    sys.exit(1)

PYTHON_SCRIPT

echo ""
echo "📊 Verificando estado final..."
sleep 1

# Contar tareas por estado
echo "Estado de tareas:"
curl -s "$API_BASE/tasks?project_id=$PROJECT_ID&per_page=100" | \
    python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tasks = data.get('tasks', [])
    
    todo = sum(1 for t in tasks if t['status'] == 'todo')
    done = sum(1 for t in tasks if t['status'] == 'done')
    in_progress = sum(1 for t in tasks if t['status'] == 'in_progress')
    
    print(f'  ✅ Done: {done}')
    print(f'  📝 Todo: {todo}')
    print(f'  🔄 In Progress: {in_progress}')
    print(f'  📊 Total: {len(tasks)}')
    print(f'  📈 Completitud: {done/len(tasks)*100:.1f}%' if tasks else '  No tasks')
except:
    print('  Error parsing response')
"

echo ""
echo "🎉 Sincronización con Archon completada"
