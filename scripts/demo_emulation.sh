#!/bin/bash
# Demo Emulation Script for Quantum Currency System
# Runs a 3-node validation demonstration

echo "🔬 Iniciando Demo de Validación Armónica de 3 Nodos"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "scripts/demo_emulation.py" ]; then
    echo "❌ Error: No se encuentra el script de demostración"
    echo "Por favor ejecutar desde el directorio raíz del proyecto"
    exit 1
fi

# Run the Python demo
echo "🔄 Ejecutando validación armónica..."
python scripts/demo_emulation.py

echo ""
echo "🎯 Demo finalizada"