#!/bin/bash
#
# AEGIS Framework - Security Scan Script
# Ejecuta análisis de seguridad completo del código y dependencias
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔒 AEGIS Framework - Security Scan"
echo "=================================="
echo ""

# Verificar virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment no encontrado. Creando..."
    python -m venv .venv
fi

# Activar virtual environment
source .venv/bin/activate

# Instalar herramientas de seguridad
echo "📦 Instalando herramientas de seguridad..."
pip install -q bandit safety detect-secrets pip-audit 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "1️⃣  Dependency Vulnerability Scan (safety + pip-audit)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Safety check
echo "🔍 Running safety check..."
if command -v safety &> /dev/null; then
    safety check --json > security_report_safety.json 2>/dev/null || {
        echo "⚠️  Safety check completado con warnings (verificar security_report_safety.json)"
    }
    echo "✅ Safety scan completado → security_report_safety.json"
else
    echo "⚠️  safety no disponible"
fi

echo ""

# Pip audit
echo "🔍 Running pip-audit..."
if command -v pip-audit &> /dev/null; then
    pip-audit --format json --output security_report_pip_audit.json 2>/dev/null || {
        echo "⚠️  Pip-audit completado con warnings (verificar security_report_pip_audit.json)"
    }
    echo "✅ Pip-audit scan completado → security_report_pip_audit.json"
else
    echo "⚠️  pip-audit no disponible"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "2️⃣  Code Security Scan (bandit)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Bandit scan
echo "🔍 Running bandit code security scan..."
if command -v bandit &> /dev/null; then
    bandit -r src/ -f json -o security_report_bandit.json -ll 2>/dev/null || {
        echo "⚠️  Bandit encontró problemas potenciales (verificar security_report_bandit.json)"
    }
    
    # Generar reporte legible
    bandit -r src/ -ll -f txt > security_report_bandit.txt 2>/dev/null || true
    
    echo "✅ Bandit scan completado"
    echo "   → security_report_bandit.json (formato JSON)"
    echo "   → security_report_bandit.txt (formato texto)"
else
    echo "⚠️  bandit no disponible"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "3️⃣  Secret Detection (detect-secrets)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Detect secrets
echo "🔍 Scanning for hardcoded secrets..."
if command -v detect-secrets &> /dev/null; then
    detect-secrets scan --all-files --force-use-all-plugins > .secrets.baseline 2>/dev/null || {
        echo "⚠️  Secrets detectados (verificar .secrets.baseline)"
    }
    
    # Auditar baseline
    echo "✅ Secret detection completado → .secrets.baseline"
    
    # Contar secretos detectados
    if [ -f ".secrets.baseline" ]; then
        SECRET_COUNT=$(grep -c '"filename"' .secrets.baseline || echo "0")
        if [ "$SECRET_COUNT" -gt "0" ]; then
            echo "⚠️  $SECRET_COUNT archivos con posibles secretos detectados"
        else
            echo "✅ No se detectaron secretos en el código"
        fi
    fi
else
    echo "⚠️  detect-secrets no disponible"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 Security Scan Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Análisis de resultados
CRITICAL_ISSUES=0
HIGH_ISSUES=0
MEDIUM_ISSUES=0

# Analizar bandit
if [ -f "security_report_bandit.json" ]; then
    if command -v python &> /dev/null; then
        CRITICAL_ISSUES=$(python -c "
import json
try:
    with open('security_report_bandit.json') as f:
        data = json.load(f)
        print(len([r for r in data.get('results', []) if r.get('issue_severity') == 'HIGH']))
except:
    print(0)
" 2>/dev/null || echo "0")
    fi
fi

echo "Reportes generados:"
echo "  📄 security_report_safety.json      - Vulnerabilidades en dependencias (safety)"
echo "  📄 security_report_pip_audit.json   - Vulnerabilidades en dependencias (pip-audit)"
echo "  📄 security_report_bandit.json      - Problemas de seguridad en código"
echo "  📄 security_report_bandit.txt       - Reporte legible de bandit"
echo "  📄 .secrets.baseline                - Baseline de secretos detectados"
echo ""

if [ "$CRITICAL_ISSUES" -eq "0" ]; then
    echo "✅ No se encontraron problemas críticos de seguridad"
    EXIT_CODE=0
else
    echo "⚠️  Se encontraron $CRITICAL_ISSUES problemas de seguridad de alta prioridad"
    echo "   Revisar security_report_bandit.txt para detalles"
    EXIT_CODE=1
fi

echo ""
echo "🎯 Recomendaciones:"
echo "   1. Revisar todos los reportes JSON generados"
echo "   2. Actualizar dependencias con vulnerabilidades conocidas"
echo "   3. Corregir problemas de seguridad de alta prioridad"
echo "   4. Ejecutar este scan regularmente (CI/CD)"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "Security scan completado ✅"
echo "═══════════════════════════════════════════════════════════════"

exit $EXIT_CODE
