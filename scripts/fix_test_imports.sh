#!/bin/bash
# Script para actualizar imports en tests
# Convierte imports antiguos a nuevos paths (src.aegis.*)

echo "🔄 Actualizando imports en tests..."
echo ""

cd tests

# Backup de tests
echo "📦 Creando backup..."
mkdir -p ../backup/tests_backup_$(date +%Y%m%d_%H%M%S)
cp *.py ../backup/tests_backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
echo "✓ Backup creado"
echo ""

# Actualizar imports
echo "🔧 Actualizando imports..."

# Crypto framework
echo "  - crypto_framework"
sed -i 's/from crypto_framework/from src.aegis.security.crypto_framework/g' *.py 2>/dev/null || true
sed -i 's/import crypto_framework/import src.aegis.security.crypto_framework as crypto_framework/g' *.py 2>/dev/null || true

# Consensus
echo "  - consensus_protocol"
sed -i 's/from consensus_protocol/from src.aegis.blockchain.consensus_protocol/g' *.py 2>/dev/null || true
sed -i 's/import consensus_protocol/import src.aegis.blockchain.consensus_protocol as consensus_protocol/g' *.py 2>/dev/null || true

echo "  - consensus_algorithm"
sed -i 's/from consensus_algorithm/from src.aegis.blockchain.consensus_algorithm/g' *.py 2>/dev/null || true
sed -i 's/import consensus_algorithm/import src.aegis.blockchain.consensus_algorithm as consensus_algorithm/g' *.py 2>/dev/null || true

# P2P Network
echo "  - p2p_network"
sed -i 's/import p2p_network as p2p/from src.aegis.networking import p2p_network as p2p/g' *.py 2>/dev/null || true
sed -i 's/from p2p_network/from src.aegis.networking.p2p_network/g' *.py 2>/dev/null || true

# Main CLI
echo "  - main (CLI)"
sed -i 's/from main import/from src.aegis.cli.main import/g' *.py 2>/dev/null || true

echo "✓ Imports actualizados"
echo ""

# Ejecutar tests para verificar
echo "🧪 Ejecutando tests para verificar..."
cd ..

python3 -m pytest tests/ -v --tb=short 2>&1 | tee test_results.log

# Contar resultados
PASSED=$(grep -c "PASSED" test_results.log 2>/dev/null || echo "0")
FAILED=$(grep -c "FAILED" test_results.log 2>/dev/null || echo "0")
ERRORS=$(grep -c "ERROR" test_results.log 2>/dev/null || echo "0")

echo ""
echo "📊 Resultados:"
echo "   ✓ Pasados: $PASSED"
echo "   ✗ Fallidos: $FAILED"
echo "   ⚠ Errores: $ERRORS"

if [ $FAILED -eq 0 ] && [ $ERRORS -eq 0 ]; then
    echo ""
    echo "✅ ¡Todos los tests pasaron!"
else
    echo ""
    echo "⚠️  Algunos tests fallaron. Ver test_results.log para detalles"
fi
