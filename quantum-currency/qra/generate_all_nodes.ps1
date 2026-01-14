# generate_all_nodes.ps1
# Generate QRA keys for all nodes in the quantum currency system

Write-Host "[QRA] Generating bioresonant QRA keys for all nodes"
python qra/generator.py --generate_all_nodes
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to generate QRA keys"
    exit 1
}
Write-Host "[QRA] Generated bioresonant QRA keys for all nodes"