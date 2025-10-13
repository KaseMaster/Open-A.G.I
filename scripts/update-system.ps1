# ============================================================================
# AEGIS Framework - Script de Actualización del Sistema para Windows
# ============================================================================
# Descripción: Script para actualizar dependencias y componentes del sistema
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

param(
    [switch]$CheckOnly,
    [switch]$Python,
    [switch]$NodeJS,
    [switch]$System,
    [switch]$All,
    [switch]$Force,
    [switch]$Backup,
    [switch]$Help
)

# Configuración de colores
$Colors = @{
    Red = 'Red'
    Green = 'Green'
    Yellow = 'Yellow'
    Blue = 'Blue'
    Cyan = 'Cyan'
    Magenta = 'Magenta'
    White = 'White'
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Show-Help {
    Write-ColorOutput "🛡️  AEGIS Framework - Actualizador del Sistema" "Cyan"
    Write-ColorOutput "===============================================" "Cyan"
    Write-Host ""
    Write-ColorOutput "DESCRIPCIÓN:" "Yellow"
    Write-Host "  Actualiza dependencias y componentes del sistema AEGIS"
    Write-Host ""
    Write-ColorOutput "USO:" "Yellow"
    Write-Host "  .\update-system.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput "OPCIONES:" "Yellow"
    Write-Host "  -CheckOnly          Solo verificar actualizaciones disponibles"
    Write-Host "  -Python             Actualizar dependencias de Python"
    Write-Host "  -NodeJS             Actualizar dependencias de Node.js"
    Write-Host "  -System             Actualizar herramientas del sistema"
    Write-Host "  -All                Actualizar todo (Python + Node.js + Sistema)"
    Write-Host "  -Force              Forzar reinstalación de dependencias"
    Write-Host "  -Backup             Crear respaldo antes de actualizar"
    Write-Host "  -Help               Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput "EJEMPLOS:" "Yellow"
    Write-Host "  .\update-system.ps1 -CheckOnly                        # Solo verificar"
    Write-Host "  .\update-system.ps1 -Python                          # Actualizar Python"
    Write-Host "  .\update-system.ps1 -NodeJS                          # Actualizar Node.js"
    Write-Host "  .\update-system.ps1 -All -Backup                     # Actualizar todo con respaldo"
    Write-Host "  .\update-system.ps1 -System -Force                   # Forzar actualización del sistema"
    Write-Host ""
    exit 0
}

function Test-Prerequisites {
    Write-ColorOutput "🔍 Verificando prerrequisitos..." "Blue"
    
    # Verificar si estamos en el directorio correcto
    if (-not (Test-Path "main.py")) {
        Write-ColorOutput "❌ Error: No se encontró main.py. Ejecuta este script desde el directorio raíz del proyecto AEGIS." "Red"
        exit 1
    }
    
    # Verificar PowerShell
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Write-ColorOutput "❌ Error: Se requiere PowerShell 5.0 o superior" "Red"
        exit 1
    }
    
    Write-ColorOutput "✅ Prerrequisitos verificados" "Green"
}

function Test-InternetConnection {
    Write-ColorOutput "🌐 Verificando conexión a internet..." "Blue"
    
    try {
        $response = Invoke-WebRequest -Uri "https://www.google.com" -TimeoutSec 10 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-ColorOutput "✅ Conexión a internet verificada" "Green"
            return $true
        }
    }
    catch {
        Write-ColorOutput "❌ Error: No hay conexión a internet" "Red"
        return $false
    }
    
    return $false
}

function Get-PythonUpdates {
    Write-ColorOutput "🐍 Verificando actualizaciones de Python..." "Blue"
    
    $updates = @()
    
    # Verificar si existe requirements.txt
    if (Test-Path "requirements.txt") {
        try {
            # Activar entorno virtual si existe
            if (Test-Path "venv\Scripts\Activate.ps1") {
                & "venv\Scripts\Activate.ps1"
            }
            
            # Verificar paquetes desactualizados
            $outdated = pip list --outdated --format=json 2>$null | ConvertFrom-Json
            
            if ($outdated) {
                foreach ($package in $outdated) {
                    $updates += @{
                        Type = "Python"
                        Name = $package.name
                        CurrentVersion = $package.version
                        LatestVersion = $package.latest_version
                    }
                }
            }
        }
        catch {
            Write-ColorOutput "⚠️  Error verificando paquetes de Python: $($_.Exception.Message)" "Yellow"
        }
    }
    
    return $updates
}

function Get-NodeJSUpdates {
    Write-ColorOutput "📦 Verificando actualizaciones de Node.js..." "Blue"
    
    $updates = @()
    
    # Verificar Secure Chat UI
    if (Test-Path "dapps\secure-chat\ui\package.json") {
        try {
            Push-Location "dapps\secure-chat\ui"
            
            $outdated = npm outdated --json 2>$null | ConvertFrom-Json
            
            if ($outdated) {
                foreach ($package in $outdated.PSObject.Properties) {
                    $updates += @{
                        Type = "Node.js (Secure Chat UI)"
                        Name = $package.Name
                        CurrentVersion = $package.Value.current
                        LatestVersion = $package.Value.latest
                        Location = "dapps\secure-chat\ui"
                    }
                }
            }
            
            Pop-Location
        }
        catch {
            Write-ColorOutput "⚠️  Error verificando paquetes de Secure Chat UI: $($_.Exception.Message)" "Yellow"
            Pop-Location
        }
    }
    
    # Verificar AEGIS Token
    if (Test-Path "dapps\aegis-token\package.json") {
        try {
            Push-Location "dapps\aegis-token"
            
            $outdated = npm outdated --json 2>$null | ConvertFrom-Json
            
            if ($outdated) {
                foreach ($package in $outdated.PSObject.Properties) {
                    $updates += @{
                        Type = "Node.js (AEGIS Token)"
                        Name = $package.Name
                        CurrentVersion = $package.Value.current
                        LatestVersion = $package.Value.latest
                        Location = "dapps\aegis-token"
                    }
                }
            }
            
            Pop-Location
        }
        catch {
            Write-ColorOutput "⚠️  Error verificando paquetes de AEGIS Token: $($_.Exception.Message)" "Yellow"
            Pop-Location
        }
    }
    
    return $updates
}

function Get-SystemUpdates {
    Write-ColorOutput "🔧 Verificando actualizaciones del sistema..." "Blue"
    
    $updates = @()
    
    # Verificar Chocolatey si está instalado
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        try {
            $chocoOutdated = choco outdated --limit-output --ignore-unfound 2>$null
            
            if ($chocoOutdated) {
                foreach ($line in $chocoOutdated -split "`n") {
                    if ($line -and $line -notmatch "^Chocolatey") {
                        $parts = $line -split "\|"
                        if ($parts.Length -ge 3) {
                            $updates += @{
                                Type = "Sistema (Chocolatey)"
                                Name = $parts[0]
                                CurrentVersion = $parts[1]
                                LatestVersion = $parts[2]
                            }
                        }
                    }
                }
            }
        }
        catch {
            Write-ColorOutput "⚠️  Error verificando actualizaciones de Chocolatey: $($_.Exception.Message)" "Yellow"
        }
    }
    
    return $updates
}

function Show-UpdateSummary {
    param([array]$Updates)
    
    if ($Updates.Count -eq 0) {
        Write-ColorOutput "✅ No se encontraron actualizaciones disponibles" "Green"
        return
    }
    
    Write-ColorOutput "📋 Actualizaciones Disponibles:" "Cyan"
    Write-ColorOutput "===============================" "Cyan"
    Write-Host ""
    
    $groupedUpdates = $Updates | Group-Object Type
    
    foreach ($group in $groupedUpdates) {
        Write-ColorOutput "🔸 $($group.Name) ($($group.Count) paquetes):" "Yellow"
        
        foreach ($update in $group.Group) {
            Write-ColorOutput "  📦 $($update.Name): $($update.CurrentVersion) → $($update.LatestVersion)" "White"
        }
        
        Write-Host ""
    }
}

function Update-PythonPackages {
    param([bool]$Force)
    
    Write-ColorOutput "🐍 Actualizando paquetes de Python..." "Blue"
    
    if (-not (Test-Path "requirements.txt")) {
        Write-ColorOutput "⚠️  No se encontró requirements.txt" "Yellow"
        return
    }
    
    try {
        # Activar entorno virtual si existe
        if (Test-Path "venv\Scripts\Activate.ps1") {
            & "venv\Scripts\Activate.ps1"
        }
        
        if ($Force) {
            Write-ColorOutput "🔄 Reinstalando todas las dependencias de Python..." "Blue"
            pip install --force-reinstall -r requirements.txt
        }
        else {
            Write-ColorOutput "🔄 Actualizando dependencias de Python..." "Blue"
            pip install --upgrade -r requirements.txt
        }
        
        Write-ColorOutput "✅ Paquetes de Python actualizados" "Green"
    }
    catch {
        Write-ColorOutput "❌ Error actualizando paquetes de Python: $($_.Exception.Message)" "Red"
    }
}

function Update-NodeJSPackages {
    param([bool]$Force)
    
    Write-ColorOutput "📦 Actualizando paquetes de Node.js..." "Blue"
    
    # Actualizar Secure Chat UI
    if (Test-Path "dapps\secure-chat\ui\package.json") {
        try {
            Write-ColorOutput "🔄 Actualizando Secure Chat UI..." "Blue"
            Push-Location "dapps\secure-chat\ui"
            
            if ($Force) {
                Remove-Item "node_modules" -Recurse -Force -ErrorAction SilentlyContinue
                Remove-Item "package-lock.json" -Force -ErrorAction SilentlyContinue
                npm install
            }
            else {
                npm update
            }
            
            Pop-Location
            Write-ColorOutput "✅ Secure Chat UI actualizado" "Green"
        }
        catch {
            Write-ColorOutput "❌ Error actualizando Secure Chat UI: $($_.Exception.Message)" "Red"
            Pop-Location
        }
    }
    
    # Actualizar AEGIS Token
    if (Test-Path "dapps\aegis-token\package.json") {
        try {
            Write-ColorOutput "🔄 Actualizando AEGIS Token..." "Blue"
            Push-Location "dapps\aegis-token"
            
            if ($Force) {
                Remove-Item "node_modules" -Recurse -Force -ErrorAction SilentlyContinue
                Remove-Item "package-lock.json" -Force -ErrorAction SilentlyContinue
                npm install
            }
            else {
                npm update
            }
            
            Pop-Location
            Write-ColorOutput "✅ AEGIS Token actualizado" "Green"
        }
        catch {
            Write-ColorOutput "❌ Error actualizando AEGIS Token: $($_.Exception.Message)" "Red"
            Pop-Location
        }
    }
}

function Update-SystemPackages {
    param([bool]$Force)
    
    Write-ColorOutput "🔧 Actualizando herramientas del sistema..." "Blue"
    
    # Actualizar Chocolatey si está instalado
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        try {
            Write-ColorOutput "🔄 Actualizando paquetes de Chocolatey..." "Blue"
            
            if ($Force) {
                choco upgrade all --yes --force
            }
            else {
                choco upgrade all --yes
            }
            
            Write-ColorOutput "✅ Herramientas del sistema actualizadas" "Green"
        }
        catch {
            Write-ColorOutput "❌ Error actualizando herramientas del sistema: $($_.Exception.Message)" "Red"
        }
    }
    else {
        Write-ColorOutput "⚠️  Chocolatey no está instalado. Saltando actualización del sistema." "Yellow"
    }
}

function New-BackupBeforeUpdate {
    Write-ColorOutput "💾 Creando respaldo antes de la actualización..." "Blue"
    
    if (Test-Path "scripts\backup-config.ps1") {
        try {
            & "scripts\backup-config.ps1" -BackupPath "backups\pre-update"
            Write-ColorOutput "✅ Respaldo creado exitosamente" "Green"
        }
        catch {
            Write-ColorOutput "❌ Error creando respaldo: $($_.Exception.Message)" "Red"
            
            $continue = Read-Host "¿Deseas continuar sin respaldo? (s/N)"
            if ($continue -ne "s" -and $continue -ne "S") {
                Write-ColorOutput "❌ Actualización cancelada por el usuario" "Yellow"
                exit 0
            }
        }
    }
    else {
        Write-ColorOutput "⚠️  Script de respaldo no encontrado. Continuando sin respaldo." "Yellow"
    }
}

function Test-UpdateSuccess {
    Write-ColorOutput "🔍 Verificando éxito de la actualización..." "Blue"
    
    $success = $true
    
    # Verificar Python
    if (Test-Path "requirements.txt") {
        try {
            if (Test-Path "venv\Scripts\Activate.ps1") {
                & "venv\Scripts\Activate.ps1"
            }
            
            pip check 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-ColorOutput "⚠️  Advertencia: Conflictos detectados en paquetes de Python" "Yellow"
                $success = $false
            }
        }
        catch {
            Write-ColorOutput "⚠️  Error verificando paquetes de Python" "Yellow"
            $success = $false
        }
    }
    
    # Verificar Node.js
    $nodeProjects = @("dapps\secure-chat\ui", "dapps\aegis-token")
    
    foreach ($project in $nodeProjects) {
        if (Test-Path "$project\package.json") {
            try {
                Push-Location $project
                npm audit --audit-level=high 2>$null
                if ($LASTEXITCODE -ne 0) {
                    Write-ColorOutput "⚠️  Advertencia: Vulnerabilidades detectadas en $project" "Yellow"
                    $success = $false
                }
                Pop-Location
            }
            catch {
                Write-ColorOutput "⚠️  Error verificando $project" "Yellow"
                Pop-Location
                $success = $false
            }
        }
    }
    
    if ($success) {
        Write-ColorOutput "✅ Actualización completada exitosamente" "Green"
    }
    else {
        Write-ColorOutput "⚠️  Actualización completada con advertencias" "Yellow"
    }
    
    return $success
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Main {
    if ($Help) {
        Show-Help
    }
    
    Test-Prerequisites
    
    if (-not (Test-InternetConnection)) {
        Write-ColorOutput "❌ Error: Se requiere conexión a internet para actualizar" "Red"
        exit 1
    }
    
    # Crear respaldo si se solicita
    if ($Backup) {
        New-BackupBeforeUpdate
    }
    
    # Recopilar actualizaciones disponibles
    $allUpdates = @()
    
    if ($Python -or $All) {
        $allUpdates += Get-PythonUpdates
    }
    
    if ($NodeJS -or $All) {
        $allUpdates += Get-NodeJSUpdates
    }
    
    if ($System -or $All) {
        $allUpdates += Get-SystemUpdates
    }
    
    # Si no se especifica ninguna opción, verificar todo
    if (-not ($Python -or $NodeJS -or $System -or $All)) {
        $allUpdates += Get-PythonUpdates
        $allUpdates += Get-NodeJSUpdates
        $allUpdates += Get-SystemUpdates
    }
    
    # Mostrar resumen de actualizaciones
    Show-UpdateSummary -Updates $allUpdates
    
    # Si solo es verificación, salir
    if ($CheckOnly) {
        Write-ColorOutput "🔍 Verificación completada" "Blue"
        exit 0
    }
    
    # Si no hay actualizaciones, salir
    if ($allUpdates.Count -eq 0) {
        exit 0
    }
    
    # Confirmar actualización
    if (-not $Force) {
        $confirm = Read-Host "¿Deseas proceder con las actualizaciones? (s/N)"
        if ($confirm -ne "s" -and $confirm -ne "S") {
            Write-ColorOutput "❌ Actualización cancelada por el usuario" "Yellow"
            exit 0
        }
    }
    
    # Realizar actualizaciones
    Write-ColorOutput "🚀 Iniciando proceso de actualización..." "Blue"
    Write-Host ""
    
    if ($Python -or $All) {
        Update-PythonPackages -Force $Force
    }
    
    if ($NodeJS -or $All) {
        Update-NodeJSPackages -Force $Force
    }
    
    if ($System -or $All) {
        Update-SystemPackages -Force $Force
    }
    
    # Verificar éxito de la actualización
    Write-Host ""
    Test-UpdateSuccess
    
    Write-Host ""
    Write-ColorOutput "💡 Recomendación: Reinicia los servicios para aplicar las actualizaciones" "Yellow"
    Write-ColorOutput "   Usa: .\scripts\stop-all-services.ps1 && .\scripts\start-all-services.ps1" "White"
}

# Ejecutar función principal
Main