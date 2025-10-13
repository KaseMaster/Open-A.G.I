# ============================================================================
# AEGIS Framework - Instalador de Dependencias para Windows
# ============================================================================
# Descripción: Script para instalar únicamente las dependencias necesarias
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

param(
    [switch]$SkipTor,
    [switch]$SkipDocker,
    [switch]$Force,
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
    Write-ColorOutput "🛡️  AEGIS Framework - Instalador de Dependencias" "Cyan"
    Write-ColorOutput "=================================================" "Cyan"
    Write-Host ""
    Write-ColorOutput "DESCRIPCIÓN:" "Yellow"
    Write-Host "  Instala únicamente las dependencias necesarias para AEGIS Framework"
    Write-Host ""
    Write-ColorOutput "USO:" "Yellow"
    Write-Host "  .\install-dependencies.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput "OPCIONES:" "Yellow"
    Write-Host "  -SkipTor      Omitir instalación de Tor"
    Write-Host "  -SkipDocker   Omitir instalación de Docker"
    Write-Host "  -Force        Forzar reinstalación de dependencias existentes"
    Write-Host "  -Help         Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput "EJEMPLOS:" "Yellow"
    Write-Host "  .\install-dependencies.ps1"
    Write-Host "  .\install-dependencies.ps1 -SkipTor -SkipDocker"
    Write-Host "  .\install-dependencies.ps1 -Force"
    Write-Host ""
    exit 0
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-SystemRequirements {
    Write-ColorOutput "🔍 Verificando requisitos del sistema..." "Blue"
    
    # Verificar versión de Windows
    $osVersion = [System.Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10) {
        Write-ColorOutput "❌ Error: Se requiere Windows 10 o superior" "Red"
        return $false
    }
    Write-ColorOutput "✅ Windows $($osVersion.Major).$($osVersion.Minor) - Compatible" "Green"
    
    # Verificar PowerShell
    $psVersion = $PSVersionTable.PSVersion
    if ($psVersion.Major -lt 5) {
        Write-ColorOutput "❌ Error: Se requiere PowerShell 5.0 o superior" "Red"
        return $false
    }
    Write-ColorOutput "✅ PowerShell $($psVersion.Major).$($psVersion.Minor) - Compatible" "Green"
    
    # Verificar RAM
    $ram = Get-CimInstance -ClassName Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory
    $ramGB = [math]::Round($ram / 1GB, 2)
    if ($ramGB -lt 4) {
        Write-ColorOutput "⚠️  Advertencia: RAM insuficiente ($ramGB GB). Se recomiendan 4 GB mínimo" "Yellow"
    } else {
        Write-ColorOutput "✅ RAM: $ramGB GB - Suficiente" "Green"
    }
    
    # Verificar espacio en disco
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
    $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
    if ($freeSpaceGB -lt 2) {
        Write-ColorOutput "❌ Error: Espacio insuficiente en disco ($freeSpaceGB GB). Se requieren 2 GB mínimo" "Red"
        return $false
    }
    Write-ColorOutput "✅ Espacio libre: $freeSpaceGB GB - Suficiente" "Green"
    
    return $true
}

function Install-Chocolatey {
    Write-ColorOutput "🍫 Instalando Chocolatey..." "Blue"
    
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        if (-not $Force) {
            Write-ColorOutput "✅ Chocolatey ya está instalado" "Green"
            return $true
        }
        Write-ColorOutput "🔄 Forzando reinstalación de Chocolatey..." "Yellow"
    }
    
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        
        # Actualizar PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        Write-ColorOutput "✅ Chocolatey instalado correctamente" "Green"
        return $true
    }
    catch {
        Write-ColorOutput "❌ Error instalando Chocolatey: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Install-Python {
    Write-ColorOutput "🐍 Instalando Python 3.11..." "Blue"
    
    # Verificar si Python ya está instalado
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and -not $Force) {
        $pythonVersion = & python --version 2>&1
        if ($pythonVersion -match "Python 3\.([8-9]|1[0-9])") {
            Write-ColorOutput "✅ Python ya está instalado: $pythonVersion" "Green"
            return $true
        }
    }
    
    try {
        if ($Force) {
            Write-ColorOutput "🔄 Forzando reinstalación de Python..." "Yellow"
            choco uninstall python -y --force
        }
        
        choco install python --version=3.11.9 -y --force
        
        # Actualizar PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        # Verificar instalación
        Start-Sleep -Seconds 3
        $pythonVersion = & python --version 2>&1
        Write-ColorOutput "✅ Python instalado: $pythonVersion" "Green"
        
        # Actualizar pip
        Write-ColorOutput "📦 Actualizando pip..." "Blue"
        & python -m pip install --upgrade pip
        
        return $true
    }
    catch {
        Write-ColorOutput "❌ Error instalando Python: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Install-NodeJS {
    Write-ColorOutput "📦 Instalando Node.js 20 LTS..." "Blue"
    
    # Verificar si Node.js ya está instalado
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCmd -and -not $Force) {
        $nodeVersion = & node --version 2>&1
        if ($nodeVersion -match "v(1[8-9]|2[0-9])") {
            Write-ColorOutput "✅ Node.js ya está instalado: $nodeVersion" "Green"
            return $true
        }
    }
    
    try {
        if ($Force) {
            Write-ColorOutput "🔄 Forzando reinstalación de Node.js..." "Yellow"
            choco uninstall nodejs -y --force
        }
        
        choco install nodejs --version=20.11.1 -y --force
        
        # Actualizar PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        # Verificar instalación
        Start-Sleep -Seconds 3
        $nodeVersion = & node --version 2>&1
        $npmVersion = & npm --version 2>&1
        Write-ColorOutput "✅ Node.js instalado: $nodeVersion" "Green"
        Write-ColorOutput "✅ NPM instalado: v$npmVersion" "Green"
        
        return $true
    }
    catch {
        Write-ColorOutput "❌ Error instalando Node.js: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Install-Git {
    Write-ColorOutput "📝 Instalando Git..." "Blue"
    
    # Verificar si Git ya está instalado
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCmd -and -not $Force) {
        $gitVersion = & git --version 2>&1
        Write-ColorOutput "✅ Git ya está instalado: $gitVersion" "Green"
        return $true
    }
    
    try {
        if ($Force) {
            Write-ColorOutput "🔄 Forzando reinstalación de Git..." "Yellow"
            choco uninstall git -y --force
        }
        
        choco install git -y --force
        
        # Actualizar PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        # Verificar instalación
        Start-Sleep -Seconds 3
        $gitVersion = & git --version 2>&1
        Write-ColorOutput "✅ Git instalado: $gitVersion" "Green"
        
        return $true
    }
    catch {
        Write-ColorOutput "❌ Error instalando Git: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Install-Tor {
    if ($SkipTor) {
        Write-ColorOutput "⏭️  Omitiendo instalación de Tor" "Yellow"
        return $true
    }
    
    Write-ColorOutput "🧅 Instalando Tor..." "Blue"
    
    # Verificar si Tor ya está instalado
    $torCmd = Get-Command tor -ErrorAction SilentlyContinue
    if ($torCmd -and -not $Force) {
        $torVersion = & tor --version 2>&1 | Select-String "Tor version" | Select-Object -First 1
        Write-ColorOutput "✅ Tor ya está instalado: $torVersion" "Green"
        return $true
    }
    
    try {
        if ($Force) {
            Write-ColorOutput "🔄 Forzando reinstalación de Tor..." "Yellow"
            choco uninstall tor -y --force
        }
        
        choco install tor -y --force
        
        # Actualizar PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        # Verificar instalación
        Start-Sleep -Seconds 3
        $torVersion = & tor --version 2>&1 | Select-String "Tor version" | Select-Object -First 1
        Write-ColorOutput "✅ Tor instalado: $torVersion" "Green"
        
        return $true
    }
    catch {
        Write-ColorOutput "❌ Error instalando Tor: $($_.Exception.Message)" "Red"
        Write-ColorOutput "ℹ️  Tor es opcional. El sistema funcionará sin él." "Cyan"
        return $true
    }
}

function Install-Docker {
    if ($SkipDocker) {
        Write-ColorOutput "⏭️  Omitiendo instalación de Docker" "Yellow"
        return $true
    }
    
    Write-ColorOutput "🐳 Instalando Docker Desktop..." "Blue"
    
    # Verificar si Docker ya está instalado
    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if ($dockerCmd -and -not $Force) {
        $dockerVersion = & docker --version 2>&1
        Write-ColorOutput "✅ Docker ya está instalado: $dockerVersion" "Green"
        return $true
    }
    
    try {
        if ($Force) {
            Write-ColorOutput "🔄 Forzando reinstalación de Docker..." "Yellow"
            choco uninstall docker-desktop -y --force
        }
        
        choco install docker-desktop -y --force
        
        Write-ColorOutput "✅ Docker Desktop instalado" "Green"
        Write-ColorOutput "ℹ️  Nota: Reinicia el sistema para completar la instalación de Docker" "Cyan"
        
        return $true
    }
    catch {
        Write-ColorOutput "❌ Error instalando Docker: $($_.Exception.Message)" "Red"
        Write-ColorOutput "ℹ️  Docker es opcional. El sistema funcionará sin él." "Cyan"
        return $true
    }
}

function Test-Dependencies {
    Write-ColorOutput "🔍 Verificando dependencias instaladas..." "Blue"
    
    $allGood = $true
    
    # Verificar Python
    try {
        $pythonVersion = & python --version 2>&1
        if ($pythonVersion -match "Python 3\.([8-9]|1[0-9])") {
            Write-ColorOutput "✅ Python: $pythonVersion" "Green"
        } else {
            Write-ColorOutput "❌ Python: Versión no compatible" "Red"
            $allGood = $false
        }
    }
    catch {
        Write-ColorOutput "❌ Python: No encontrado" "Red"
        $allGood = $false
    }
    
    # Verificar pip
    try {
        $pipVersion = & python -m pip --version 2>&1
        Write-ColorOutput "✅ pip: $pipVersion" "Green"
    }
    catch {
        Write-ColorOutput "❌ pip: No encontrado" "Red"
        $allGood = $false
    }
    
    # Verificar Node.js
    try {
        $nodeVersion = & node --version 2>&1
        if ($nodeVersion -match "v(1[8-9]|2[0-9])") {
            Write-ColorOutput "✅ Node.js: $nodeVersion" "Green"
        } else {
            Write-ColorOutput "❌ Node.js: Versión no compatible" "Red"
            $allGood = $false
        }
    }
    catch {
        Write-ColorOutput "❌ Node.js: No encontrado" "Red"
        $allGood = $false
    }
    
    # Verificar npm
    try {
        $npmVersion = & npm --version 2>&1
        Write-ColorOutput "✅ npm: v$npmVersion" "Green"
    }
    catch {
        Write-ColorOutput "❌ npm: No encontrado" "Red"
        $allGood = $false
    }
    
    # Verificar Git
    try {
        $gitVersion = & git --version 2>&1
        Write-ColorOutput "✅ Git: $gitVersion" "Green"
    }
    catch {
        Write-ColorOutput "❌ Git: No encontrado" "Red"
        $allGood = $false
    }
    
    # Verificar Tor (opcional)
    if (-not $SkipTor) {
        try {
            $torVersion = & tor --version 2>&1 | Select-String "Tor version" | Select-Object -First 1
            Write-ColorOutput "✅ Tor: $torVersion" "Green"
        }
        catch {
            Write-ColorOutput "⚠️  Tor: No encontrado (opcional)" "Yellow"
        }
    }
    
    # Verificar Docker (opcional)
    if (-not $SkipDocker) {
        try {
            $dockerVersion = & docker --version 2>&1
            Write-ColorOutput "✅ Docker: $dockerVersion" "Green"
        }
        catch {
            Write-ColorOutput "⚠️  Docker: No encontrado (opcional)" "Yellow"
        }
    }
    
    return $allGood
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Main {
    if ($Help) {
        Show-Help
    }
    
    Write-ColorOutput "🛡️  AEGIS Framework - Instalador de Dependencias" "Cyan"
    Write-ColorOutput "=================================================" "Cyan"
    Write-Host ""
    
    # Verificar permisos de administrador
    if (-not (Test-Administrator)) {
        Write-ColorOutput "❌ Error: Se requieren permisos de administrador" "Red"
        Write-ColorOutput "💡 Ejecuta PowerShell como administrador e intenta de nuevo" "Yellow"
        exit 1
    }
    
    # Verificar requisitos del sistema
    if (-not (Test-SystemRequirements)) {
        Write-ColorOutput "❌ Los requisitos del sistema no se cumplen" "Red"
        exit 1
    }
    
    Write-Host ""
    Write-ColorOutput "🚀 Iniciando instalación de dependencias..." "Blue"
    Write-Host ""
    
    # Instalar Chocolatey
    if (-not (Install-Chocolatey)) {
        Write-ColorOutput "❌ Error crítico: No se pudo instalar Chocolatey" "Red"
        exit 1
    }
    
    Write-Host ""
    
    # Instalar dependencias principales
    $installationResults = @()
    
    $installationResults += Install-Python
    $installationResults += Install-NodeJS
    $installationResults += Install-Git
    $installationResults += Install-Tor
    $installationResults += Install-Docker
    
    Write-Host ""
    Write-ColorOutput "🔍 Verificación final..." "Blue"
    Write-Host ""
    
    # Verificar todas las dependencias
    $verificationResult = Test-Dependencies
    
    Write-Host ""
    Write-ColorOutput "📊 RESUMEN DE INSTALACIÓN" "Cyan"
    Write-ColorOutput "=========================" "Cyan"
    
    if ($verificationResult) {
        Write-ColorOutput "✅ Todas las dependencias principales instaladas correctamente" "Green"
        Write-Host ""
        Write-ColorOutput "🎉 ¡Instalación completada con éxito!" "Green"
        Write-Host ""
        Write-ColorOutput "📋 PRÓXIMOS PASOS:" "Yellow"
        Write-Host "1. Ejecuta el script de configuración:"
        Write-ColorOutput "   .\scripts\setup-config.ps1" "Cyan"
        Write-Host "2. O ejecuta el despliegue completo:"
        Write-ColorOutput "   .\scripts\auto-deploy-windows.ps1" "Cyan"
        Write-Host ""
    } else {
        Write-ColorOutput "⚠️  Algunas dependencias no se instalaron correctamente" "Yellow"
        Write-Host ""
        Write-ColorOutput "💡 RECOMENDACIONES:" "Yellow"
        Write-Host "1. Reinicia el sistema"
        Write-Host "2. Ejecuta el script nuevamente con -Force:"
        Write-ColorOutput "   .\scripts\install-dependencies.ps1 -Force" "Cyan"
        Write-Host "3. Verifica manualmente las dependencias faltantes"
        Write-Host ""
    }
    
    Write-ColorOutput "📚 Para más información, consulta:" "Cyan"
    Write-Host "- DEPLOYMENT_GUIDE_COMPLETE.md"
    Write-Host "- DEPENDENCIES_GUIDE.md"
    Write-Host "- TROUBLESHOOTING_GUIDE.md"
    Write-Host ""
}

# Ejecutar función principal
Main