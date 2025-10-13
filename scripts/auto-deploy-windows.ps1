# ========================================
# AEGIS Framework - Auto Deploy Windows
# ========================================
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: Diciembre 2024
# ========================================

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("full", "deps", "config", "dev", "prod")]
    [string]$Mode = "full",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTor,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# Configuración de colores para output
$Host.UI.RawUI.ForegroundColor = "White"

# Funciones de utilidad
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✅ $Message" "Green"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "❌ $Message" "Red"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput "⚠️  $Message" "Yellow"
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput "ℹ️  $Message" "Cyan"
}

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-ColorOutput "🚀 $Message" "Magenta"
    Write-ColorOutput ("=" * ($Message.Length + 4)) "Magenta"
}

# Verificar si se ejecuta como administrador
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Verificar requisitos del sistema
function Test-SystemRequirements {
    Write-Header "Verificando Requisitos del Sistema"
    
    # Verificar versión de Windows
    $osVersion = [System.Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10) {
        Write-Error "Se requiere Windows 10 o superior. Versión actual: $($osVersion)"
        return $false
    }
    Write-Success "Versión de Windows: $($osVersion) ✓"
    
    # Verificar PowerShell
    $psVersion = $PSVersionTable.PSVersion
    if ($psVersion.Major -lt 5) {
        Write-Error "Se requiere PowerShell 5.0 o superior. Versión actual: $($psVersion)"
        return $false
    }
    Write-Success "PowerShell versión: $($psVersion) ✓"
    
    # Verificar memoria RAM
    $totalRAM = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
    if ($totalRAM -lt 8) {
        Write-Warning "RAM disponible: ${totalRAM}GB. Se recomienda 8GB o más."
    } else {
        Write-Success "RAM disponible: ${totalRAM}GB ✓"
    }
    
    # Verificar espacio en disco
    $freeSpace = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
    if ($freeSpace -lt 10) {
        Write-Error "Espacio libre insuficiente: ${freeSpace}GB. Se requieren al menos 10GB."
        return $false
    }
    Write-Success "Espacio libre: ${freeSpace}GB ✓"
    
    return $true
}

# Instalar Chocolatey
function Install-Chocolatey {
    Write-Header "Instalando Chocolatey"
    
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Success "Chocolatey ya está instalado"
        return $true
    }
    
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        
        # Actualizar PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        Write-Success "Chocolatey instalado correctamente"
        return $true
    }
    catch {
        Write-Error "Error instalando Chocolatey: $($_.Exception.Message)"
        return $false
    }
}

# Instalar dependencias principales
function Install-Dependencies {
    Write-Header "Instalando Dependencias Principales"
    
    $dependencies = @(
        @{Name="python"; Version="3.11.0"; Description="Python 3.11"},
        @{Name="nodejs"; Version="20.10.0"; Description="Node.js 20 LTS"},
        @{Name="git"; Version=""; Description="Git"},
        @{Name="tor"; Version=""; Description="Tor Browser"}
    )
    
    foreach ($dep in $dependencies) {
        Write-Info "Instalando $($dep.Description)..."
        
        try {
            if ($dep.Version) {
                choco install $dep.Name --version=$dep.Version -y --force
            } else {
                choco install $dep.Name -y --force
            }
            Write-Success "$($dep.Description) instalado correctamente"
        }
        catch {
            Write-Error "Error instalando $($dep.Description): $($_.Exception.Message)"
            if (-not $Force) {
                return $false
            }
        }
    }
    
    # Actualizar PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
    
    return $true
}

# Verificar instalaciones
function Test-Dependencies {
    Write-Header "Verificando Instalaciones"
    
    $commands = @(
        @{Name="python"; Command="python --version"; Description="Python"},
        @{Name="pip"; Command="pip --version"; Description="Pip"},
        @{Name="node"; Command="node --version"; Description="Node.js"},
        @{Name="npm"; Command="npm --version"; Description="NPM"},
        @{Name="git"; Command="git --version"; Description="Git"}
    )
    
    $allOk = $true
    
    foreach ($cmd in $commands) {
        try {
            $result = Invoke-Expression $cmd.Command 2>$null
            Write-Success "$($cmd.Description): $result"
        }
        catch {
            Write-Error "$($cmd.Description) no está disponible"
            $allOk = $false
        }
    }
    
    if (-not $SkipTor) {
        try {
            $torPath = Get-Command tor -ErrorAction SilentlyContinue
            if ($torPath) {
                Write-Success "Tor: Instalado en $($torPath.Source)"
            } else {
                Write-Warning "Tor no encontrado en PATH, pero puede estar instalado"
            }
        }
        catch {
            Write-Warning "No se pudo verificar la instalación de Tor"
        }
    }
    
    return $allOk
}

# Configurar proyecto
function Initialize-Project {
    Write-Header "Configurando Proyecto AEGIS"
    
    # Verificar si estamos en el directorio correcto
    if (-not (Test-Path "main.py")) {
        Write-Error "No se encontró main.py. Asegúrate de estar en el directorio raíz del proyecto."
        return $false
    }
    
    try {
        # Crear entorno virtual Python
        Write-Info "Creando entorno virtual Python..."
        python -m venv venv
        
        # Activar entorno virtual
        Write-Info "Activando entorno virtual..."
        & ".\venv\Scripts\Activate.ps1"
        
        # Instalar dependencias Python
        Write-Info "Instalando dependencias Python..."
        pip install --upgrade pip
        pip install -r requirements.txt
        
        if (Test-Path "requirements-dev.txt") {
            pip install -r requirements-dev.txt
        }
        
        Write-Success "Dependencias Python instaladas"
        
        # Instalar dependencias Node.js para secure-chat
        if (Test-Path "dapps\secure-chat\ui\package.json") {
            Write-Info "Instalando dependencias de Secure Chat UI..."
            Set-Location "dapps\secure-chat\ui"
            npm install
            Set-Location "..\..\..\"
            Write-Success "Dependencias de Secure Chat UI instaladas"
        }
        
        # Instalar dependencias Node.js para aegis-token
        if (Test-Path "dapps\aegis-token\package.json") {
            Write-Info "Instalando dependencias de AEGIS Token..."
            Set-Location "dapps\aegis-token"
            npm install
            Set-Location "..\..\"
            Write-Success "Dependencias de AEGIS Token instaladas"
        }
        
        return $true
    }
    catch {
        Write-Error "Error configurando proyecto: $($_.Exception.Message)"
        return $false
    }
}

# Configurar archivos de configuración
function Initialize-Configuration {
    Write-Header "Configurando Archivos de Configuración"
    
    try {
        # Crear directorio de configuración si no existe
        if (-not (Test-Path "config")) {
            New-Item -ItemType Directory -Path "config" -Force
        }
        
        # Copiar archivos de ejemplo
        if (Test-Path ".env.example" -and -not (Test-Path ".env")) {
            Copy-Item ".env.example" ".env"
            Write-Success "Archivo .env creado desde .env.example"
        }
        
        if (Test-Path "config\config.example.yml" -and -not (Test-Path "config\config.yml")) {
            Copy-Item "config\config.example.yml" "config\config.yml"
            Write-Success "Archivo config.yml creado desde config.example.yml"
        }
        
        # Generar configuración de Tor si no existe
        if (-not (Test-Path "config\torrc")) {
            $torrcContent = @"
# Configuración Tor para AEGIS
ControlPort 9051
HashedControlPassword 16:872860B76453A77D60CA2BB8C1A7042072093276A3D701AD684053EC4C

# Servicio oculto
HiddenServiceDir ./tor_data/aegis_service/
HiddenServicePort 80 127.0.0.1:8080
HiddenServicePort 3000 127.0.0.1:3000

# Configuración de rendimiento
NumEntryGuards 8
CircuitBuildTimeout 30
LearnCircuitBuildTimeout 0
"@
            $torrcContent | Out-File -FilePath "config\torrc" -Encoding UTF8
            Write-Success "Archivo torrc creado"
        }
        
        # Crear directorio de datos Tor
        if (-not (Test-Path "tor_data")) {
            New-Item -ItemType Directory -Path "tor_data" -Force
            New-Item -ItemType Directory -Path "tor_data\aegis_service" -Force
            Write-Success "Directorio de datos Tor creado"
        }
        
        # Crear directorio de logs
        if (-not (Test-Path "logs")) {
            New-Item -ItemType Directory -Path "logs" -Force
            Write-Success "Directorio de logs creado"
        }
        
        return $true
    }
    catch {
        Write-Error "Error configurando archivos: $($_.Exception.Message)"
        return $false
    }
}

# Ejecutar pruebas de verificación
function Test-Installation {
    Write-Header "Ejecutando Pruebas de Verificación"
    
    try {
        # Activar entorno virtual
        & ".\venv\Scripts\Activate.ps1"
        
        # Ejecutar pruebas básicas
        if (Test-Path "scripts\system_validation_test.py") {
            Write-Info "Ejecutando pruebas de validación del sistema..."
            python scripts\system_validation_test.py
        }
        
        # Probar importaciones críticas
        Write-Info "Verificando importaciones Python críticas..."
        python -c "
import sys
import flask
import requests
import cryptography
print('✅ Todas las importaciones críticas funcionan correctamente')
"
        
        Write-Success "Todas las pruebas pasaron correctamente"
        return $true
    }
    catch {
        Write-Error "Error en las pruebas: $($_.Exception.Message)"
        return $false
    }
}

# Mostrar información de finalización
function Show-CompletionInfo {
    Write-Header "Instalación Completada"
    
    Write-Success "AEGIS Framework ha sido instalado correctamente en Windows"
    Write-Host ""
    
    Write-Info "Para iniciar los servicios, ejecuta los siguientes comandos:"
    Write-Host ""
    Write-ColorOutput "# Terminal 1 - Dashboard Principal:" "Yellow"
    Write-Host "python main.py start-dashboard --config .\config\app_config.json"
    Write-Host ""
    Write-ColorOutput "# Terminal 2 - Tor Service:" "Yellow"
    Write-Host "tor -f .\config\torrc"
    Write-Host ""
    Write-ColorOutput "# Terminal 3 - Secure Chat UI:" "Yellow"
    Write-Host "cd dapps\secure-chat\ui"
    Write-Host "npm run dev"
    Write-Host ""
    Write-ColorOutput "# Terminal 4 - Blockchain Local:" "Yellow"
    Write-Host "cd dapps\aegis-token"
    Write-Host "npx hardhat node"
    Write-Host ""
    
    Write-Info "URLs de acceso:"
    Write-Host "• Dashboard: http://localhost:8080"
    Write-Host "• Secure Chat: http://localhost:5173"
    Write-Host "• Blockchain: http://localhost:8545"
    Write-Host ""
    
    Write-Info "Documentación adicional:"
    Write-Host "• Guía completa: docs\DEPLOYMENT_GUIDE_COMPLETE.md"
    Write-Host "• Troubleshooting: docs\TROUBLESHOOTING.md"
    Write-Host "• Configuración: docs\CONFIGURATION.md"
}

# Función principal
function Main {
    Write-Header "AEGIS Framework - Auto Deploy Windows v2.0.0"
    
    # Verificar permisos de administrador
    if (-not (Test-Administrator)) {
        Write-Error "Este script debe ejecutarse como Administrador"
        Write-Info "Haz clic derecho en PowerShell y selecciona 'Ejecutar como administrador'"
        exit 1
    }
    
    Write-Info "Modo de instalación: $Mode"
    if ($SkipTor) { Write-Info "Tor será omitido" }
    if ($Verbose) { Write-Info "Modo verbose activado" }
    if ($Force) { Write-Info "Modo force activado - continuará ante errores" }
    
    # Verificar requisitos del sistema
    if (-not (Test-SystemRequirements)) {
        Write-Error "Los requisitos del sistema no se cumplen"
        exit 1
    }
    
    # Ejecutar según el modo seleccionado
    switch ($Mode) {
        "deps" {
            if (-not (Install-Chocolatey)) { exit 1 }
            if (-not (Install-Dependencies)) { exit 1 }
            if (-not (Test-Dependencies)) { exit 1 }
        }
        "config" {
            if (-not (Initialize-Configuration)) { exit 1 }
        }
        "dev" {
            if (-not (Install-Chocolatey)) { exit 1 }
            if (-not (Install-Dependencies)) { exit 1 }
            if (-not (Test-Dependencies)) { exit 1 }
            if (-not (Initialize-Project)) { exit 1 }
            if (-not (Initialize-Configuration)) { exit 1 }
            # En modo dev, no ejecutar pruebas automáticas
        }
        "prod" {
            if (-not (Install-Chocolatey)) { exit 1 }
            if (-not (Install-Dependencies)) { exit 1 }
            if (-not (Test-Dependencies)) { exit 1 }
            if (-not (Initialize-Project)) { exit 1 }
            if (-not (Initialize-Configuration)) { exit 1 }
            if (-not (Test-Installation)) { exit 1 }
        }
        default { # "full"
            if (-not (Install-Chocolatey)) { exit 1 }
            if (-not (Install-Dependencies)) { exit 1 }
            if (-not (Test-Dependencies)) { exit 1 }
            if (-not (Initialize-Project)) { exit 1 }
            if (-not (Initialize-Configuration)) { exit 1 }
            if (-not (Test-Installation)) { exit 1 }
        }
    }
    
    Show-CompletionInfo
    Write-Success "¡Instalación completada exitosamente!"
}

# Manejo de errores global
trap {
    Write-Error "Error inesperado: $($_.Exception.Message)"
    Write-Info "Línea: $($_.InvocationInfo.ScriptLineNumber)"
    Write-Info "Para soporte, consulta: docs\TROUBLESHOOTING.md"
    exit 1
}

# Ejecutar función principal
Main