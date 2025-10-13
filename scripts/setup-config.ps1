# ============================================================================
# AEGIS Framework - Configurador de Archivos para Windows
# ============================================================================
# Descripción: Script para configurar únicamente los archivos de configuración
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

param(
    [switch]$SkipTor,
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
    Write-ColorOutput "🛡️  AEGIS Framework - Configurador de Archivos" "Cyan"
    Write-ColorOutput "===============================================" "Cyan"
    Write-Host ""
    Write-ColorOutput "DESCRIPCIÓN:" "Yellow"
    Write-Host "  Configura únicamente los archivos de configuración para AEGIS Framework"
    Write-Host ""
    Write-ColorOutput "USO:" "Yellow"
    Write-Host "  .\setup-config.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput "OPCIONES:" "Yellow"
    Write-Host "  -SkipTor      Omitir configuración de Tor"
    Write-Host "  -Force        Sobrescribir archivos existentes"
    Write-Host "  -Help         Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput "EJEMPLOS:" "Yellow"
    Write-Host "  .\setup-config.ps1"
    Write-Host "  .\setup-config.ps1 -SkipTor"
    Write-Host "  .\setup-config.ps1 -Force"
    Write-Host ""
    exit 0
}

function Test-Prerequisites {
    Write-ColorOutput "🔍 Verificando prerequisitos..." "Blue"
    
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
    
    return $allGood
}

function New-ConfigDirectory {
    Write-ColorOutput "📁 Creando directorio de configuración..." "Blue"
    
    if (-not (Test-Path "config")) {
        New-Item -ItemType Directory -Path "config" -Force | Out-Null
        Write-ColorOutput "✅ Directorio 'config' creado" "Green"
    } else {
        Write-ColorOutput "✅ Directorio 'config' ya existe" "Green"
    }
    
    # Crear subdirectorios necesarios
    $subdirs = @("tor_data", "logs")
    foreach ($subdir in $subdirs) {
        if (-not (Test-Path $subdir)) {
            New-Item -ItemType Directory -Path $subdir -Force | Out-Null
            Write-ColorOutput "✅ Directorio '$subdir' creado" "Green"
        } else {
            Write-ColorOutput "✅ Directorio '$subdir' ya existe" "Green"
        }
    }
}

function New-EnvFile {
    Write-ColorOutput "🔧 Configurando archivo .env..." "Blue"
    
    $envPath = ".env"
    $envExamplePath = ".env.example"
    
    if ((Test-Path $envPath) -and (-not $Force)) {
        Write-ColorOutput "✅ Archivo .env ya existe (usa -Force para sobrescribir)" "Green"
        return
    }
    
    if (Test-Path $envExamplePath) {
        Copy-Item $envExamplePath $envPath -Force
        Write-ColorOutput "✅ Archivo .env creado desde .env.example" "Green"
    } else {
        # Crear .env básico
        $envContent = @"
# ============================================================================
# AEGIS Framework - Configuración de Variables de Entorno
# ============================================================================

# Blockchain Configuration
PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
RPC_URL=http://localhost:8545
NETWORK_ID=1337

# Tor Configuration
TOR_ENABLED=true
TOR_SOCKS_PORT=9050
TOR_CONTROL_PORT=9051
TOR_PASSWORD=

# Security Configuration
ENCRYPTION_KEY=your_encryption_key_here_32_chars
JWT_SECRET=your_jwt_secret_here_at_least_32_characters_long
SESSION_SECRET=your_session_secret_here_at_least_32_characters

# Network Configuration
P2P_PORT=8888
API_PORT=8080
DASHBOARD_PORT=8080
SECURE_CHAT_PORT=5173
BLOCKCHAIN_RPC_PORT=8545

# Database Configuration (if using external DB)
DATABASE_URL=
REDIS_URL=

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/aegis.log

# Development Configuration
NODE_ENV=development
DEBUG=false

# Optional: External Services
IPFS_GATEWAY=https://ipfs.io/ipfs/
ETHEREUM_MAINNET_RPC=
POLYGON_RPC=

# Optional: Monitoring
SENTRY_DSN=
PROMETHEUS_PORT=9090
"@
        
        $envContent | Out-File -FilePath $envPath -Encoding UTF8
        Write-ColorOutput "✅ Archivo .env creado con configuración básica" "Green"
    }
    
    Write-ColorOutput "⚠️  IMPORTANTE: Edita el archivo .env con tus valores específicos" "Yellow"
}

function New-AppConfigFile {
    Write-ColorOutput "⚙️  Configurando archivo app_config.json..." "Blue"
    
    $configPath = "config\app_config.json"
    $configExamplePath = "config\app_config.example.json"
    
    if ((Test-Path $configPath) -and (-not $Force)) {
        Write-ColorOutput "✅ Archivo app_config.json ya existe (usa -Force para sobrescribir)" "Green"
        return
    }
    
    if (Test-Path $configExamplePath) {
        Copy-Item $configExamplePath $configPath -Force
        Write-ColorOutput "✅ Archivo app_config.json creado desde ejemplo" "Green"
    } else {
        # Crear configuración básica
        $configContent = @{
            "app" = @{
                "name" = "AEGIS Framework"
                "version" = "2.0.0"
                "environment" = "development"
                "debug" = $true
            }
            "server" = @{
                "host" = "localhost"
                "port" = 8080
                "cors_enabled" = $true
                "cors_origins" = @("http://localhost:5173", "http://localhost:3000")
            }
            "security" = @{
                "jwt_expiration" = 3600
                "session_timeout" = 1800
                "max_login_attempts" = 5
                "password_min_length" = 8
            }
            "blockchain" = @{
                "network" = "localhost"
                "rpc_url" = "http://localhost:8545"
                "chain_id" = 1337
                "gas_limit" = 6721975
                "gas_price" = "20000000000"
            }
            "p2p" = @{
                "enabled" = $true
                "port" = 8888
                "max_peers" = 50
                "discovery_enabled" = $true
            }
            "tor" = @{
                "enabled" = $true
                "socks_port" = 9050
                "control_port" = 9051
                "data_directory" = "./tor_data"
            }
            "logging" = @{
                "level" = "INFO"
                "file" = "logs/dashboard.log"
                "max_size" = "10MB"
                "backup_count" = 5
            }
            "features" = @{
                "secure_chat" = $true
                "blockchain_integration" = $true
                "tor_integration" = $true
                "p2p_networking" = $true
            }
        }
        
        $configJson = $configContent | ConvertTo-Json -Depth 10
        $configJson | Out-File -FilePath $configPath -Encoding UTF8
        Write-ColorOutput "✅ Archivo app_config.json creado con configuración básica" "Green"
    }
}

function New-TorrcFile {
    if ($SkipTor) {
        Write-ColorOutput "⏭️  Omitiendo configuración de Tor" "Yellow"
        return
    }
    
    Write-ColorOutput "🧅 Configurando archivo torrc..." "Blue"
    
    $torrcPath = "config\torrc"
    
    if ((Test-Path $torrcPath) -and (-not $Force)) {
        Write-ColorOutput "✅ Archivo torrc ya existe (usa -Force para sobrescribir)" "Green"
        return
    }
    
    $torrcContent = @"
# ============================================================================
# AEGIS Framework - Configuración de Tor
# ============================================================================

# Puerto SOCKS para conexiones de aplicaciones
SocksPort 9050

# Puerto de control para gestión programática
ControlPort 9051

# Directorio de datos de Tor
DataDirectory ./tor_data

# Autenticación por cookie (más segura que contraseña)
CookieAuthentication 1

# Archivo de cookie de autenticación
CookieAuthFile ./tor_data/control_auth_cookie

# Configuración de logging
Log notice file ./logs/tor.log

# Configuración de red
# Usar bridges si es necesario (descomenta las siguientes líneas)
# UseBridges 1
# Bridge obfs4 [IP:Puerto] [Fingerprint]

# Configuración de seguridad
# Evitar nodos de salida en ciertos países (opcional)
# ExcludeExitNodes {us},{gb},{au},{ca},{nz},{dk},{fr},{nl},{no},{be}

# Configuración de rendimiento
# Ancho de banda (opcional, en KB/s)
# BandwidthRate 1024 KB
# BandwidthBurst 2048 KB

# Configuración de circuitos
# Tiempo de vida de circuitos (en segundos)
MaxCircuitDirtiness 600

# Número de saltos en circuitos
# PathsNeededToBuildCircuits 0.95

# Configuración de servicios ocultos (si es necesario)
# HiddenServiceDir ./tor_data/hidden_service/
# HiddenServicePort 80 127.0.0.1:8080

# Configuración de cliente
# ClientOnly 1

# Configuración de DNS
# DNSPort 9053
# AutomapHostsOnResolve 1

# Configuración de transparencia (Linux)
# TransPort 9040

# Configuración de seguridad adicional
# DisableAllSwap 1
# HardwareAccel 1

# Configuración de red Tor
# FascistFirewall 1
# FirewallPorts 80,443,9001,9030

# Configuración de directorio de consenso
# DirReqStatistics 0
# EntryStatistics 0
# ExitPortStatistics 0
"@
    
    $torrcContent | Out-File -FilePath $torrcPath -Encoding UTF8
    Write-ColorOutput "✅ Archivo torrc creado" "Green"
    
    # Crear directorio de datos de Tor si no existe
    if (-not (Test-Path "tor_data")) {
        New-Item -ItemType Directory -Path "tor_data" -Force | Out-Null
        Write-ColorOutput "✅ Directorio tor_data creado" "Green"
    }
}

function New-LogsDirectory {
    Write-ColorOutput "📝 Configurando directorio de logs..." "Blue"
    
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" -Force | Out-Null
        Write-ColorOutput "✅ Directorio 'logs' creado" "Green"
    } else {
        Write-ColorOutput "✅ Directorio 'logs' ya existe" "Green"
    }
    
    # Crear archivos de log vacíos
    $logFiles = @(
        "logs\dashboard.log",
        "logs\secure-chat.log",
        "logs\blockchain.log",
        "logs\tor.log",
        "logs\error.log",
        "logs\access.log"
    )
    
    foreach ($logFile in $logFiles) {
        if (-not (Test-Path $logFile)) {
            New-Item -ItemType File -Path $logFile -Force | Out-Null
        }
    }
    
    Write-ColorOutput "✅ Archivos de log inicializados" "Green"
}

function Test-ConfigurationFiles {
    Write-ColorOutput "🔍 Verificando archivos de configuración..." "Blue"
    
    $requiredFiles = @(
        @{ Path = ".env"; Description = "Variables de entorno" },
        @{ Path = "config\app_config.json"; Description = "Configuración principal" }
    )
    
    if (-not $SkipTor) {
        $requiredFiles += @{ Path = "config\torrc"; Description = "Configuración de Tor" }
    }
    
    $allGood = $true
    
    foreach ($file in $requiredFiles) {
        if (Test-Path $file.Path) {
            Write-ColorOutput "✅ $($file.Description): $($file.Path)" "Green"
        } else {
            Write-ColorOutput "❌ $($file.Description): $($file.Path) - No encontrado" "Red"
            $allGood = $false
        }
    }
    
    # Verificar directorios
    $requiredDirs = @("config", "logs", "tor_data")
    foreach ($dir in $requiredDirs) {
        if (Test-Path $dir) {
            Write-ColorOutput "✅ Directorio: $dir" "Green"
        } else {
            Write-ColorOutput "❌ Directorio: $dir - No encontrado" "Red"
            $allGood = $false
        }
    }
    
    return $allGood
}

function Show-NextSteps {
    Write-ColorOutput "📋 PRÓXIMOS PASOS:" "Yellow"
    Write-Host ""
    Write-Host "1. Edita el archivo .env con tus valores específicos:"
    Write-ColorOutput "   notepad .env" "Cyan"
    Write-Host ""
    Write-Host "2. Revisa la configuración principal:"
    Write-ColorOutput "   notepad config\app_config.json" "Cyan"
    Write-Host ""
    Write-Host "3. Instala las dependencias de Python:"
    Write-ColorOutput "   python -m venv venv" "Cyan"
    Write-ColorOutput "   venv\Scripts\activate" "Cyan"
    Write-ColorOutput "   pip install -r requirements.txt" "Cyan"
    Write-Host ""
    Write-Host "4. Instala las dependencias de Node.js:"
    Write-ColorOutput "   cd dapps\secure-chat\ui && npm install" "Cyan"
    Write-ColorOutput "   cd ..\..\..\dapps\aegis-token && npm install" "Cyan"
    Write-Host ""
    Write-Host "5. Inicia los servicios:"
    Write-ColorOutput "   .\scripts\start-all-services.ps1" "Cyan"
    Write-Host ""
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Main {
    if ($Help) {
        Show-Help
    }
    
    Write-ColorOutput "🛡️  AEGIS Framework - Configurador de Archivos" "Cyan"
    Write-ColorOutput "===============================================" "Cyan"
    Write-Host ""
    
    # Verificar prerequisitos
    if (-not (Test-Prerequisites)) {
        Write-ColorOutput "❌ Los prerequisitos no se cumplen" "Red"
        Write-ColorOutput "💡 Ejecuta primero: .\scripts\install-dependencies.ps1" "Yellow"
        exit 1
    }
    
    Write-Host ""
    Write-ColorOutput "🚀 Iniciando configuración de archivos..." "Blue"
    Write-Host ""
    
    # Crear estructura de directorios
    New-ConfigDirectory
    
    # Crear archivos de configuración
    New-EnvFile
    New-AppConfigFile
    New-TorrcFile
    New-LogsDirectory
    
    Write-Host ""
    Write-ColorOutput "🔍 Verificación final..." "Blue"
    Write-Host ""
    
    # Verificar configuración
    $verificationResult = Test-ConfigurationFiles
    
    Write-Host ""
    Write-ColorOutput "📊 RESUMEN DE CONFIGURACIÓN" "Cyan"
    Write-ColorOutput "============================" "Cyan"
    
    if ($verificationResult) {
        Write-ColorOutput "✅ Todos los archivos de configuración creados correctamente" "Green"
        Write-Host ""
        Write-ColorOutput "🎉 ¡Configuración completada con éxito!" "Green"
        Write-Host ""
        Show-NextSteps
    } else {
        Write-ColorOutput "⚠️  Algunos archivos de configuración no se crearon correctamente" "Yellow"
        Write-Host ""
        Write-ColorOutput "💡 RECOMENDACIONES:" "Yellow"
        Write-Host "1. Ejecuta el script nuevamente con -Force:"
        Write-ColorOutput "   .\scripts\setup-config.ps1 -Force" "Cyan"
        Write-Host "2. Verifica manualmente los archivos faltantes"
        Write-Host "3. Consulta la documentación de solución de problemas"
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