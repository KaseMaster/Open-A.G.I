# ============================================================================
# AEGIS Framework - Script de Despliegue en Producción para Windows
# ============================================================================
# Descripción: Script para desplegar AEGIS Framework en entorno de producción
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

param(
    [string]$Environment = "production",
    [string]$Domain = "",
    [string]$SSLCertPath = "",
    [string]$SSLKeyPath = "",
    [switch]$SkipBackup,
    [switch]$SkipTests,
    [switch]$Force,
    [switch]$DryRun,
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

# Configuración de producción
$ProductionConfig = @{
    RequiredMemoryGB = 4
    RequiredDiskSpaceGB = 20
    RequiredPorts = @(80, 443, 8080, 3000, 8545, 9050)
    BackupRetentionDays = 30
    HealthCheckTimeout = 60
    DeploymentTimeout = 300
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Show-Help {
    Write-ColorOutput "🚀 AEGIS Framework - Despliegue en Producción" "Cyan"
    Write-ColorOutput "=============================================" "Cyan"
    Write-Host ""
    Write-ColorOutput "DESCRIPCIÓN:" "Yellow"
    Write-Host "  Despliega AEGIS Framework en un entorno de producción seguro"
    Write-Host ""
    Write-ColorOutput "USO:" "Yellow"
    Write-Host "  .\deploy-production.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput "OPCIONES:" "Yellow"
    Write-Host "  -Environment <env>       Entorno de despliegue (default: production)"
    Write-Host "  -Domain <dominio>        Dominio para el despliegue (ej: aegis.example.com)"
    Write-Host "  -SSLCertPath <ruta>      Ruta al certificado SSL"
    Write-Host "  -SSLKeyPath <ruta>       Ruta a la clave privada SSL"
    Write-Host "  -SkipBackup              Omitir backup antes del despliegue"
    Write-Host "  -SkipTests               Omitir pruebas antes del despliegue"
    Write-Host "  -Force                   Forzar despliegue sin confirmaciones"
    Write-Host "  -DryRun                  Simular despliegue sin ejecutar cambios"
    Write-Host "  -Help                    Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput "EJEMPLOS:" "Yellow"
    Write-Host "  .\deploy-production.ps1 -Domain aegis.company.com"
    Write-Host "  .\deploy-production.ps1 -Domain aegis.company.com -SSLCertPath cert.pem -SSLKeyPath key.pem"
    Write-Host "  .\deploy-production.ps1 -DryRun                              # Simulación"
    Write-Host "  .\deploy-production.ps1 -Force -SkipTests                    # Despliegue rápido"
    Write-Host ""
    Write-ColorOutput "REQUISITOS:" "Yellow"
    Write-Host "  - Windows Server 2019+ o Windows 10/11 Pro"
    Write-Host "  - 4GB+ RAM disponible"
    Write-Host "  - 20GB+ espacio en disco"
    Write-Host "  - Puertos 80, 443, 8080, 3000, 8545, 9050 disponibles"
    Write-Host "  - Certificados SSL válidos (recomendado)"
    Write-Host ""
    exit 0
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Crear directorio de logs si no existe
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" -Force | Out-Null
    }
    
    Add-Content -Path "logs\deployment.log" -Value $logEntry
    
    if ($Level -eq "ERROR") {
        Write-ColorOutput "❌ $Message" "Red"
    }
    elseif ($Level -eq "WARN") {
        Write-ColorOutput "⚠️  $Message" "Yellow"
    }
    elseif ($Level -eq "SUCCESS") {
        Write-ColorOutput "✅ $Message" "Green"
    }
    else {
        Write-ColorOutput "ℹ️  $Message" "White"
    }
}

function Test-Prerequisites {
    Write-ColorOutput "🔍 Verificando requisitos del sistema..." "Cyan"
    
    $issues = @()
    
    # Verificar sistema operativo
    $os = Get-WmiObject -Class Win32_OperatingSystem
    $osVersion = [System.Version]$os.Version
    
    if ($osVersion.Major -lt 10) {
        $issues += "Sistema operativo no soportado. Se requiere Windows 10+ o Windows Server 2019+"
    }
    
    # Verificar memoria RAM
    $totalMemoryGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    if ($totalMemoryGB -lt $ProductionConfig.RequiredMemoryGB) {
        $issues += "Memoria insuficiente: ${totalMemoryGB}GB disponible, ${ProductionConfig.RequiredMemoryGB}GB requerido"
    }
    
    # Verificar espacio en disco
    $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'"
    $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
    if ($freeSpaceGB -lt $ProductionConfig.RequiredDiskSpaceGB) {
        $issues += "Espacio en disco insuficiente: ${freeSpaceGB}GB disponible, ${ProductionConfig.RequiredDiskSpaceGB}GB requerido"
    }
    
    # Verificar puertos
    foreach ($port in $ProductionConfig.RequiredPorts) {
        $portInUse = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($portInUse) {
            $issues += "Puerto $port está en uso"
        }
    }
    
    # Verificar PowerShell
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        $issues += "PowerShell 5.0+ requerido"
    }
    
    # Verificar permisos de administrador
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        $issues += "Se requieren permisos de administrador"
    }
    
    if ($issues.Count -gt 0) {
        Write-Log "Requisitos del sistema no cumplidos:" "ERROR"
        foreach ($issue in $issues) {
            Write-Log "  - $issue" "ERROR"
        }
        return $false
    }
    
    Write-Log "Todos los requisitos del sistema cumplidos" "SUCCESS"
    return $true
}

function Test-Dependencies {
    Write-ColorOutput "🔧 Verificando dependencias..." "Cyan"
    
    $dependencies = @(
        @{ Name = "Python"; Command = "python --version"; MinVersion = "3.8" }
        @{ Name = "Node.js"; Command = "node --version"; MinVersion = "18.0" }
        @{ Name = "npm"; Command = "npm --version"; MinVersion = "8.0" }
        @{ Name = "Git"; Command = "git --version"; MinVersion = "2.0" }
    )
    
    $missing = @()
    
    foreach ($dep in $dependencies) {
        try {
            $output = Invoke-Expression $dep.Command 2>$null
            if ($output) {
                Write-Log "$($dep.Name): $output" "INFO"
            }
            else {
                $missing += $dep.Name
            }
        }
        catch {
            $missing += $dep.Name
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Log "Dependencias faltantes: $($missing -join ', ')" "ERROR"
        Write-Log "Ejecuta .\install-dependencies.ps1 para instalar las dependencias" "ERROR"
        return $false
    }
    
    Write-Log "Todas las dependencias están disponibles" "SUCCESS"
    return $true
}

function Backup-CurrentDeployment {
    if ($SkipBackup) {
        Write-Log "Omitiendo backup (SkipBackup especificado)" "WARN"
        return $true
    }
    
    Write-ColorOutput "💾 Creando backup del despliegue actual..." "Cyan"
    
    $backupDir = "backups\production-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    
    if ($DryRun) {
        Write-Log "DRY RUN: Crearía backup en $backupDir" "INFO"
        return $true
    }
    
    try {
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        
        # Backup de configuración
        $configFiles = @(".env", "config\app_config.json", "config\torrc")
        foreach ($file in $configFiles) {
            if (Test-Path $file) {
                Copy-Item $file "$backupDir\" -Force
                Write-Log "Backup creado: $file" "INFO"
            }
        }
        
        # Backup de datos
        if (Test-Path "data") {
            Copy-Item "data" "$backupDir\data" -Recurse -Force
            Write-Log "Backup de datos creado" "INFO"
        }
        
        # Backup de logs importantes
        if (Test-Path "logs") {
            New-Item -ItemType Directory -Path "$backupDir\logs" -Force | Out-Null
            Get-ChildItem "logs\*.log" | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) } | Copy-Item -Destination "$backupDir\logs\" -Force
            Write-Log "Backup de logs recientes creado" "INFO"
        }
        
        Write-Log "Backup completado en: $backupDir" "SUCCESS"
        return $true
    }
    catch {
        Write-Log "Error creando backup: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Run-Tests {
    if ($SkipTests) {
        Write-Log "Omitiendo pruebas (SkipTests especificado)" "WARN"
        return $true
    }
    
    Write-ColorOutput "🧪 Ejecutando pruebas..." "Cyan"
    
    if ($DryRun) {
        Write-Log "DRY RUN: Ejecutaría pruebas del sistema" "INFO"
        return $true
    }
    
    try {
        # Ejecutar health check
        if (Test-Path "scripts\health-check.ps1") {
            $healthResult = & ".\scripts\health-check.ps1" -Json
            if ($LASTEXITCODE -ne 0) {
                Write-Log "Health check falló" "ERROR"
                return $false
            }
            Write-Log "Health check pasó" "SUCCESS"
        }
        
        # Verificar configuración
        if (Test-Path ".env") {
            $envContent = Get-Content ".env"
            $requiredVars = @("FLASK_ENV", "SECRET_KEY", "DATABASE_URL")
            
            foreach ($var in $requiredVars) {
                if (-not ($envContent | Where-Object { $_ -like "$var=*" })) {
                    Write-Log "Variable de entorno faltante: $var" "ERROR"
                    return $false
                }
            }
            Write-Log "Configuración de entorno validada" "SUCCESS"
        }
        
        return $true
    }
    catch {
        Write-Log "Error ejecutando pruebas: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Setup-ProductionConfig {
    Write-ColorOutput "⚙️  Configurando entorno de producción..." "Cyan"
    
    if ($DryRun) {
        Write-Log "DRY RUN: Configuraría entorno de producción" "INFO"
        return $true
    }
    
    try {
        # Crear configuración de producción
        $prodEnv = @"
# AEGIS Framework - Configuración de Producción
FLASK_ENV=production
DEBUG=False
TESTING=False

# Seguridad
SECRET_KEY=$(New-Guid)
SECURITY_PASSWORD_SALT=$(New-Guid)

# Base de datos
DATABASE_URL=sqlite:///production.db

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=True

# Servicios
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
SECURECHAT_PORT=3000
BLOCKCHAIN_PORT=8545
TOR_PORT=9050

# SSL/TLS
"@

        if ($Domain) {
            $prodEnv += "`nDOMAIN=$Domain"
        }
        
        if ($SSLCertPath -and $SSLKeyPath) {
            $prodEnv += "`nSSL_CERT_PATH=$SSLCertPath"
            $prodEnv += "`nSSL_KEY_PATH=$SSLKeyPath"
            $prodEnv += "`nSSL_ENABLED=True"
        }
        
        # Backup del .env actual si existe
        if (Test-Path ".env") {
            Copy-Item ".env" ".env.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')" -Force
        }
        
        # Escribir nueva configuración
        $prodEnv | Out-File -FilePath ".env" -Encoding UTF8
        Write-Log "Configuración de producción creada" "SUCCESS"
        
        # Configurar app_config.json para producción
        $appConfig = @{
            environment = $Environment
            debug = $false
            logging = @{
                level = "INFO"
                file = "logs/aegis-production.log"
                max_size = "100MB"
                backup_count = 5
            }
            security = @{
                csrf_enabled = $true
                session_timeout = 3600
                max_login_attempts = 5
                password_policy = @{
                    min_length = 12
                    require_uppercase = $true
                    require_lowercase = $true
                    require_numbers = $true
                    require_symbols = $true
                }
            }
            performance = @{
                cache_enabled = $true
                compression_enabled = $true
                static_file_caching = $true
            }
        }
        
        if ($Domain) {
            $appConfig.domain = $Domain
        }
        
        $appConfig | ConvertTo-Json -Depth 10 | Out-File -FilePath "config\app_config.json" -Encoding UTF8
        Write-Log "Configuración de aplicación actualizada" "SUCCESS"
        
        return $true
    }
    catch {
        Write-Log "Error configurando producción: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Deploy-Services {
    Write-ColorOutput "🚀 Desplegando servicios..." "Cyan"
    
    if ($DryRun) {
        Write-Log "DRY RUN: Desplegaría todos los servicios" "INFO"
        return $true
    }
    
    try {
        # Detener servicios existentes
        if (Test-Path "scripts\stop-all-services.ps1") {
            Write-Log "Deteniendo servicios existentes..." "INFO"
            & ".\scripts\stop-all-services.ps1" -Force
        }
        
        # Instalar/actualizar dependencias Python
        Write-Log "Instalando dependencias Python..." "INFO"
        if (Test-Path "venv") {
            & "venv\Scripts\Activate.ps1"
            pip install -r requirements.txt --upgrade
        }
        else {
            python -m venv venv
            & "venv\Scripts\Activate.ps1"
            pip install -r requirements.txt
        }
        
        # Instalar/actualizar dependencias Node.js
        Write-Log "Instalando dependencias Node.js..." "INFO"
        
        # Secure Chat UI
        if (Test-Path "dapps\secure-chat\ui") {
            Set-Location "dapps\secure-chat\ui"
            npm ci --production
            npm run build
            Set-Location "..\..\..\"
        }
        
        # AEGIS Token
        if (Test-Path "dapps\aegis-token") {
            Set-Location "dapps\aegis-token"
            npm ci --production
            Set-Location "..\.."
        }
        
        # Configurar servicios del sistema (si es Windows Server)
        $osInfo = Get-WmiObject -Class Win32_OperatingSystem
        if ($osInfo.ProductType -ne 1) {  # No es workstation (es server)
            Write-Log "Configurando servicios del sistema..." "INFO"
            # Aquí se configurarían servicios de Windows si fuera necesario
        }
        
        # Iniciar servicios
        Write-Log "Iniciando servicios..." "INFO"
        if (Test-Path "scripts\start-all-services.ps1") {
            & ".\scripts\start-all-services.ps1"
        }
        
        Write-Log "Servicios desplegados exitosamente" "SUCCESS"
        return $true
    }
    catch {
        Write-Log "Error desplegando servicios: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Test-Deployment {
    Write-ColorOutput "🔍 Verificando despliegue..." "Cyan"
    
    if ($DryRun) {
        Write-Log "DRY RUN: Verificaría el despliegue" "INFO"
        return $true
    }
    
    $maxAttempts = $ProductionConfig.HealthCheckTimeout / 5
    $attempt = 0
    
    while ($attempt -lt $maxAttempts) {
        $attempt++
        Write-Log "Intento de verificación $attempt/$maxAttempts..." "INFO"
        
        try {
            # Verificar Dashboard
            $dashboardResponse = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
            if ($dashboardResponse.StatusCode -eq 200) {
                Write-Log "Dashboard: OK" "SUCCESS"
            }
            
            # Verificar Secure Chat UI
            $chatResponse = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
            if ($chatResponse.StatusCode -eq 200) {
                Write-Log "Secure Chat UI: OK" "SUCCESS"
            }
            
            # Verificar Blockchain
            try {
                $blockchainTest = Invoke-WebRequest -Uri "http://localhost:8545" -Method POST -Body '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' -ContentType "application/json" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
                if ($blockchainTest.StatusCode -eq 200) {
                    Write-Log "Blockchain: OK" "SUCCESS"
                }
            }
            catch {
                Write-Log "Blockchain: No disponible (normal si no está configurado)" "WARN"
            }
            
            Write-Log "Despliegue verificado exitosamente" "SUCCESS"
            return $true
        }
        catch {
            Write-Log "Verificación falló (intento $attempt): $($_.Exception.Message)" "WARN"
            if ($attempt -lt $maxAttempts) {
                Start-Sleep -Seconds 5
            }
        }
    }
    
    Write-Log "Verificación del despliegue falló después de $maxAttempts intentos" "ERROR"
    return $false
}

function Show-DeploymentSummary {
    Write-ColorOutput "`n🎉 Resumen del Despliegue" "Cyan"
    Write-ColorOutput "=========================" "Cyan"
    
    Write-ColorOutput "✅ Despliegue completado exitosamente" "Green"
    Write-Host ""
    
    Write-ColorOutput "🌐 URLs de Acceso:" "Yellow"
    Write-Host "  Dashboard:      http://localhost:8080"
    Write-Host "  Secure Chat UI: http://localhost:3000"
    Write-Host "  Blockchain RPC: http://localhost:8545"
    
    if ($Domain) {
        Write-Host ""
        Write-ColorOutput "🌍 URLs Públicas:" "Yellow"
        Write-Host "  Dashboard:      https://$Domain:8080"
        Write-Host "  Secure Chat UI: https://$Domain:3000"
    }
    
    Write-Host ""
    Write-ColorOutput "📋 Próximos Pasos:" "Yellow"
    Write-Host "  1. Verificar que todos los servicios estén funcionando"
    Write-Host "  2. Configurar firewall para los puertos necesarios"
    Write-Host "  3. Configurar certificados SSL si no se hizo"
    Write-Host "  4. Configurar monitoreo y alertas"
    Write-Host "  5. Realizar backup regular de la configuración"
    
    Write-Host ""
    Write-ColorOutput "🔧 Comandos Útiles:" "Yellow"
    Write-Host "  Verificar estado:    .\scripts\health-check.ps1"
    Write-Host "  Ver logs:           Get-Content logs\aegis-production.log -Tail 50"
    Write-Host "  Reiniciar servicios: .\scripts\stop-all-services.ps1; .\scripts\start-all-services.ps1"
    Write-Host "  Monitorear:         .\scripts\monitor-services.ps1 -Continuous"
    
    Write-Host ""
    Write-ColorOutput "⚠️  Recordatorios de Seguridad:" "Yellow"
    Write-Host "  - Cambiar contraseñas por defecto"
    Write-Host "  - Configurar firewall apropiadamente"
    Write-Host "  - Habilitar logging de auditoría"
    Write-Host "  - Configurar backups automáticos"
    Write-Host "  - Revisar configuración de Tor"
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Main {
    if ($Help) {
        Show-Help
    }
    
    Write-ColorOutput "🚀 AEGIS Framework - Despliegue en Producción" "Cyan"
    Write-ColorOutput "=============================================" "Cyan"
    Write-Host ""
    
    # Verificar directorio
    if (-not (Test-Path "main.py")) {
        Write-Log "No se encontró main.py. Ejecuta este script desde el directorio raíz del proyecto AEGIS." "ERROR"
        exit 1
    }
    
    # Confirmación si no es DryRun y no es Force
    if (-not $DryRun -and -not $Force) {
        Write-ColorOutput "⚠️  ADVERTENCIA: Este script desplegará AEGIS Framework en producción." "Yellow"
        Write-ColorOutput "Esto puede sobrescribir configuraciones existentes." "Yellow"
        Write-Host ""
        $confirmation = Read-Host "¿Continuar con el despliegue? (y/N)"
        if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
            Write-ColorOutput "Despliegue cancelado por el usuario." "Yellow"
            exit 0
        }
    }
    
    $startTime = Get-Date
    Write-Log "Iniciando despliegue en producción..." "INFO"
    
    # Ejecutar pasos del despliegue
    $steps = @(
        @{ Name = "Verificar requisitos"; Function = { Test-Prerequisites } }
        @{ Name = "Verificar dependencias"; Function = { Test-Dependencies } }
        @{ Name = "Crear backup"; Function = { Backup-CurrentDeployment } }
        @{ Name = "Ejecutar pruebas"; Function = { Run-Tests } }
        @{ Name = "Configurar producción"; Function = { Setup-ProductionConfig } }
        @{ Name = "Desplegar servicios"; Function = { Deploy-Services } }
        @{ Name = "Verificar despliegue"; Function = { Test-Deployment } }
    )
    
    foreach ($step in $steps) {
        Write-ColorOutput "`n📋 $($step.Name)..." "Blue"
        
        $success = & $step.Function
        
        if (-not $success) {
            Write-Log "Paso falló: $($step.Name)" "ERROR"
            Write-ColorOutput "`n❌ Despliegue falló en: $($step.Name)" "Red"
            Write-ColorOutput "Revisa los logs en logs\deployment.log para más detalles." "Yellow"
            exit 1
        }
    }
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Log "Despliegue completado en $($duration.TotalMinutes.ToString('F2')) minutos" "SUCCESS"
    
    if (-not $DryRun) {
        Show-DeploymentSummary
    }
    else {
        Write-ColorOutput "`n✅ Simulación de despliegue completada exitosamente" "Green"
        Write-ColorOutput "Ejecuta sin -DryRun para realizar el despliegue real." "Yellow"
    }
}

# Ejecutar función principal
Main