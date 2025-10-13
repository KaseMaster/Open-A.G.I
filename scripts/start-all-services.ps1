# ============================================================================
# AEGIS Framework - Script de Inicio de Servicios (Windows)
# ============================================================================
# Descripción: Inicia todos los servicios del sistema AEGIS de forma ordenada
# Autor: AEGIS Security Team
# Versión: 1.0.0
# Fecha: 2024
# ============================================================================

param(
    [switch]$SkipTor,
    [switch]$SkipBlockchain,
    [switch]$SkipUI,
    [switch]$Verbose,
    [switch]$Help
)

# Configuración de colores
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Cyan"
    Magenta = "Magenta"
    White = "White"
}

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Show-Help {
    Write-ColorOutput "=== AEGIS Framework - Inicio de Servicios ===" "Blue"
    Write-ColorOutput ""
    Write-ColorOutput "USO:" "Yellow"
    Write-ColorOutput "  .\start-all-services.ps1 [OPCIONES]" "White"
    Write-ColorOutput ""
    Write-ColorOutput "OPCIONES:" "Yellow"
    Write-ColorOutput "  -SkipTor          Omitir inicio de Tor" "White"
    Write-ColorOutput "  -SkipBlockchain   Omitir inicio de blockchain local" "White"
    Write-ColorOutput "  -SkipUI           Omitir inicio de interfaces de usuario" "White"
    Write-ColorOutput "  -Verbose          Mostrar salida detallada" "White"
    Write-ColorOutput "  -Help             Mostrar esta ayuda" "White"
    Write-ColorOutput ""
    Write-ColorOutput "EJEMPLOS:" "Yellow"
    Write-ColorOutput "  .\start-all-services.ps1                    # Iniciar todos los servicios" "White"
    Write-ColorOutput "  .\start-all-services.ps1 -SkipTor          # Iniciar sin Tor" "White"
    Write-ColorOutput "  .\start-all-services.ps1 -Verbose          # Modo detallado" "White"
    exit 0
}

function Test-Prerequisites {
    Write-ColorOutput "🔍 Verificando prerequisitos..." "Blue"
    
    $errors = @()
    
    # Verificar Python
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Python: $pythonVersion" "Green"
        } else {
            $errors += "Python no encontrado"
        }
    } catch {
        $errors += "Python no encontrado"
    }
    
    # Verificar Node.js
    try {
        $nodeVersion = node --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Node.js: $nodeVersion" "Green"
        } else {
            $errors += "Node.js no encontrado"
        }
    } catch {
        $errors += "Node.js no encontrado"
    }
    
    # Verificar entorno virtual de Python
    if (Test-Path "venv\Scripts\activate.ps1") {
        Write-ColorOutput "✅ Entorno virtual Python encontrado" "Green"
    } else {
        $errors += "Entorno virtual Python no encontrado (venv/)"
    }
    
    # Verificar archivos de configuración
    $configFiles = @(".env", "config\app_config.json")
    foreach ($file in $configFiles) {
        if (Test-Path $file) {
            Write-ColorOutput "✅ Configuración: $file" "Green"
        } else {
            $errors += "Archivo de configuración faltante: $file"
        }
    }
    
    if ($errors.Count -gt 0) {
        Write-ColorOutput "❌ Errores encontrados:" "Red"
        foreach ($error in $errors) {
            Write-ColorOutput "   - $error" "Red"
        }
        Write-ColorOutput ""
        Write-ColorOutput "💡 Ejecuta el script de despliegue primero:" "Yellow"
        Write-ColorOutput "   .\auto-deploy-windows.ps1" "White"
        exit 1
    }
    
    Write-ColorOutput "✅ Todos los prerequisitos verificados" "Green"
    Write-ColorOutput ""
}

function Start-TorService {
    if ($SkipTor) {
        Write-ColorOutput "⏭️  Omitiendo inicio de Tor" "Yellow"
        return
    }
    
    Write-ColorOutput "🧅 Iniciando servicio Tor..." "Blue"
    
    # Verificar si Tor ya está ejecutándose
    $torProcess = Get-Process -Name "tor" -ErrorAction SilentlyContinue
    if ($torProcess) {
        Write-ColorOutput "⚠️  Tor ya está ejecutándose (PID: $($torProcess.Id))" "Yellow"
        return
    }
    
    # Verificar configuración de Tor
    if (-not (Test-Path "config\torrc")) {
        Write-ColorOutput "❌ Archivo config\torrc no encontrado" "Red"
        return
    }
    
    # Crear directorio de datos de Tor si no existe
    if (-not (Test-Path "tor_data")) {
        New-Item -ItemType Directory -Path "tor_data" -Force | Out-Null
        Write-ColorOutput "📁 Directorio tor_data creado" "Green"
    }
    
    # Iniciar Tor en segundo plano
    try {
        $torJob = Start-Job -ScriptBlock {
            param($configPath)
            Set-Location $using:PWD
            tor -f $configPath
        } -ArgumentList "config\torrc"
        
        # Esperar un momento para verificar que inició correctamente
        Start-Sleep -Seconds 3
        
        $jobState = Get-Job -Id $torJob.Id | Select-Object -ExpandProperty State
        if ($jobState -eq "Running") {
            Write-ColorOutput "✅ Tor iniciado correctamente (Job ID: $($torJob.Id))" "Green"
            Write-ColorOutput "   SOCKS Proxy: 127.0.0.1:9050" "White"
            Write-ColorOutput "   Control Port: 127.0.0.1:9051" "White"
        } else {
            Write-ColorOutput "❌ Error al iniciar Tor" "Red"
            Receive-Job -Id $torJob.Id
        }
    } catch {
        Write-ColorOutput "❌ Error al iniciar Tor: $($_.Exception.Message)" "Red"
    }
    
    Write-ColorOutput ""
}

function Start-Dashboard {
    Write-ColorOutput "🖥️  Iniciando Dashboard AEGIS..." "Blue"
    
    # Activar entorno virtual y iniciar dashboard
    try {
        $dashboardJob = Start-Job -ScriptBlock {
            Set-Location $using:PWD
            & "venv\Scripts\activate.ps1"
            python main.py start-dashboard --config config\app_config.json
        }
        
        # Esperar un momento para verificar que inició
        Start-Sleep -Seconds 5
        
        $jobState = Get-Job -Id $dashboardJob.Id | Select-Object -ExpandProperty State
        if ($jobState -eq "Running") {
            Write-ColorOutput "✅ Dashboard iniciado correctamente (Job ID: $($dashboardJob.Id))" "Green"
            Write-ColorOutput "   URL: http://localhost:8080" "White"
        } else {
            Write-ColorOutput "❌ Error al iniciar Dashboard" "Red"
            if ($Verbose) {
                Receive-Job -Id $dashboardJob.Id
            }
        }
    } catch {
        Write-ColorOutput "❌ Error al iniciar Dashboard: $($_.Exception.Message)" "Red"
    }
    
    Write-ColorOutput ""
}

function Start-SecureChatUI {
    if ($SkipUI) {
        Write-ColorOutput "⏭️  Omitiendo inicio de Secure Chat UI" "Yellow"
        return
    }
    
    Write-ColorOutput "💬 Iniciando Secure Chat UI..." "Blue"
    
    # Verificar directorio
    if (-not (Test-Path "dapps\secure-chat\ui")) {
        Write-ColorOutput "❌ Directorio dapps\secure-chat\ui no encontrado" "Red"
        return
    }
    
    # Verificar node_modules
    if (-not (Test-Path "dapps\secure-chat\ui\node_modules")) {
        Write-ColorOutput "⚠️  Dependencias no instaladas, instalando..." "Yellow"
        Set-Location "dapps\secure-chat\ui"
        npm install
        Set-Location "..\..\..\"
    }
    
    try {
        $uiJob = Start-Job -ScriptBlock {
            Set-Location "$using:PWD\dapps\secure-chat\ui"
            npm run dev
        }
        
        # Esperar un momento para verificar que inició
        Start-Sleep -Seconds 8
        
        $jobState = Get-Job -Id $uiJob.Id | Select-Object -ExpandProperty State
        if ($jobState -eq "Running") {
            Write-ColorOutput "✅ Secure Chat UI iniciado correctamente (Job ID: $($uiJob.Id))" "Green"
            Write-ColorOutput "   URL: http://localhost:5173" "White"
        } else {
            Write-ColorOutput "❌ Error al iniciar Secure Chat UI" "Red"
            if ($Verbose) {
                Receive-Job -Id $uiJob.Id
            }
        }
    } catch {
        Write-ColorOutput "❌ Error al iniciar Secure Chat UI: $($_.Exception.Message)" "Red"
    }
    
    Write-ColorOutput ""
}

function Start-BlockchainNode {
    if ($SkipBlockchain) {
        Write-ColorOutput "⏭️  Omitiendo inicio de blockchain local" "Yellow"
        return
    }
    
    Write-ColorOutput "⛓️  Iniciando nodo blockchain local..." "Blue"
    
    # Verificar directorio
    if (-not (Test-Path "dapps\aegis-token")) {
        Write-ColorOutput "❌ Directorio dapps\aegis-token no encontrado" "Red"
        return
    }
    
    # Verificar node_modules
    if (-not (Test-Path "dapps\aegis-token\node_modules")) {
        Write-ColorOutput "⚠️  Dependencias no instaladas, instalando..." "Yellow"
        Set-Location "dapps\aegis-token"
        npm install
        Set-Location "..\.."
    }
    
    try {
        $blockchainJob = Start-Job -ScriptBlock {
            Set-Location "$using:PWD\dapps\aegis-token"
            npx hardhat node
        }
        
        # Esperar un momento para verificar que inició
        Start-Sleep -Seconds 10
        
        $jobState = Get-Job -Id $blockchainJob.Id | Select-Object -ExpandProperty State
        if ($jobState -eq "Running") {
            Write-ColorOutput "✅ Nodo blockchain iniciado correctamente (Job ID: $($blockchainJob.Id))" "Green"
            Write-ColorOutput "   RPC URL: http://localhost:8545" "White"
            Write-ColorOutput "   Chain ID: 31337" "White"
        } else {
            Write-ColorOutput "❌ Error al iniciar nodo blockchain" "Red"
            if ($Verbose) {
                Receive-Job -Id $blockchainJob.Id
            }
        }
    } catch {
        Write-ColorOutput "❌ Error al iniciar nodo blockchain: $($_.Exception.Message)" "Red"
    }
    
    Write-ColorOutput ""
}

function Test-ServicesHealth {
    Write-ColorOutput "🏥 Verificando salud de servicios..." "Blue"
    
    $services = @(
        @{ Name = "Dashboard"; URL = "http://localhost:8080"; Skip = $false },
        @{ Name = "Secure Chat UI"; URL = "http://localhost:5173"; Skip = $SkipUI },
        @{ Name = "Blockchain RPC"; URL = "http://localhost:8545"; Skip = $SkipBlockchain }
    )
    
    foreach ($service in $services) {
        if ($service.Skip) {
            Write-ColorOutput "⏭️  $($service.Name): Omitido" "Yellow"
            continue
        }
        
        try {
            $response = Invoke-WebRequest -Uri $service.URL -Method HEAD -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-ColorOutput "✅ $($service.Name): Funcionando" "Green"
            } else {
                Write-ColorOutput "⚠️  $($service.Name): Respuesta inesperada ($($response.StatusCode))" "Yellow"
            }
        } catch {
            Write-ColorOutput "❌ $($service.Name): No responde" "Red"
        }
    }
    
    # Verificar Tor SOCKS proxy si no se omitió
    if (-not $SkipTor) {
        try {
            $torTest = Test-NetConnection -ComputerName "127.0.0.1" -Port 9050 -WarningAction SilentlyContinue
            if ($torTest.TcpTestSucceeded) {
                Write-ColorOutput "✅ Tor SOCKS Proxy: Funcionando" "Green"
            } else {
                Write-ColorOutput "❌ Tor SOCKS Proxy: No responde" "Red"
            }
        } catch {
            Write-ColorOutput "❌ Tor SOCKS Proxy: Error de conexión" "Red"
        }
    }
    
    Write-ColorOutput ""
}

function Show-ServiceStatus {
    Write-ColorOutput "📊 Estado de servicios AEGIS:" "Blue"
    Write-ColorOutput "================================" "Blue"
    
    # Mostrar trabajos activos
    $jobs = Get-Job | Where-Object { $_.State -eq "Running" }
    if ($jobs) {
        Write-ColorOutput "🔄 Trabajos en ejecución:" "Green"
        foreach ($job in $jobs) {
            Write-ColorOutput "   Job ID $($job.Id): $($job.Name)" "White"
        }
    } else {
        Write-ColorOutput "⚠️  No hay trabajos en ejecución" "Yellow"
    }
    
    Write-ColorOutput ""
    
    # URLs de acceso
    Write-ColorOutput "🌐 URLs de acceso:" "Green"
    Write-ColorOutput "   Dashboard:      http://localhost:8080" "White"
    if (-not $SkipUI) {
        Write-ColorOutput "   Secure Chat:    http://localhost:5173" "White"
    }
    if (-not $SkipBlockchain) {
        Write-ColorOutput "   Blockchain RPC: http://localhost:8545" "White"
    }
    if (-not $SkipTor) {
        Write-ColorOutput "   Tor SOCKS:      127.0.0.1:9050" "White"
        Write-ColorOutput "   Tor Control:    127.0.0.1:9051" "White"
    }
    
    Write-ColorOutput ""
    Write-ColorOutput "💡 Para detener servicios:" "Yellow"
    Write-ColorOutput "   Get-Job | Stop-Job" "White"
    Write-ColorOutput "   Get-Job | Remove-Job" "White"
    Write-ColorOutput ""
    Write-ColorOutput "📋 Para ver logs:" "Yellow"
    Write-ColorOutput "   Get-Content logs\dashboard.log -Wait" "White"
    Write-ColorOutput "   Get-Content logs\error.log -Wait" "White"
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Main {
    if ($Help) {
        Show-Help
    }
    
    Write-ColorOutput "🚀 AEGIS Framework - Iniciando Servicios" "Blue"
    Write-ColorOutput "=========================================" "Blue"
    Write-ColorOutput ""
    
    # Verificar prerequisitos
    Test-Prerequisites
    
    # Crear directorio de logs si no existe
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" -Force | Out-Null
        Write-ColorOutput "📁 Directorio logs creado" "Green"
    }
    
    # Iniciar servicios en orden
    Write-ColorOutput "🔄 Iniciando servicios..." "Blue"
    Write-ColorOutput ""
    
    # 1. Tor (si está habilitado)
    Start-TorService
    
    # 2. Dashboard (siempre)
    Start-Dashboard
    
    # 3. Blockchain (si está habilitado)
    Start-BlockchainNode
    
    # 4. UI (si está habilitado)
    Start-SecureChatUI
    
    # Esperar un momento para que todos los servicios se estabilicen
    Write-ColorOutput "⏳ Esperando estabilización de servicios..." "Yellow"
    Start-Sleep -Seconds 10
    
    # Verificar salud de servicios
    Test-ServicesHealth
    
    # Mostrar estado final
    Show-ServiceStatus
    
    Write-ColorOutput "🎉 ¡Todos los servicios iniciados!" "Green"
    Write-ColorOutput ""
    Write-ColorOutput "⚠️  IMPORTANTE: Mantén esta ventana abierta para que los servicios sigan funcionando" "Yellow"
    Write-ColorOutput "   Presiona Ctrl+C para detener todos los servicios" "Yellow"
    
    # Mantener el script ejecutándose
    try {
        Write-ColorOutput ""
        Write-ColorOutput "Presiona Ctrl+C para detener todos los servicios..." "White"
        while ($true) {
            Start-Sleep -Seconds 30
            
            # Verificar que los trabajos sigan ejecutándose
            $runningJobs = Get-Job | Where-Object { $_.State -eq "Running" }
            if ($runningJobs.Count -eq 0) {
                Write-ColorOutput "⚠️  Todos los trabajos han terminado" "Yellow"
                break
            }
        }
    } catch {
        Write-ColorOutput ""
        Write-ColorOutput "🛑 Deteniendo servicios..." "Yellow"
        Get-Job | Stop-Job
        Get-Job | Remove-Job
        Write-ColorOutput "✅ Servicios detenidos" "Green"
    }
}

# Ejecutar función principal
Main