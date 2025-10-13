# ============================================================================
# AEGIS Framework - Script de Detención de Servicios (Windows)
# ============================================================================
# Descripción: Detiene todos los servicios del sistema AEGIS de forma segura
# Autor: AEGIS Security Team
# Versión: 1.0.0
# Fecha: 2024
# ============================================================================

param(
    [switch]$Force,
    [switch]$Verbose,
    [switch]$Help
)

# Configuración de colores
$Colors = @{
    Red     = "Red"
    Green   = "Green"
    Yellow  = "Yellow"
    Blue    = "Blue"
    Magenta = "Magenta"
    Cyan    = "Cyan"
    White   = "White"
}

# Variables globales
$PidsFile = ".\logs\service_pids.txt"
$LogFile = ".\logs\stop_services.log"

# Funciones de utilidad
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Log-Info {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [INFO] $Message"
    Write-ColorOutput "[INFO] $Message" -Color $Colors.Blue
    Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
}

function Log-Success {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [SUCCESS] $Message"
    Write-ColorOutput "[SUCCESS] $Message" -Color $Colors.Green
    Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
}

function Log-Warning {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [WARNING] $Message"
    Write-ColorOutput "[WARNING] $Message" -Color $Colors.Yellow
    Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
}

function Log-Error {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [ERROR] $Message"
    Write-ColorOutput "[ERROR] $Message" -Color $Colors.Red
    Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
}

function Log-Debug {
    param([string]$Message)
    if ($Verbose) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $logMessage = "[$timestamp] [DEBUG] $Message"
        Write-ColorOutput "[DEBUG] $Message" -Color $Colors.Cyan
        Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
    }
}

function Show-Help {
    Write-ColorOutput "=== AEGIS Framework - Detención de Servicios ===" -Color $Colors.Blue
    Write-Host ""
    Write-ColorOutput "USO:" -Color $Colors.Yellow
    Write-Host "  .\stop-all-services.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput "OPCIONES:" -Color $Colors.Yellow
    Write-Host "  -Force      Forzar detención inmediata (kill)"
    Write-Host "  -Verbose    Mostrar salida detallada"
    Write-Host "  -Help       Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput "EJEMPLOS:" -Color $Colors.Yellow
    Write-Host "  .\stop-all-services.ps1           # Detención normal"
    Write-Host "  .\stop-all-services.ps1 -Force   # Detención forzada"
    Write-Host "  .\stop-all-services.ps1 -Verbose # Modo detallado"
    exit 0
}

function Stop-ProcessSafely {
    param(
        [string]$ProcessName,
        [int]$ProcessId,
        [bool]$ForceKill = $false
    )
    
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        
        if (-not $process) {
            Log-Warning "Proceso $ProcessName (PID: $ProcessId) ya no existe"
            return $true
        }
        
        Log-Info "Deteniendo $ProcessName (PID: $ProcessId)..."
        
        if ($ForceKill) {
            Log-Debug "Forzando detención de $ProcessName"
            $process.Kill()
            Start-Sleep -Seconds 1
        } else {
            # Intentar detención suave primero
            Log-Debug "Intentando detención suave de $ProcessName"
            
            if ($process.ProcessName -eq "node") {
                # Para procesos Node.js, enviar Ctrl+C
                try {
                    [System.Diagnostics.Process]::Start("taskkill", "/PID $ProcessId /T") | Out-Null
                } catch {
                    $process.CloseMainWindow() | Out-Null
                }
            } elseif ($process.ProcessName -eq "python") {
                # Para procesos Python, intentar cerrar ventana principal
                $process.CloseMainWindow() | Out-Null
            } elseif ($process.ProcessName -eq "tor") {
                # Para Tor, usar taskkill con tree
                try {
                    Start-Process -FilePath "taskkill" -ArgumentList "/PID", $ProcessId, "/T", "/F" -Wait -NoNewWindow
                } catch {
                    $process.Kill()
                }
            } else {
                # Método genérico
                $process.CloseMainWindow() | Out-Null
            }
            
            # Esperar un momento para detención suave
            Start-Sleep -Seconds 3
            
            # Verificar si el proceso sigue ejecutándose
            $stillRunning = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
            if ($stillRunning) {
                Log-Warning "Detención suave falló, forzando detención de $ProcessName"
                $process.Kill()
                Start-Sleep -Seconds 1
            }
        }
        
        # Verificación final
        $finalCheck = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (-not $finalCheck) {
            Log-Success "✅ $ProcessName detenido correctamente"
            return $true
        } else {
            Log-Error "❌ No se pudo detener $ProcessName"
            return $false
        }
        
    } catch {
        Log-Error "Error al detener $ProcessName (PID: $ProcessId): $($_.Exception.Message)"
        return $false
    }
}

function Stop-ProcessByName {
    param(
        [string]$ProcessName,
        [string]$CommandLineFilter = "",
        [bool]$ForceKill = $false
    )
    
    try {
        $processes = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
        
        if (-not $processes) {
            Log-Info "No se encontraron procesos '$ProcessName'"
            return
        }
        
        foreach ($process in $processes) {
            # Filtrar por línea de comandos si se especifica
            if ($CommandLineFilter -ne "") {
                try {
                    $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($process.Id)").CommandLine
                    if ($commandLine -notlike "*$CommandLineFilter*") {
                        continue
                    }
                } catch {
                    Log-Debug "No se pudo obtener línea de comandos para PID $($process.Id)"
                }
            }
            
            Stop-ProcessSafely -ProcessName $ProcessName -ProcessId $process.Id -ForceKill $ForceKill
        }
    } catch {
        Log-Error "Error al buscar procesos '$ProcessName': $($_.Exception.Message)"
    }
}

function Stop-ServicesFromPidFile {
    if (-not (Test-Path $PidsFile)) {
        Log-Warning "⚠️  Archivo de PIDs no encontrado: $PidsFile"
        return
    }
    
    Log-Info "📋 Leyendo servicios desde archivo de PIDs..."
    
    try {
        $pidEntries = Get-Content $PidsFile -ErrorAction Stop
        $stoppedCount = 0
        $totalCount = 0
        
        foreach ($entry in $pidEntries) {
            if ($entry -match "^(.+):(\d+)$") {
                $serviceName = $matches[1]
                $pid = [int]$matches[2]
                $totalCount++
                
                Log-Debug "Procesando entrada: $serviceName (PID: $pid)"
                
                if (Stop-ProcessSafely -ProcessName $serviceName -ProcessId $pid -ForceKill $Force) {
                    $stoppedCount++
                }
            } else {
                Log-Warning "Formato de entrada inválido: $entry"
            }
        }
        
        Log-Success "✅ Detenidos $stoppedCount de $totalCount servicios del archivo PID"
        
        # Eliminar archivo de PIDs
        Remove-Item $PidsFile -Force -ErrorAction SilentlyContinue
        Log-Debug "Archivo de PIDs eliminado"
        
    } catch {
        Log-Error "Error al leer archivo de PIDs: $($_.Exception.Message)"
    }
}

function Stop-ServicesByName {
    Log-Info "🔍 Buscando servicios AEGIS por nombre de proceso..."
    
    # Definir servicios conocidos con sus filtros
    $services = @(
        @{ Name = "python"; Filter = "main.py"; Description = "Dashboard AEGIS" },
        @{ Name = "node"; Filter = "vite"; Description = "Secure Chat UI (Vite)" },
        @{ Name = "node"; Filter = "hardhat"; Description = "Blockchain Local (Hardhat)" },
        @{ Name = "tor"; Filter = "torrc"; Description = "Tor Service" }
    )
    
    foreach ($service in $services) {
        Log-Info "Buscando $($service.Description)..."
        Stop-ProcessByName -ProcessName $service.Name -CommandLineFilter $service.Filter -ForceKill $Force
    }
}

function Stop-AdditionalProcesses {
    Log-Info "🧹 Limpiando procesos adicionales..."
    
    # Procesos que podrían quedar ejecutándose
    $additionalProcesses = @(
        "npm",
        "npx"
    )
    
    foreach ($processName in $additionalProcesses) {
        $processes = Get-Process -Name $processName -ErrorAction SilentlyContinue
        if ($processes) {
            Log-Info "Deteniendo procesos $processName..."
            foreach ($process in $processes) {
                Stop-ProcessSafely -ProcessName $processName -ProcessId $process.Id -ForceKill $true
            }
        }
    }
}

function Check-RemainingProcesses {
    Log-Info "🔍 Verificando procesos restantes..."
    
    $aegisProcesses = @()
    
    # Buscar procesos Python con main.py
    $pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue
    foreach ($process in $pythonProcesses) {
        try {
            $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($process.Id)").CommandLine
            if ($commandLine -like "*main.py*") {
                $aegisProcesses += @{ Name = "Python AEGIS"; PID = $process.Id }
            }
        } catch { }
    }
    
    # Buscar procesos Node.js relacionados
    $nodeProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue
    foreach ($process in $nodeProcesses) {
        try {
            $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($process.Id)").CommandLine
            if ($commandLine -like "*vite*" -or $commandLine -like "*hardhat*") {
                $aegisProcesses += @{ Name = "Node.js AEGIS"; PID = $process.Id }
            }
        } catch { }
    }
    
    # Buscar procesos Tor
    $torProcesses = Get-Process -Name "tor" -ErrorAction SilentlyContinue
    foreach ($process in $torProcesses) {
        $aegisProcesses += @{ Name = "Tor"; PID = $process.Id }
    }
    
    if ($aegisProcesses.Count -gt 0) {
        Log-Warning "⚠️  Procesos AEGIS aún ejecutándose:"
        foreach ($proc in $aegisProcesses) {
            Write-Host "   - $($proc.Name) (PID: $($proc.PID))"
        }
        
        if ($Force) {
            Log-Info "Forzando detención de procesos restantes..."
            foreach ($proc in $aegisProcesses) {
                Stop-ProcessSafely -ProcessName $proc.Name -ProcessId $proc.PID -ForceKill $true
            }
        } else {
            Log-Warning "💡 Usa -Force para detener procesos restantes"
        }
    } else {
        Log-Success "✅ No se encontraron procesos AEGIS ejecutándose"
    }
}

function Test-PortsAvailability {
    Log-Info "🔌 Verificando disponibilidad de puertos..."
    
    $ports = @(8080, 5173, 8545, 9050, 9051)
    $busyPorts = @()
    
    foreach ($port in $ports) {
        try {
            $connection = Test-NetConnection -ComputerName "localhost" -Port $port -WarningAction SilentlyContinue
            if ($connection.TcpTestSucceeded) {
                $busyPorts += $port
            }
        } catch {
            # Puerto disponible
        }
    }
    
    if ($busyPorts.Count -gt 0) {
        Log-Warning "⚠️  Puertos aún ocupados: $($busyPorts -join ', ')"
        Log-Warning "   Algunos servicios podrían seguir ejecutándose"
    } else {
        Log-Success "✅ Todos los puertos AEGIS están disponibles"
    }
}

function Clean-LogFiles {
    Log-Info "🧹 Limpiando archivos de log antiguos..."
    
    $logFiles = @(
        ".\logs\dashboard.log",
        ".\logs\secure-chat.log",
        ".\logs\blockchain.log",
        ".\logs\tor.log"
    )
    
    foreach ($logFile in $logFiles) {
        if (Test-Path $logFile) {
            try {
                # Mantener solo las últimas 1000 líneas
                $content = Get-Content $logFile -Tail 1000 -ErrorAction SilentlyContinue
                if ($content) {
                    $content | Set-Content $logFile -ErrorAction SilentlyContinue
                    Log-Debug "Log truncado: $logFile"
                }
            } catch {
                Log-Debug "No se pudo truncar log: $logFile"
            }
        }
    }
}

function Show-StopSummary {
    Write-Host ""
    Write-ColorOutput "📊 Resumen de detención de servicios AEGIS:" -Color $Colors.Blue
    Write-ColorOutput "===========================================" -Color $Colors.Blue
    
    Log-Success "🛑 Proceso de detención completado"
    
    # Mostrar estado de puertos
    Test-PortsAvailability
    
    Write-Host ""
    Log-Success "💡 Para reiniciar servicios:"
    Write-Host "   .\scripts\start-all-services.ps1"
    
    Write-Host ""
    Log-Success "📋 Para verificar estado:"
    Write-Host "   Get-Process python, node, tor -ErrorAction SilentlyContinue"
    
    Write-Host ""
    Log-Success "📁 Logs disponibles en:"
    Write-Host "   .\logs\"
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Main {
    # Mostrar ayuda si se solicita
    if ($Help) {
        Show-Help
    }
    
    Write-ColorOutput "🛑 AEGIS Framework - Deteniendo Servicios" -Color $Colors.Blue
    Write-ColorOutput "==========================================" -Color $Colors.Blue
    Write-Host ""
    
    # Crear directorio de logs si no existe
    if (-not (Test-Path ".\logs")) {
        New-Item -ItemType Directory -Path ".\logs" -Force | Out-Null
    }
    
    # Inicializar log
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] Iniciando detención de servicios AEGIS" | Set-Content $LogFile
    
    if ($Force) {
        Log-Warning "⚠️  Modo FORZADO activado - detención inmediata"
    }
    
    if ($Verbose) {
        Log-Info "🔍 Modo VERBOSE activado"
    }
    
    Write-Host ""
    
    # 1. Detener servicios desde archivo PID (método preferido)
    Stop-ServicesFromPidFile
    
    Write-Host ""
    
    # 2. Buscar y detener servicios por nombre
    Stop-ServicesByName
    
    Write-Host ""
    
    # 3. Limpiar procesos adicionales
    Stop-AdditionalProcesses
    
    Write-Host ""
    
    # 4. Verificar procesos restantes
    Check-RemainingProcesses
    
    Write-Host ""
    
    # 5. Limpiar logs si es necesario
    if (-not $Verbose) {
        Clean-LogFiles
    }
    
    # 6. Mostrar resumen final
    Show-StopSummary
    
    Log-Success "✅ ¡Detención de servicios completada!"
}

# Ejecutar función principal
try {
    Main
} catch {
    Log-Error "Error crítico durante la detención: $($_.Exception.Message)"
    Write-Host ""
    Log-Warning "💡 Si los servicios siguen ejecutándose, usa:"
    Write-Host "   .\stop-all-services.ps1 -Force"
    exit 1
}