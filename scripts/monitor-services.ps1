# ============================================================================
# AEGIS Framework - Script de Monitoreo de Servicios para Windows
# ============================================================================
# Descripción: Script para monitorear el estado de todos los servicios AEGIS
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

param(
    [int]$Interval = 30,
    [switch]$Continuous,
    [switch]$Alerts,
    [switch]$LogToFile,
    [string]$LogPath = "logs\monitor.log",
    [switch]$Json,
    [switch]$Detailed,
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

# Configuración de servicios
$Services = @{
    Dashboard = @{
        Name = "AEGIS Dashboard"
        Port = 8080
        HealthEndpoint = "http://localhost:8080/health"
        ProcessName = "python"
        ProcessArgs = "main.py start-dashboard"
    }
    SecureChatUI = @{
        Name = "Secure Chat UI"
        Port = 3000
        HealthEndpoint = "http://localhost:3000"
        ProcessName = "node"
        ProcessArgs = "npm run dev"
    }
    Blockchain = @{
        Name = "Local Blockchain"
        Port = 8545
        HealthEndpoint = "http://localhost:8545"
        ProcessName = "node"
        ProcessArgs = "npx hardhat node"
    }
    Tor = @{
        Name = "Tor Service"
        Port = 9050
        ProcessName = "tor"
        ProcessArgs = "-f"
    }
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Show-Help {
    Write-ColorOutput "🛡️  AEGIS Framework - Monitor de Servicios" "Cyan"
    Write-ColorOutput "===========================================" "Cyan"
    Write-Host ""
    Write-ColorOutput "DESCRIPCIÓN:" "Yellow"
    Write-Host "  Monitorea el estado de todos los servicios AEGIS en tiempo real"
    Write-Host ""
    Write-ColorOutput "USO:" "Yellow"
    Write-Host "  .\monitor-services.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput "OPCIONES:" "Yellow"
    Write-Host "  -Interval <segundos>    Intervalo entre verificaciones (default: 30)"
    Write-Host "  -Continuous             Monitoreo continuo (Ctrl+C para salir)"
    Write-Host "  -Alerts                 Mostrar alertas cuando servicios fallen"
    Write-Host "  -LogToFile              Guardar logs en archivo"
    Write-Host "  -LogPath <ruta>         Ruta del archivo de log (default: logs\monitor.log)"
    Write-Host "  -Json                   Salida en formato JSON"
    Write-Host "  -Detailed               Información detallada de cada servicio"
    Write-Host "  -Help                   Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput "EJEMPLOS:" "Yellow"
    Write-Host "  .\monitor-services.ps1                                    # Verificación única"
    Write-Host "  .\monitor-services.ps1 -Continuous                       # Monitoreo continuo"
    Write-Host "  .\monitor-services.ps1 -Continuous -Interval 10          # Cada 10 segundos"
    Write-Host "  .\monitor-services.ps1 -Alerts -LogToFile                # Con alertas y logs"
    Write-Host "  .\monitor-services.ps1 -Json                             # Salida JSON"
    Write-Host "  .\monitor-services.ps1 -Detailed                         # Información detallada"
    Write-Host ""
    exit 0
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    if ($LogToFile) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $logEntry = "[$timestamp] [$Level] $Message"
        
        # Crear directorio si no existe
        $logDir = Split-Path $LogPath -Parent
        if ($logDir -and -not (Test-Path $logDir)) {
            New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        }
        
        Add-Content -Path $LogPath -Value $logEntry
    }
}

function Test-Port {
    param(
        [int]$Port,
        [string]$Host = "localhost",
        [int]$TimeoutMs = 3000
    )
    
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $tcpClient.BeginConnect($Host, $Port, $null, $null)
        $wait = $asyncResult.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        
        if ($wait) {
            $tcpClient.EndConnect($asyncResult)
            $tcpClient.Close()
            return $true
        }
        else {
            $tcpClient.Close()
            return $false
        }
    }
    catch {
        return $false
    }
}

function Test-HttpEndpoint {
    param(
        [string]$Url,
        [int]$TimeoutSec = 5
    )
    
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSec -UseBasicParsing -ErrorAction Stop
        return @{
            Success = $true
            StatusCode = $response.StatusCode
            ResponseTime = 0  # PowerShell no proporciona tiempo de respuesta fácilmente
        }
    }
    catch {
        return @{
            Success = $false
            Error = $_.Exception.Message
            StatusCode = $null
        }
    }
}

function Get-ProcessInfo {
    param(
        [string]$ProcessName,
        [string]$ProcessArgs
    )
    
    $processes = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
    
    if ($processes) {
        foreach ($process in $processes) {
            try {
                $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($process.Id)").CommandLine
                if ($commandLine -and $commandLine -like "*$ProcessArgs*") {
                    return @{
                        Found = $true
                        PID = $process.Id
                        CPU = $process.CPU
                        Memory = [math]::Round($process.WorkingSet64 / 1MB, 2)
                        StartTime = $process.StartTime
                        CommandLine = $commandLine
                    }
                }
            }
            catch {
                # Ignorar errores de acceso a procesos
            }
        }
    }
    
    return @{
        Found = $false
    }
}

function Get-ServiceStatus {
    param(
        [string]$ServiceKey,
        [hashtable]$ServiceConfig
    )
    
    $status = @{
        Name = $ServiceConfig.Name
        Key = $ServiceKey
        Status = "Unknown"
        Port = $ServiceConfig.Port
        Process = @{}
        Health = @{}
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    }
    
    # Verificar puerto
    if ($ServiceConfig.Port) {
        $portOpen = Test-Port -Port $ServiceConfig.Port
        $status.Port = @{
            Number = $ServiceConfig.Port
            Open = $portOpen
        }
    }
    
    # Verificar proceso
    if ($ServiceConfig.ProcessName) {
        $processInfo = Get-ProcessInfo -ProcessName $ServiceConfig.ProcessName -ProcessArgs $ServiceConfig.ProcessArgs
        $status.Process = $processInfo
    }
    
    # Verificar endpoint de salud
    if ($ServiceConfig.HealthEndpoint) {
        $healthCheck = Test-HttpEndpoint -Url $ServiceConfig.HealthEndpoint
        $status.Health = $healthCheck
    }
    
    # Determinar estado general
    $isRunning = $false
    
    if ($ServiceConfig.HealthEndpoint) {
        $isRunning = $status.Health.Success
    }
    elseif ($ServiceConfig.Port) {
        $isRunning = $status.Port.Open
    }
    elseif ($ServiceConfig.ProcessName) {
        $isRunning = $status.Process.Found
    }
    
    $status.Status = if ($isRunning) { "Running" } else { "Stopped" }
    
    return $status
}

function Show-ServiceStatus {
    param(
        [hashtable]$Status,
        [bool]$ShowDetailed = $false
    )
    
    $statusColor = if ($Status.Status -eq "Running") { "Green" } else { "Red" }
    $statusIcon = if ($Status.Status -eq "Running") { "✅" } else { "❌" }
    
    Write-ColorOutput "$statusIcon $($Status.Name): $($Status.Status)" $statusColor
    
    if ($ShowDetailed) {
        # Información del puerto
        if ($Status.Port -and $Status.Port.Number) {
            $portStatus = if ($Status.Port.Open) { "Abierto" } else { "Cerrado" }
            $portColor = if ($Status.Port.Open) { "Green" } else { "Red" }
            Write-ColorOutput "   🔌 Puerto $($Status.Port.Number): $portStatus" $portColor
        }
        
        # Información del proceso
        if ($Status.Process.Found) {
            Write-ColorOutput "   🔄 PID: $($Status.Process.PID)" "White"
            Write-ColorOutput "   💾 Memoria: $($Status.Process.Memory) MB" "White"
            if ($Status.Process.StartTime) {
                $uptime = (Get-Date) - $Status.Process.StartTime
                Write-ColorOutput "   ⏱️  Tiempo activo: $($uptime.ToString('dd\.hh\:mm\:ss'))" "White"
            }
        }
        
        # Información de salud
        if ($Status.Health.Success -ne $null) {
            if ($Status.Health.Success) {
                Write-ColorOutput "   🏥 Health Check: OK (HTTP $($Status.Health.StatusCode))" "Green"
            }
            else {
                Write-ColorOutput "   🏥 Health Check: FAIL" "Red"
                if ($Status.Health.Error) {
                    Write-ColorOutput "      Error: $($Status.Health.Error)" "Red"
                }
            }
        }
        
        Write-Host ""
    }
}

function Show-SystemSummary {
    param(
        [array]$ServiceStatuses
    )
    
    $runningCount = ($ServiceStatuses | Where-Object { $_.Status -eq "Running" }).Count
    $totalCount = $ServiceStatuses.Count
    
    Write-Host ""
    Write-ColorOutput "📊 Resumen del Sistema:" "Cyan"
    Write-ColorOutput "======================" "Cyan"
    
    $summaryColor = if ($runningCount -eq $totalCount) { "Green" } elseif ($runningCount -gt 0) { "Yellow" } else { "Red" }
    Write-ColorOutput "🔧 Servicios activos: $runningCount/$totalCount" $summaryColor
    
    # Mostrar métricas del sistema
    $cpu = Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 1
    $cpuUsage = [math]::Round(100 - $cpu.CounterSamples[0].CookedValue, 2)
    
    $memory = Get-WmiObject -Class Win32_OperatingSystem
    $totalMemory = [math]::Round($memory.TotalVisibleMemorySize / 1MB, 2)
    $freeMemory = [math]::Round($memory.FreePhysicalMemory / 1MB, 2)
    $usedMemory = $totalMemory - $freeMemory
    $memoryPercent = [math]::Round(($usedMemory / $totalMemory) * 100, 2)
    
    Write-ColorOutput "💻 CPU: $cpuUsage%" "White"
    Write-ColorOutput "🧠 Memoria: $usedMemory GB / $totalMemory GB ($memoryPercent%)" "White"
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-ColorOutput "🕒 Última verificación: $timestamp" "White"
}

function Show-JsonOutput {
    param(
        [array]$ServiceStatuses
    )
    
    $output = @{
        timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
        services = $ServiceStatuses
        summary = @{
            total = $ServiceStatuses.Count
            running = ($ServiceStatuses | Where-Object { $_.Status -eq "Running" }).Count
            stopped = ($ServiceStatuses | Where-Object { $_.Status -eq "Stopped" }).Count
        }
    }
    
    $output | ConvertTo-Json -Depth 10
}

function Send-Alert {
    param(
        [string]$ServiceName,
        [string]$Status,
        [string]$PreviousStatus
    )
    
    if ($Status -ne $PreviousStatus) {
        $alertMessage = "🚨 ALERTA: $ServiceName cambió de $PreviousStatus a $Status"
        
        if ($Status -eq "Stopped") {
            Write-ColorOutput $alertMessage "Red"
        }
        elseif ($Status -eq "Running" -and $PreviousStatus -eq "Stopped") {
            Write-ColorOutput "✅ RECUPERADO: $ServiceName está funcionando nuevamente" "Green"
        }
        
        Write-Log -Message $alertMessage -Level "ALERT"
    }
}

function Start-Monitoring {
    Write-ColorOutput "🛡️  Iniciando monitoreo de servicios AEGIS..." "Cyan"
    Write-ColorOutput "Intervalo: $Interval segundos" "White"
    Write-ColorOutput "Presiona Ctrl+C para detener" "Yellow"
    Write-Host ""
    
    $previousStatuses = @{}
    
    do {
        Clear-Host
        Write-ColorOutput "🛡️  AEGIS Framework - Monitor de Servicios" "Cyan"
        Write-ColorOutput "===========================================" "Cyan"
        Write-Host ""
        
        $serviceStatuses = @()
        
        foreach ($serviceKey in $Services.Keys) {
            $serviceConfig = $Services[$serviceKey]
            $status = Get-ServiceStatus -ServiceKey $serviceKey -ServiceConfig $serviceConfig
            $serviceStatuses += $status
            
            # Enviar alertas si está habilitado
            if ($Alerts -and $previousStatuses.ContainsKey($serviceKey)) {
                Send-Alert -ServiceName $status.Name -Status $status.Status -PreviousStatus $previousStatuses[$serviceKey]
            }
            
            $previousStatuses[$serviceKey] = $status.Status
            
            # Mostrar estado del servicio
            if ($Json) {
                # No mostrar nada aquí, se mostrará al final
            }
            else {
                Show-ServiceStatus -Status $status -ShowDetailed $Detailed
            }
            
            # Log del estado
            Write-Log -Message "$($status.Name): $($status.Status)"
        }
        
        if ($Json) {
            Show-JsonOutput -ServiceStatuses $serviceStatuses
        }
        else {
            Show-SystemSummary -ServiceStatuses $serviceStatuses
        }
        
        if ($Continuous) {
            Start-Sleep -Seconds $Interval
        }
        
    } while ($Continuous)
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Main {
    if ($Help) {
        Show-Help
    }
    
    # Verificar si estamos en el directorio correcto
    if (-not (Test-Path "main.py")) {
        Write-ColorOutput "❌ Error: No se encontró main.py. Ejecuta este script desde el directorio raíz del proyecto AEGIS." "Red"
        exit 1
    }
    
    # Configurar manejo de Ctrl+C
    $null = Register-EngineEvent PowerShell.Exiting -Action {
        Write-ColorOutput "`n🛑 Monitoreo detenido por el usuario" "Yellow"
        Write-Log -Message "Monitoreo detenido por el usuario" -Level "INFO"
    }
    
    try {
        if ($Continuous) {
            Start-Monitoring
        }
        else {
            # Verificación única
            Write-ColorOutput "🛡️  AEGIS Framework - Estado de Servicios" "Cyan"
            Write-ColorOutput "=========================================" "Cyan"
            Write-Host ""
            
            $serviceStatuses = @()
            
            foreach ($serviceKey in $Services.Keys) {
                $serviceConfig = $Services[$serviceKey]
                $status = Get-ServiceStatus -ServiceKey $serviceKey -ServiceConfig $serviceConfig
                $serviceStatuses += $status
                
                if ($Json) {
                    # No mostrar nada aquí
                }
                else {
                    Show-ServiceStatus -Status $status -ShowDetailed $Detailed
                }
                
                Write-Log -Message "$($status.Name): $($status.Status)"
            }
            
            if ($Json) {
                Show-JsonOutput -ServiceStatuses $serviceStatuses
            }
            else {
                Show-SystemSummary -ServiceStatuses $serviceStatuses
            }
        }
    }
    catch {
        Write-ColorOutput "❌ Error durante el monitoreo: $($_.Exception.Message)" "Red"
        Write-Log -Message "Error durante el monitoreo: $($_.Exception.Message)" -Level "ERROR"
        exit 1
    }
}

# Ejecutar función principal
Main