# ============================================================================
# AEGIS Framework - Verificador de Salud del Sistema para Windows
# ============================================================================
# Descripción: Script para verificar el estado de salud de todos los servicios
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

param(
    [switch]$Detailed,
    [switch]$Json,
    [switch]$Continuous,
    [int]$Interval = 30,
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
    Write-ColorOutput "🛡️  AEGIS Framework - Verificador de Salud del Sistema" "Cyan"
    Write-ColorOutput "=======================================================" "Cyan"
    Write-Host ""
    Write-ColorOutput "DESCRIPCIÓN:" "Yellow"
    Write-Host "  Verifica el estado de salud de todos los servicios AEGIS"
    Write-Host ""
    Write-ColorOutput "USO:" "Yellow"
    Write-Host "  .\health-check.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput "OPCIONES:" "Yellow"
    Write-Host "  -Detailed     Mostrar información detallada"
    Write-Host "  -Json         Salida en formato JSON"
    Write-Host "  -Continuous   Monitoreo continuo"
    Write-Host "  -Interval     Intervalo en segundos para monitoreo continuo (default: 30)"
    Write-Host "  -Help         Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput "EJEMPLOS:" "Yellow"
    Write-Host "  .\health-check.ps1"
    Write-Host "  .\health-check.ps1 -Detailed"
    Write-Host "  .\health-check.ps1 -Json"
    Write-Host "  .\health-check.ps1 -Continuous -Interval 60"
    Write-Host ""
    exit 0
}

function Test-SystemRequirements {
    $requirements = @{
        "OS" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "Memory" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "Disk" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "PowerShell" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
    }
    
    # Verificar OS
    try {
        $osInfo = Get-CimInstance -ClassName Win32_OperatingSystem
        $osVersion = $osInfo.Version
        $requirements["OS"]["Details"] = "Windows $($osInfo.Caption) - Version $osVersion"
        
        if ([Version]$osVersion -ge [Version]"10.0") {
            $requirements["OS"]["Status"] = "OK"
            $requirements["OS"]["Healthy"] = $true
        } else {
            $requirements["OS"]["Status"] = "Outdated"
        }
    }
    catch {
        $requirements["OS"]["Status"] = "Error"
        $requirements["OS"]["Details"] = $_.Exception.Message
    }
    
    # Verificar memoria
    try {
        $memory = Get-CimInstance -ClassName Win32_ComputerSystem
        $totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 2)
        $requirements["Memory"]["Details"] = "$totalMemoryGB GB"
        
        if ($totalMemoryGB -ge 4) {
            $requirements["Memory"]["Status"] = "OK"
            $requirements["Memory"]["Healthy"] = $true
        } else {
            $requirements["Memory"]["Status"] = "Insufficient"
        }
    }
    catch {
        $requirements["Memory"]["Status"] = "Error"
        $requirements["Memory"]["Details"] = $_.Exception.Message
    }
    
    # Verificar espacio en disco
    try {
        $disk = Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 -and $_.DeviceID -eq "C:" }
        $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
        $requirements["Disk"]["Details"] = "$freeSpaceGB GB free"
        
        if ($freeSpaceGB -ge 5) {
            $requirements["Disk"]["Status"] = "OK"
            $requirements["Disk"]["Healthy"] = $true
        } else {
            $requirements["Disk"]["Status"] = "Low Space"
        }
    }
    catch {
        $requirements["Disk"]["Status"] = "Error"
        $requirements["Disk"]["Details"] = $_.Exception.Message
    }
    
    # Verificar PowerShell
    try {
        $psVersion = $PSVersionTable.PSVersion
        $requirements["PowerShell"]["Details"] = "Version $psVersion"
        
        if ($psVersion.Major -ge 5) {
            $requirements["PowerShell"]["Status"] = "OK"
            $requirements["PowerShell"]["Healthy"] = $true
        } else {
            $requirements["PowerShell"]["Status"] = "Outdated"
        }
    }
    catch {
        $requirements["PowerShell"]["Status"] = "Error"
        $requirements["PowerShell"]["Details"] = $_.Exception.Message
    }
    
    return $requirements
}

function Test-Dependencies {
    $dependencies = @{
        "Python" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "Node.js" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "npm" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "Git" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "Tor" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
    }
    
    # Verificar Python
    try {
        $pythonVersion = & python --version 2>&1
        $dependencies["Python"]["Details"] = $pythonVersion
        
        if ($pythonVersion -match "Python 3\.([8-9]|1[0-9])") {
            $dependencies["Python"]["Status"] = "OK"
            $dependencies["Python"]["Healthy"] = $true
        } else {
            $dependencies["Python"]["Status"] = "Incompatible Version"
        }
    }
    catch {
        $dependencies["Python"]["Status"] = "Not Found"
        $dependencies["Python"]["Details"] = "Python no está instalado o no está en PATH"
    }
    
    # Verificar Node.js
    try {
        $nodeVersion = & node --version 2>&1
        $dependencies["Node.js"]["Details"] = $nodeVersion
        
        if ($nodeVersion -match "v(1[8-9]|2[0-9])") {
            $dependencies["Node.js"]["Status"] = "OK"
            $dependencies["Node.js"]["Healthy"] = $true
        } else {
            $dependencies["Node.js"]["Status"] = "Incompatible Version"
        }
    }
    catch {
        $dependencies["Node.js"]["Status"] = "Not Found"
        $dependencies["Node.js"]["Details"] = "Node.js no está instalado o no está en PATH"
    }
    
    # Verificar npm
    try {
        $npmVersion = & npm --version 2>&1
        $dependencies["npm"]["Details"] = "v$npmVersion"
        $dependencies["npm"]["Status"] = "OK"
        $dependencies["npm"]["Healthy"] = $true
    }
    catch {
        $dependencies["npm"]["Status"] = "Not Found"
        $dependencies["npm"]["Details"] = "npm no está instalado o no está en PATH"
    }
    
    # Verificar Git
    try {
        $gitVersion = & git --version 2>&1
        $dependencies["Git"]["Details"] = $gitVersion
        $dependencies["Git"]["Status"] = "OK"
        $dependencies["Git"]["Healthy"] = $true
    }
    catch {
        $dependencies["Git"]["Status"] = "Not Found"
        $dependencies["Git"]["Details"] = "Git no está instalado o no está en PATH"
    }
    
    # Verificar Tor
    try {
        $torVersion = & tor --version 2>&1 | Select-Object -First 1
        $dependencies["Tor"]["Details"] = $torVersion
        $dependencies["Tor"]["Status"] = "OK"
        $dependencies["Tor"]["Healthy"] = $true
    }
    catch {
        $dependencies["Tor"]["Status"] = "Not Found"
        $dependencies["Tor"]["Details"] = "Tor no está instalado o no está en PATH"
    }
    
    return $dependencies
}

function Test-ProjectStructure {
    $structure = @{
        "ConfigFiles" = @{
            "Status" = "Unknown"
            "Details" = @()
            "Healthy" = $false
        }
        "Directories" = @{
            "Status" = "Unknown"
            "Details" = @()
            "Healthy" = $false
        }
        "VirtualEnv" = @{
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "NodeModules" = @{
            "Status" = "Unknown"
            "Details" = @()
            "Healthy" = $false
        }
    }
    
    # Verificar archivos de configuración
    $configFiles = @(".env", "config\app_config.json", "config\torrc")
    $missingConfigs = @()
    $existingConfigs = @()
    
    foreach ($file in $configFiles) {
        if (Test-Path $file) {
            $existingConfigs += $file
        } else {
            $missingConfigs += $file
        }
    }
    
    $structure["ConfigFiles"]["Details"] = @{
        "Existing" = $existingConfigs
        "Missing" = $missingConfigs
    }
    
    if ($missingConfigs.Count -eq 0) {
        $structure["ConfigFiles"]["Status"] = "OK"
        $structure["ConfigFiles"]["Healthy"] = $true
    } else {
        $structure["ConfigFiles"]["Status"] = "Missing Files"
    }
    
    # Verificar directorios
    $requiredDirs = @("config", "logs", "tor_data", "dapps\secure-chat\ui", "dapps\aegis-token")
    $missingDirs = @()
    $existingDirs = @()
    
    foreach ($dir in $requiredDirs) {
        if (Test-Path $dir) {
            $existingDirs += $dir
        } else {
            $missingDirs += $dir
        }
    }
    
    $structure["Directories"]["Details"] = @{
        "Existing" = $existingDirs
        "Missing" = $missingDirs
    }
    
    if ($missingDirs.Count -eq 0) {
        $structure["Directories"]["Status"] = "OK"
        $structure["Directories"]["Healthy"] = $true
    } else {
        $structure["Directories"]["Status"] = "Missing Directories"
    }
    
    # Verificar entorno virtual de Python
    if (Test-Path "venv\Scripts\activate.ps1") {
        $structure["VirtualEnv"]["Status"] = "OK"
        $structure["VirtualEnv"]["Details"] = "Virtual environment found"
        $structure["VirtualEnv"]["Healthy"] = $true
    } else {
        $structure["VirtualEnv"]["Status"] = "Not Found"
        $structure["VirtualEnv"]["Details"] = "Python virtual environment not found"
    }
    
    # Verificar node_modules
    $nodeModulesPaths = @("dapps\secure-chat\ui\node_modules", "dapps\aegis-token\node_modules")
    $missingNodeModules = @()
    $existingNodeModules = @()
    
    foreach ($path in $nodeModulesPaths) {
        if (Test-Path $path) {
            $existingNodeModules += $path
        } else {
            $missingNodeModules += $path
        }
    }
    
    $structure["NodeModules"]["Details"] = @{
        "Existing" = $existingNodeModules
        "Missing" = $missingNodeModules
    }
    
    if ($missingNodeModules.Count -eq 0) {
        $structure["NodeModules"]["Status"] = "OK"
        $structure["NodeModules"]["Healthy"] = $true
    } else {
        $structure["NodeModules"]["Status"] = "Missing Dependencies"
    }
    
    return $structure
}

function Test-NetworkPorts {
    $ports = @{
        "Dashboard" = @{
            "Port" = 8080
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "SecureChat" = @{
            "Port" = 5173
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "Blockchain" = @{
            "Port" = 8545
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "TorSOCKS" = @{
            "Port" = 9050
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "TorControl" = @{
            "Port" = 9051
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
    }
    
    foreach ($service in $ports.Keys) {
        $port = $ports[$service]["Port"]
        
        try {
            $connection = Test-NetConnection -ComputerName "localhost" -Port $port -InformationLevel Quiet -WarningAction SilentlyContinue
            
            if ($connection) {
                $ports[$service]["Status"] = "Open"
                $ports[$service]["Details"] = "Service is running"
                $ports[$service]["Healthy"] = $true
            } else {
                $ports[$service]["Status"] = "Closed"
                $ports[$service]["Details"] = "Service is not running"
            }
        }
        catch {
            $ports[$service]["Status"] = "Error"
            $ports[$service]["Details"] = $_.Exception.Message
        }
    }
    
    return $ports
}

function Test-ServiceHealth {
    $services = @{
        "Dashboard" = @{
            "URL" = "http://localhost:8080"
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "SecureChat" = @{
            "URL" = "http://localhost:5173"
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
        "Blockchain" = @{
            "URL" = "http://localhost:8545"
            "Status" = "Unknown"
            "Details" = ""
            "Healthy" = $false
        }
    }
    
    foreach ($service in $services.Keys) {
        $url = $services[$service]["URL"]
        
        try {
            $response = Invoke-WebRequest -Uri $url -Method GET -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
            
            if ($response.StatusCode -eq 200) {
                $services[$service]["Status"] = "Healthy"
                $services[$service]["Details"] = "HTTP 200 OK"
                $services[$service]["Healthy"] = $true
            } else {
                $services[$service]["Status"] = "Unhealthy"
                $services[$service]["Details"] = "HTTP $($response.StatusCode)"
            }
        }
        catch {
            $services[$service]["Status"] = "Unreachable"
            $services[$service]["Details"] = $_.Exception.Message
        }
    }
    
    return $services
}

function Get-SystemMetrics {
    $metrics = @{
        "CPU" = @{
            "Usage" = 0
            "Details" = ""
        }
        "Memory" = @{
            "Usage" = 0
            "Details" = ""
        }
        "Disk" = @{
            "Usage" = 0
            "Details" = ""
        }
        "Network" = @{
            "Status" = "Unknown"
            "Details" = ""
        }
    }
    
    try {
        # CPU Usage
        $cpu = Get-CimInstance -ClassName Win32_Processor | Measure-Object -Property LoadPercentage -Average
        $metrics["CPU"]["Usage"] = [math]::Round($cpu.Average, 2)
        $metrics["CPU"]["Details"] = "$($metrics["CPU"]["Usage"])% average load"
        
        # Memory Usage
        $memory = Get-CimInstance -ClassName Win32_OperatingSystem
        $totalMemory = $memory.TotalVisibleMemorySize
        $freeMemory = $memory.FreePhysicalMemory
        $usedMemory = $totalMemory - $freeMemory
        $memoryUsagePercent = [math]::Round(($usedMemory / $totalMemory) * 100, 2)
        
        $metrics["Memory"]["Usage"] = $memoryUsagePercent
        $metrics["Memory"]["Details"] = "$memoryUsagePercent% used ($([math]::Round($usedMemory/1MB, 2)) GB / $([math]::Round($totalMemory/1MB, 2)) GB)"
        
        # Disk Usage
        $disk = Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 -and $_.DeviceID -eq "C:" }
        $diskUsagePercent = [math]::Round((($disk.Size - $disk.FreeSpace) / $disk.Size) * 100, 2)
        
        $metrics["Disk"]["Usage"] = $diskUsagePercent
        $metrics["Disk"]["Details"] = "$diskUsagePercent% used ($([math]::Round(($disk.Size - $disk.FreeSpace)/1GB, 2)) GB / $([math]::Round($disk.Size/1GB, 2)) GB)"
        
        # Network connectivity
        $networkTest = Test-NetConnection -ComputerName "8.8.8.8" -Port 53 -InformationLevel Quiet -WarningAction SilentlyContinue
        if ($networkTest) {
            $metrics["Network"]["Status"] = "Connected"
            $metrics["Network"]["Details"] = "Internet connectivity available"
        } else {
            $metrics["Network"]["Status"] = "Disconnected"
            $metrics["Network"]["Details"] = "No internet connectivity"
        }
    }
    catch {
        Write-ColorOutput "Error getting system metrics: $($_.Exception.Message)" "Red"
    }
    
    return $metrics
}

function Show-HealthReport {
    param(
        [hashtable]$HealthData,
        [switch]$Detailed
    )
    
    if ($Json) {
        $HealthData | ConvertTo-Json -Depth 10
        return
    }
    
    Write-ColorOutput "🛡️  AEGIS Framework - Reporte de Salud del Sistema" "Cyan"
    Write-ColorOutput "=================================================" "Cyan"
    Write-Host ""
    
    # Resumen general
    $totalChecks = 0
    $healthyChecks = 0
    
    foreach ($category in $HealthData.Keys) {
        foreach ($item in $HealthData[$category].Keys) {
            $totalChecks++
            if ($HealthData[$category][$item]["Healthy"]) {
                $healthyChecks++
            }
        }
    }
    
    $healthPercentage = [math]::Round(($healthyChecks / $totalChecks) * 100, 1)
    
    Write-ColorOutput "📊 RESUMEN GENERAL" "Yellow"
    Write-Host "Verificaciones saludables: $healthyChecks/$totalChecks ($healthPercentage%)"
    Write-Host ""
    
    # Mostrar cada categoría
    foreach ($category in $HealthData.Keys) {
        Write-ColorOutput "🔍 $category" "Blue"
        Write-ColorOutput ("=" * ($category.Length + 3)) "Blue"
        
        foreach ($item in $HealthData[$category].Keys) {
            $status = $HealthData[$category][$item]["Status"]
            $details = $HealthData[$category][$item]["Details"]
            $healthy = $HealthData[$category][$item]["Healthy"]
            
            $statusColor = if ($healthy) { "Green" } else { "Red" }
            $statusIcon = if ($healthy) { "✅" } else { "❌" }
            
            Write-Host "$statusIcon $item`: " -NoNewline
            Write-ColorOutput $status $statusColor
            
            if ($Detailed -and $details) {
                if ($details -is [hashtable]) {
                    foreach ($key in $details.Keys) {
                        Write-Host "   $key`: $($details[$key] -join ', ')"
                    }
                } elseif ($details -is [array]) {
                    Write-Host "   $($details -join ', ')"
                } else {
                    Write-Host "   $details"
                }
            }
        }
        Write-Host ""
    }
    
    # Mostrar métricas del sistema si están disponibles
    if ($HealthData.ContainsKey("Metrics")) {
        Write-ColorOutput "📈 MÉTRICAS DEL SISTEMA" "Magenta"
        Write-ColorOutput "======================" "Magenta"
        
        foreach ($metric in $HealthData["Metrics"].Keys) {
            $usage = $HealthData["Metrics"][$metric]["Usage"]
            $details = $HealthData["Metrics"][$metric]["Details"]
            
            $color = "Green"
            if ($usage -gt 80) { $color = "Red" }
            elseif ($usage -gt 60) { $color = "Yellow" }
            
            Write-Host "$metric`: " -NoNewline
            Write-ColorOutput $details $color
        }
        Write-Host ""
    }
    
    # Recomendaciones
    if ($healthPercentage -lt 100) {
        Write-ColorOutput "💡 RECOMENDACIONES" "Yellow"
        Write-ColorOutput "==================" "Yellow"
        
        foreach ($category in $HealthData.Keys) {
            foreach ($item in $HealthData[$category].Keys) {
                if (-not $HealthData[$category][$item]["Healthy"]) {
                    $status = $HealthData[$category][$item]["Status"]
                    
                    switch ($status) {
                        "Not Found" {
                            Write-Host "• Instala $item usando el script de dependencias"
                        }
                        "Incompatible Version" {
                            Write-Host "• Actualiza $item a una versión compatible"
                        }
                        "Missing Files" {
                            Write-Host "• Ejecuta el script de configuración para crear archivos faltantes"
                        }
                        "Missing Directories" {
                            Write-Host "• Crea los directorios faltantes o ejecuta el script de configuración"
                        }
                        "Closed" {
                            Write-Host "• Inicia el servicio $item"
                        }
                        "Unreachable" {
                            Write-Host "• Verifica que el servicio $item esté ejecutándose correctamente"
                        }
                    }
                }
            }
        }
        Write-Host ""
    }
    
    # Estado general
    if ($healthPercentage -eq 100) {
        Write-ColorOutput "🎉 ¡Todos los sistemas están funcionando correctamente!" "Green"
    } elseif ($healthPercentage -ge 80) {
        Write-ColorOutput "⚠️  La mayoría de los sistemas están funcionando, pero hay algunos problemas menores" "Yellow"
    } else {
        Write-ColorOutput "🚨 Se detectaron problemas significativos que requieren atención" "Red"
    }
    
    Write-Host ""
    Write-ColorOutput "🕒 Última verificación: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "Cyan"
}

function Start-ContinuousMonitoring {
    param(
        [int]$Interval
    )
    
    Write-ColorOutput "🔄 Iniciando monitoreo continuo (intervalo: $Interval segundos)" "Blue"
    Write-ColorOutput "Presiona Ctrl+C para detener" "Yellow"
    Write-Host ""
    
    try {
        while ($true) {
            Clear-Host
            
            # Recopilar datos de salud
            $healthData = @{
                "SystemRequirements" = Test-SystemRequirements
                "Dependencies" = Test-Dependencies
                "ProjectStructure" = Test-ProjectStructure
                "NetworkPorts" = Test-NetworkPorts
                "ServiceHealth" = Test-ServiceHealth
                "Metrics" = Get-SystemMetrics
            }
            
            # Mostrar reporte
            Show-HealthReport -HealthData $healthData -Detailed:$Detailed
            
            # Esperar intervalo
            Start-Sleep -Seconds $Interval
        }
    }
    catch {
        Write-ColorOutput "Monitoreo detenido" "Yellow"
    }
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Main {
    if ($Help) {
        Show-Help
    }
    
    if ($Continuous) {
        Start-ContinuousMonitoring -Interval $Interval
        return
    }
    
    Write-ColorOutput "🔍 Recopilando información del sistema..." "Blue"
    
    # Recopilar todos los datos de salud
    $healthData = @{
        "SystemRequirements" = Test-SystemRequirements
        "Dependencies" = Test-Dependencies
        "ProjectStructure" = Test-ProjectStructure
        "NetworkPorts" = Test-NetworkPorts
        "ServiceHealth" = Test-ServiceHealth
    }
    
    if ($Detailed) {
        $healthData["Metrics"] = Get-SystemMetrics
    }
    
    # Mostrar reporte
    Show-HealthReport -HealthData $healthData -Detailed:$Detailed
}

# Ejecutar función principal
Main