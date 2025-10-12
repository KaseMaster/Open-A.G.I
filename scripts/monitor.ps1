# AEGIS Framework - Monitoring Script (PowerShell)
# Advanced Encrypted Governance and Intelligence System
# This script provides comprehensive monitoring and alerting for Windows

param(
    [switch]$Daemon,
    [switch]$Verbose,
    [int]$Interval = 30,
    [switch]$NoServices,
    [switch]$NoResources,
    [switch]$NoNetwork,
    [switch]$NoSecurity,
    [switch]$NoAlerts,
    [switch]$Help
)

# Configuration
$AegisDir = Split-Path -Parent $PSScriptRoot
$MonitorInterval = $Interval
$LogFile = Join-Path $AegisDir "logs\monitor.log"
$AlertThresholdCPU = 80
$AlertThresholdMemory = 85
$AlertThresholdDisk = 90
$AlertThresholdResponseTime = 5000
$MaxLogSize = 100MB
$RetentionDays = 30

# Default values
$CheckServices = -not $NoServices
$CheckResources = -not $NoResources
$CheckNetwork = -not $NoNetwork
$CheckSecurity = -not $NoSecurity
$SendAlerts = -not $NoAlerts

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    Magenta = "Magenta"
    Cyan = "Cyan"
}

# Logging functions
function Write-LogInfo {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
    Add-Content -Path $LogFile -Value "[$Timestamp] [INFO] $Message"
}

function Write-LogSuccess {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
    Add-Content -Path $LogFile -Value "[$Timestamp] [SUCCESS] $Message"
}

function Write-LogWarning {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
    Add-Content -Path $LogFile -Value "[$Timestamp] [WARNING] $Message"
}

function Write-LogError {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
    Add-Content -Path $LogFile -Value "[$Timestamp] [ERROR] $Message"
}

function Write-LogHeader {
    param([string]$Message)
    Write-Host "[AEGIS MONITOR] $Message" -ForegroundColor $Colors.Magenta
}

# Check if command exists
function Test-CommandExists {
    param([string]$Command)
    return [bool](Get-Command $Command -ErrorAction SilentlyContinue)
}

# Show help
function Show-Help {
    @"
AEGIS Framework Monitoring Script (PowerShell)

Usage: .\monitor.ps1 [OPTIONS]

OPTIONS:
    -Daemon             Run in daemon mode (continuous monitoring)
    -Verbose            Enable verbose output
    -Interval SECONDS   Set monitoring interval (default: 30)
    -NoServices         Skip service health checks
    -NoResources        Skip resource monitoring
    -NoNetwork          Skip network monitoring
    -NoSecurity         Skip security monitoring
    -NoAlerts           Disable alert notifications
    -Help               Show this help message

MONITORING CATEGORIES:
    Services    - Docker containers, processes, health endpoints
    Resources   - CPU, memory, disk usage
    Network     - Connectivity, latency, port availability
    Security    - Failed logins, suspicious activity, certificate expiry

EXAMPLES:
    .\monitor.ps1                    # Run single monitoring check
    .\monitor.ps1 -Daemon            # Run continuous monitoring
    .\monitor.ps1 -Daemon -Interval 60  # Monitor every 60 seconds
    .\monitor.ps1 -NoAlerts          # Monitor without sending alerts

"@
}

# Initialize monitoring
function Initialize-Monitoring {
    Write-LogHeader "Initializing AEGIS monitoring..."
    
    # Create logs directory
    $LogDir = Split-Path -Parent $LogFile
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    
    # Load environment variables
    $EnvFile = Join-Path $AegisDir ".env"
    if (Test-Path $EnvFile) {
        Get-Content $EnvFile | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
    }
    
    # Check required tools
    $RequiredTools = @("docker", "curl")
    foreach ($Tool in $RequiredTools) {
        if (-not (Test-CommandExists $Tool)) {
            Write-LogWarning "$Tool not found, some checks may be skipped"
        }
    }
    
    Write-LogSuccess "Monitoring initialized"
}

# Check Docker services
function Test-DockerServices {
    if (-not $CheckServices) { return }
    
    Write-LogInfo "Checking Docker services..."
    
    if (-not (Test-CommandExists "docker")) {
        Write-LogWarning "Docker not available, skipping container checks"
        return
    }
    
    # Check if Docker daemon is running
    try {
        docker info | Out-Null
    }
    catch {
        Write-LogError "Docker daemon is not running"
        Send-Alert "Docker daemon is not running"
        return
    }
    
    # Check AEGIS containers
    $Containers = @(
        "aegis-core",
        "aegis-redis",
        "aegis-postgres",
        "aegis-prometheus",
        "aegis-grafana",
        "aegis-nginx"
    )
    
    foreach ($Container in $Containers) {
        try {
            $Status = docker ps --filter "name=$Container" --format "{{.Status}}" 2>$null
            
            if ([string]::IsNullOrEmpty($Status)) {
                Write-LogWarning "Container $Container is not running"
                if ($SendAlerts) {
                    Send-Alert "Container $Container is not running"
                }
            }
            elseif ($Status -match "^Up") {
                Write-LogSuccess "Container $Container is healthy: $Status"
            }
            else {
                Write-LogError "Container $Container has issues: $Status"
                if ($SendAlerts) {
                    Send-Alert "Container $Container has issues: $Status"
                }
            }
        }
        catch {
            Write-LogError "Failed to check container $Container"
        }
    }
}

# Check health endpoints
function Test-HealthEndpoints {
    if (-not $CheckServices) { return }
    
    Write-LogInfo "Checking health endpoints..."
    
    $Endpoints = @(
        @{ Url = "http://localhost:8080/health"; Service = "AEGIS Core" },
        @{ Url = "http://localhost:9090/api/v1/query?query=up"; Service = "Prometheus" },
        @{ Url = "http://localhost:3000/api/health"; Service = "Grafana" }
    )
    
    foreach ($Endpoint in $Endpoints) {
        try {
            $StartTime = Get-Date
            $Response = Invoke-WebRequest -Uri $Endpoint.Url -TimeoutSec 10 -UseBasicParsing
            $EndTime = Get-Date
            $ResponseTime = ($EndTime - $StartTime).TotalMilliseconds
            
            if ($Response.StatusCode -eq 200) {
                Write-LogSuccess "$($Endpoint.Service) health check passed ($([int]$ResponseTime)ms)"
                
                if ($ResponseTime -gt $AlertThresholdResponseTime) {
                    Write-LogWarning "$($Endpoint.Service) response time is high: $([int]$ResponseTime)ms"
                    if ($SendAlerts) {
                        Send-Alert "$($Endpoint.Service) response time is high: $([int]$ResponseTime)ms"
                    }
                }
            }
            else {
                Write-LogError "$($Endpoint.Service) health check failed (HTTP $($Response.StatusCode))"
                if ($SendAlerts) {
                    Send-Alert "$($Endpoint.Service) health check failed (HTTP $($Response.StatusCode))"
                }
            }
        }
        catch {
            Write-LogError "$($Endpoint.Service) health check failed: $($_.Exception.Message)"
            if ($SendAlerts) {
                Send-Alert "$($Endpoint.Service) health check failed"
            }
        }
    }
}

# Check system resources
function Test-SystemResources {
    if (-not $CheckResources) { return }
    
    Write-LogInfo "Checking system resources..."
    
    # Check CPU usage
    try {
        $CpuUsage = (Get-WmiObject -Class Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        Write-LogInfo "CPU usage: $([int]$CpuUsage)%"
        
        if ($CpuUsage -gt $AlertThresholdCPU) {
            Write-LogWarning "High CPU usage: $([int]$CpuUsage)%"
            if ($SendAlerts) {
                Send-Alert "High CPU usage: $([int]$CpuUsage)%"
            }
        }
    }
    catch {
        Write-LogWarning "Failed to get CPU usage"
    }
    
    # Check memory usage
    try {
        $Memory = Get-WmiObject -Class Win32_OperatingSystem
        $TotalMemory = [math]::Round($Memory.TotalVisibleMemorySize / 1MB, 2)
        $FreeMemory = [math]::Round($Memory.FreePhysicalMemory / 1MB, 2)
        $UsedMemory = $TotalMemory - $FreeMemory
        $MemoryUsage = [math]::Round(($UsedMemory / $TotalMemory) * 100, 2)
        
        Write-LogInfo "Memory usage: $MemoryUsage% ($([math]::Round($UsedMemory, 1))GB / $([math]::Round($TotalMemory, 1))GB)"
        
        if ($MemoryUsage -gt $AlertThresholdMemory) {
            Write-LogWarning "High memory usage: $MemoryUsage%"
            if ($SendAlerts) {
                Send-Alert "High memory usage: $MemoryUsage%"
            }
        }
    }
    catch {
        Write-LogWarning "Failed to get memory usage"
    }
    
    # Check disk usage
    try {
        Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 } | ForEach-Object {
            $DiskUsage = [math]::Round((($_.Size - $_.FreeSpace) / $_.Size) * 100, 2)
            Write-LogInfo "Disk usage $($_.DeviceID): $DiskUsage%"
            
            if ($DiskUsage -gt $AlertThresholdDisk) {
                Write-LogWarning "High disk usage on $($_.DeviceID): $DiskUsage%"
                if ($SendAlerts) {
                    Send-Alert "High disk usage on $($_.DeviceID): $DiskUsage%"
                }
            }
        }
    }
    catch {
        Write-LogWarning "Failed to get disk usage"
    }
}

# Check network connectivity
function Test-NetworkConnectivity {
    if (-not $CheckNetwork) { return }
    
    Write-LogInfo "Checking network connectivity..."
    
    # Check internet connectivity
    try {
        $PingResult = Test-NetConnection -ComputerName "8.8.8.8" -InformationLevel Quiet
        if ($PingResult) {
            Write-LogSuccess "Internet connectivity: OK"
        }
        else {
            Write-LogError "Internet connectivity: FAILED"
            if ($SendAlerts) {
                Send-Alert "Internet connectivity failed"
            }
        }
    }
    catch {
        Write-LogError "Internet connectivity: FAILED"
        if ($SendAlerts) {
            Send-Alert "Internet connectivity failed"
        }
    }
    
    # Check DNS resolution
    try {
        $DnsResult = Resolve-DnsName -Name "google.com" -ErrorAction Stop
        Write-LogSuccess "DNS resolution: OK"
    }
    catch {
        Write-LogError "DNS resolution: FAILED"
        if ($SendAlerts) {
            Send-Alert "DNS resolution failed"
        }
    }
    
    # Check critical ports
    $Ports = @(
        @{ Port = 8080; Service = "AEGIS Core" },
        @{ Port = 5432; Service = "PostgreSQL" },
        @{ Port = 6379; Service = "Redis" },
        @{ Port = 9090; Service = "Prometheus" },
        @{ Port = 3000; Service = "Grafana" }
    )
    
    foreach ($PortInfo in $Ports) {
        try {
            $Connection = Test-NetConnection -ComputerName "localhost" -Port $PortInfo.Port -InformationLevel Quiet
            if ($Connection) {
                Write-LogSuccess "Port $($PortInfo.Port) ($($PortInfo.Service)): LISTENING"
            }
            else {
                Write-LogWarning "Port $($PortInfo.Port) ($($PortInfo.Service)): NOT LISTENING"
                if ($SendAlerts) {
                    Send-Alert "Port $($PortInfo.Port) ($($PortInfo.Service)) is not listening"
                }
            }
        }
        catch {
            Write-LogWarning "Failed to check port $($PortInfo.Port)"
        }
    }
}

# Check security metrics
function Test-SecurityMetrics {
    if (-not $CheckSecurity) { return }
    
    Write-LogInfo "Checking security metrics..."
    
    # Check Windows Event Log for failed logins
    try {
        $FailedLogins = Get-WinEvent -FilterHashtable @{LogName='Security'; ID=4625; StartTime=(Get-Date).AddDays(-1)} -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count
        Write-LogInfo "Failed login attempts (last 24h): $FailedLogins"
        
        if ($FailedLogins -gt 10) {
            Write-LogWarning "High number of failed login attempts: $FailedLogins"
            if ($SendAlerts) {
                Send-Alert "High number of failed login attempts: $FailedLogins"
            }
        }
    }
    catch {
        Write-LogWarning "Failed to check Windows Event Log"
    }
    
    # Check SSL certificate expiry
    $CertPath = Join-Path $AegisDir "certs\server.crt"
    if (Test-Path $CertPath) {
        try {
            $Cert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($CertPath)
            $DaysUntilExpiry = ($Cert.NotAfter - (Get-Date)).Days
            
            Write-LogInfo "SSL certificate expires in $DaysUntilExpiry days"
            
            if ($DaysUntilExpiry -lt 30) {
                Write-LogWarning "SSL certificate expires soon: $DaysUntilExpiry days"
                if ($SendAlerts) {
                    Send-Alert "SSL certificate expires in $DaysUntilExpiry days"
                }
            }
        }
        catch {
            Write-LogWarning "Failed to check SSL certificate"
        }
    }
    
    # Check for suspicious processes
    try {
        $SuspiciousProcesses = Get-Process | Where-Object { $_.ProcessName -match "(nc|netcat|nmap|tcpdump)" } | Measure-Object | Select-Object -ExpandProperty Count
        if ($SuspiciousProcesses -gt 0) {
            Write-LogWarning "Suspicious network processes detected: $SuspiciousProcesses"
            if ($SendAlerts) {
                Send-Alert "Suspicious network processes detected"
            }
        }
    }
    catch {
        Write-LogWarning "Failed to check for suspicious processes"
    }
}

# Check log files
function Test-LogFiles {
    Write-LogInfo "Checking log files..."
    
    # Check log file sizes
    $LogDirs = @(
        (Join-Path $AegisDir "logs"),
        "C:\Windows\Logs"
    )
    
    foreach ($LogDir in $LogDirs) {
        if (Test-Path $LogDir) {
            Get-ChildItem -Path $LogDir -Filter "*.log" -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
                $SizeMB = [math]::Round($_.Length / 1MB, 2)
                
                if ($SizeMB -gt 100) {
                    Write-LogWarning "Large log file: $($_.FullName) ($SizeMB MB)"
                }
            }
        }
    }
    
    # Check for error patterns in AEGIS logs
    $AegisLogPath = Join-Path $AegisDir "logs\aegis.log"
    if (Test-Path $AegisLogPath) {
        try {
            $ErrorCount = (Select-String -Path $AegisLogPath -Pattern "ERROR|CRITICAL" -ErrorAction SilentlyContinue | Measure-Object).Count
            Write-LogInfo "Errors in AEGIS log: $ErrorCount"
            
            if ($ErrorCount -gt 10) {
                Write-LogWarning "High error count in AEGIS log: $ErrorCount"
                if ($SendAlerts) {
                    Send-Alert "High error count in AEGIS log: $ErrorCount"
                }
            }
        }
        catch {
            Write-LogWarning "Failed to analyze AEGIS log file"
        }
    }
}

# Send alert notification
function Send-Alert {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    # Send to Slack if webhook is configured
    $SlackWebhook = $env:SLACK_WEBHOOK_URL
    if ($SlackWebhook) {
        try {
            $Body = @{
                text = "🚨 AEGIS Alert [$Timestamp]: $Message"
            } | ConvertTo-Json
            
            Invoke-RestMethod -Uri $SlackWebhook -Method Post -Body $Body -ContentType "application/json" | Out-Null
        }
        catch {
            Write-LogWarning "Failed to send Slack alert"
        }
    }
    
    # Send email if configured (requires Send-MailMessage or similar)
    $AlertEmail = $env:ALERT_EMAIL
    if ($AlertEmail -and (Test-CommandExists "Send-MailMessage")) {
        try {
            Send-MailMessage -To $AlertEmail -Subject "AEGIS Alert" -Body "AEGIS Alert [$Timestamp]: $Message" -SmtpServer $env:SMTP_SERVER
        }
        catch {
            Write-LogWarning "Failed to send email alert"
        }
    }
    
    # Write to alert log
    $AlertLogPath = Join-Path $AegisDir "logs\alerts.log"
    Add-Content -Path $AlertLogPath -Value "[$Timestamp] ALERT: $Message"
}

# Generate monitoring report
function New-MonitoringReport {
    $ReportFile = Join-Path $AegisDir "logs\monitor_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    
    $SystemInfo = @{
        hostname = $env:COMPUTERNAME
        uptime = (Get-CimInstance -ClassName Win32_OperatingSystem).LastBootUpTime
        load_average = "N/A (Windows)"
        disk_usage = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 } | Select-Object DeviceID, Size, FreeSpace
        memory_usage = Get-WmiObject -Class Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory
    }
    
    $Report = @{
        timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        monitoring_interval = $MonitorInterval
        checks_performed = @{
            services = $CheckServices
            resources = $CheckResources
            network = $CheckNetwork
            security = $CheckSecurity
        }
        system_info = $SystemInfo
    }
    
    $Report | ConvertTo-Json -Depth 10 | Out-File -FilePath $ReportFile -Encoding UTF8
    Write-LogInfo "Monitoring report generated: $ReportFile"
}

# Cleanup old logs
function Clear-OldLogs {
    Write-LogInfo "Cleaning up old logs..."
    
    # Remove logs older than retention period
    $LogsDir = Join-Path $AegisDir "logs"
    if (Test-Path $LogsDir) {
        $CutoffDate = (Get-Date).AddDays(-$RetentionDays)
        
        Get-ChildItem -Path $LogsDir -Filter "*.log" -Recurse | Where-Object { $_.LastWriteTime -lt $CutoffDate } | Remove-Item -Force
        Get-ChildItem -Path $LogsDir -Filter "monitor_report_*.json" | Where-Object { $_.LastWriteTime -lt $CutoffDate } | Remove-Item -Force
    }
    
    # Rotate large log files
    if (Test-Path $LogFile) {
        $LogFileInfo = Get-Item $LogFile
        if ($LogFileInfo.Length -gt $MaxLogSize) {
            $BackupName = "$LogFile.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
            Move-Item -Path $LogFile -Destination $BackupName
            New-Item -ItemType File -Path $LogFile | Out-Null
            Write-LogInfo "Log file rotated due to size"
        }
    }
}

# Run single monitoring cycle
function Start-MonitoringCycle {
    Write-LogHeader "Running monitoring cycle..."
    
    Test-DockerServices
    Test-HealthEndpoints
    Test-SystemResources
    Test-NetworkConnectivity
    Test-SecurityMetrics
    Test-LogFiles
    
    if ($Verbose) {
        New-MonitoringReport
    }
    
    Clear-OldLogs
    
    Write-LogSuccess "Monitoring cycle completed"
}

# Main function
function Main {
    if ($Help) {
        Show-Help
        return
    }
    
    # Initialize
    Initialize-Monitoring
    
    if ($Daemon) {
        Write-LogHeader "Starting AEGIS monitoring daemon (interval: $MonitorInterval seconds)"
        
        try {
            while ($true) {
                Start-MonitoringCycle
                Start-Sleep -Seconds $MonitorInterval
            }
        }
        catch {
            Write-LogError "Monitoring daemon stopped: $($_.Exception.Message)"
        }
        finally {
            Write-LogInfo "Stopping monitoring daemon..."
        }
    }
    else {
        Start-MonitoringCycle
    }
}

# Run main function
Main