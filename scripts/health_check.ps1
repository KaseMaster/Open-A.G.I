# AEGIS Framework - Health Check Script (PowerShell)
# Comprehensive system health verification for Windows

param(
    [switch]$Verbose,
    [switch]$Json,
    [int]$Timeout = 30,
    [switch]$Help
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogFile = Join-Path $ProjectRoot "logs\health_check.log"
$AlertThreshold = 80

# Health check results
$HealthStatus = "healthy"
$Issues = @()
$Warnings = @()

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Logging function
function Write-Log {
    param(
        [string]$Level,
        [string]$Message
    )
    
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    
    # Ensure log directory exists
    $LogDir = Split-Path -Parent $LogFile
    if (!(Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    
    Add-Content -Path $LogFile -Value $LogEntry
    
    if ($Verbose) {
        switch ($Level) {
            "ERROR" { Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red }
            "WARN" { Write-Host "[WARN] $Message" -ForegroundColor $Colors.Yellow }
            "INFO" { Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue }
            "SUCCESS" { Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green }
        }
    }
}

# Check if command exists
function Test-Command {
    param([string]$Command)
    
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check Docker services
function Test-DockerServices {
    Write-Log "INFO" "Checking Docker services..."
    
    if (!(Test-Command "docker")) {
        $script:Issues += "Docker not installed"
        $script:HealthStatus = "unhealthy"
        return $false
    }
    
    try {
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            $script:Issues += "Docker daemon not running"
            $script:HealthStatus = "unhealthy"
            return $false
        }
    }
    catch {
        $script:Issues += "Docker daemon not accessible"
        $script:HealthStatus = "unhealthy"
        return $false
    }
    
    # Check if docker-compose is available
    $ComposeCmd = $null
    if (Test-Command "docker-compose") {
        $ComposeCmd = "docker-compose"
    }
    elseif ((docker compose version 2>$null) -and ($LASTEXITCODE -eq 0)) {
        $ComposeCmd = "docker compose"
    }
    else {
        $script:Issues += "Docker Compose not available"
        $script:HealthStatus = "unhealthy"
        return $false
    }
    
    # Check running containers
    try {
        $Containers = docker ps --format "table {{.Names}}`t{{.Status}}" | Where-Object { $_ -match "(aegis|redis|postgres|prometheus|grafana|nginx)" }
        
        if (!$Containers) {
            $script:Warnings += "No AEGIS containers running"
            return $true
        }
        
        # Check container health
        foreach ($Container in $Containers) {
            if ($Container -match "NAMES") { continue }
            
            $Parts = $Container -split "`t"
            $Name = $Parts[0]
            $Status = $Parts[1]
            
            if ($Status -match "Up") {
                Write-Log "SUCCESS" "Container $Name is running"
            }
            else {
                $script:Issues += "Container $Name is not healthy: $Status"
                $script:HealthStatus = "degraded"
            }
        }
    }
    catch {
        $script:Warnings += "Could not check container status"
    }
    
    Write-Log "SUCCESS" "Docker services check completed"
    return $true
}

# Check system resources
function Test-SystemResources {
    Write-Log "INFO" "Checking system resources..."
    
    try {
        # CPU usage
        $CpuUsage = (Get-WmiObject -Class Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        if ($CpuUsage -gt $AlertThreshold) {
            $script:Warnings += "High CPU usage: $($CpuUsage)%"
        }
        Write-Log "INFO" "CPU usage: $($CpuUsage)%"
        
        # Memory usage
        $Memory = Get-WmiObject -Class Win32_OperatingSystem
        $TotalMemory = [math]::Round($Memory.TotalVisibleMemorySize / 1MB, 2)
        $FreeMemory = [math]::Round($Memory.FreePhysicalMemory / 1MB, 2)
        $UsedMemory = $TotalMemory - $FreeMemory
        $MemoryPercent = [math]::Round(($UsedMemory / $TotalMemory) * 100, 2)
        
        if ($MemoryPercent -gt $AlertThreshold) {
            $script:Warnings += "High memory usage: $($MemoryPercent)%"
        }
        Write-Log "INFO" "Memory usage: $($MemoryPercent)%"
        
        # Disk usage
        $Disk = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq (Split-Path -Qualifier $ProjectRoot) }
        if ($Disk) {
            $DiskPercent = [math]::Round((($Disk.Size - $Disk.FreeSpace) / $Disk.Size) * 100, 2)
            if ($DiskPercent -gt $AlertThreshold) {
                $script:Warnings += "High disk usage: $($DiskPercent)%"
            }
            Write-Log "INFO" "Disk usage: $($DiskPercent)%"
        }
    }
    catch {
        $script:Warnings += "Could not retrieve system resource information"
    }
    
    Write-Log "SUCCESS" "System resources check completed"
}

# Check network connectivity
function Test-Network {
    Write-Log "INFO" "Checking network connectivity..."
    
    # Check localhost ports
    $Ports = @(5000, 8080, 8181, 8051, 8052, 3737, 5432, 6379, 9090, 3000)
    
    foreach ($Port in $Ports) {
        try {
            $Connection = Test-NetConnection -ComputerName "localhost" -Port $Port -WarningAction SilentlyContinue
            if ($Connection.TcpTestSucceeded) {
                Write-Log "SUCCESS" "Port $Port is accessible"
            }
            else {
                Write-Log "INFO" "Port $Port is not accessible (service may be down)"
            }
        }
        catch {
            Write-Log "INFO" "Could not test port $Port"
        }
    }
    
    # Check external connectivity
    try {
        $Response = Invoke-WebRequest -Uri "https://httpbin.org/ip" -TimeoutSec 5 -UseBasicParsing
        if ($Response.StatusCode -eq 200) {
            Write-Log "SUCCESS" "External connectivity working"
        }
        else {
            $script:Warnings += "External connectivity issues"
        }
    }
    catch {
        $script:Warnings += "External connectivity issues"
    }
    
    Write-Log "SUCCESS" "Network connectivity check completed"
}

# Check AEGIS services
function Test-AegisServices {
    Write-Log "INFO" "Checking AEGIS services..."
    
    # Check main application
    try {
        $Response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 5 -UseBasicParsing
        if ($Response.StatusCode -eq 200) {
            Write-Log "SUCCESS" "AEGIS main service is responding"
        }
        else {
            $script:Issues += "AEGIS main service not responding"
            $script:HealthStatus = "degraded"
        }
    }
    catch {
        $script:Issues += "AEGIS main service not responding"
        $script:HealthStatus = "degraded"
    }
    
    # Check API server
    try {
        $Response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5 -UseBasicParsing
        if ($Response.StatusCode -eq 200) {
            Write-Log "SUCCESS" "AEGIS API server is responding"
        }
        else {
            $script:Warnings += "AEGIS API server not responding"
        }
    }
    catch {
        $script:Warnings += "AEGIS API server not responding"
    }
    
    # Check if TOR is running
    $TorProcess = Get-Process -Name "tor" -ErrorAction SilentlyContinue
    if ($TorProcess) {
        Write-Log "SUCCESS" "TOR service is running"
    }
    else {
        Write-Log "INFO" "TOR service is not running (optional)"
    }
    
    Write-Log "SUCCESS" "AEGIS services check completed"
}

# Check log files
function Test-Logs {
    Write-Log "INFO" "Checking log files..."
    
    $LogDir = Join-Path $ProjectRoot "logs"
    
    if (!(Test-Path $LogDir)) {
        $script:Warnings += "Log directory does not exist"
        return
    }
    
    # Check for recent errors
    $AegisLogPath = Join-Path $LogDir "aegis.log"
    if (Test-Path $AegisLogPath) {
        try {
            $ErrorCount = (Select-String -Path $AegisLogPath -Pattern "ERROR" | Measure-Object).Count
            if ($ErrorCount -gt 10) {
                $script:Warnings += "High error count in logs: $ErrorCount"
            }
        }
        catch {
            Write-Log "WARN" "Could not analyze log file"
        }
    }
    
    # Check log file sizes
    try {
        $LargeLogFiles = Get-ChildItem -Path $LogDir -Filter "*.log" | Where-Object { $_.Length -gt 100MB }
        foreach ($LogFile in $LargeLogFiles) {
            $script:Warnings += "Large log file: $($LogFile.Name)"
        }
    }
    catch {
        Write-Log "WARN" "Could not check log file sizes"
    }
    
    Write-Log "SUCCESS" "Log files check completed"
}

# Check security
function Test-Security {
    Write-Log "INFO" "Checking security..."
    
    # Check for default passwords
    $EnvFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $EnvFile) {
        try {
            $EnvContent = Get-Content $EnvFile -Raw
            if ($EnvContent -match "password123|admin123|secret123") {
                $script:Issues += "Default passwords detected in .env file"
                $script:HealthStatus = "unhealthy"
            }
        }
        catch {
            Write-Log "WARN" "Could not check .env file for default passwords"
        }
    }
    
    # Check for SSL certificates
    $SslDir = Join-Path $ProjectRoot "config\ssl"
    if (Test-Path $SslDir) {
        $CertFiles = Get-ChildItem -Path $SslDir -Filter "*.crt" -Recurse
        $CertFiles += Get-ChildItem -Path $SslDir -Filter "*.pem" -Recurse
        
        if ($CertFiles.Count -eq 0) {
            $script:Warnings += "No SSL certificates found"
        }
    }
    
    # Check Windows Defender status
    try {
        $DefenderStatus = Get-MpComputerStatus -ErrorAction SilentlyContinue
        if ($DefenderStatus -and !$DefenderStatus.RealTimeProtectionEnabled) {
            $script:Warnings += "Windows Defender real-time protection is disabled"
        }
    }
    catch {
        # Windows Defender cmdlets not available
    }
    
    Write-Log "SUCCESS" "Security check completed"
}

# Generate report
function New-Report {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    if ($Json) {
        # JSON output
        $Report = @{
            timestamp = $Timestamp
            status = $HealthStatus
            issues = $Issues
            warnings = $Warnings
            summary = @{
                total_issues = $Issues.Count
                total_warnings = $Warnings.Count
            }
        }
        
        $Report | ConvertTo-Json -Depth 3
    }
    else {
        # Human readable output
        Write-Host ""
        Write-Host "=========================================" -ForegroundColor $Colors.White
        Write-Host "AEGIS Framework Health Check Report" -ForegroundColor $Colors.White
        Write-Host "=========================================" -ForegroundColor $Colors.White
        Write-Host "Timestamp: $Timestamp" -ForegroundColor $Colors.White
        Write-Host "Overall Status: $HealthStatus" -ForegroundColor $Colors.White
        Write-Host ""
        
        if ($Issues.Count -gt 0) {
            Write-Host "Issues Found:" -ForegroundColor $Colors.Red
            foreach ($Issue in $Issues) {
                Write-Host "  ❌ $Issue" -ForegroundColor $Colors.Red
            }
            Write-Host ""
        }
        
        if ($Warnings.Count -gt 0) {
            Write-Host "Warnings:" -ForegroundColor $Colors.Yellow
            foreach ($Warning in $Warnings) {
                Write-Host "  ⚠️  $Warning" -ForegroundColor $Colors.Yellow
            }
            Write-Host ""
        }
        
        if ($Issues.Count -eq 0 -and $Warnings.Count -eq 0) {
            Write-Host "✅ All systems are healthy!" -ForegroundColor $Colors.Green
        }
        
        Write-Host "Summary:" -ForegroundColor $Colors.White
        Write-Host "  - Issues: $($Issues.Count)" -ForegroundColor $Colors.White
        Write-Host "  - Warnings: $($Warnings.Count)" -ForegroundColor $Colors.White
        Write-Host "=========================================" -ForegroundColor $Colors.White
    }
}

# Show usage
function Show-Usage {
    Write-Host @"
Usage: .\health_check.ps1 [OPTIONS]

AEGIS Framework Health Check Script

OPTIONS:
    -Verbose        Enable verbose output
    -Json           Output results in JSON format
    -Timeout SEC    Set timeout for checks (default: 30)
    -Help           Show this help message

EXAMPLES:
    .\health_check.ps1                Run basic health check
    .\health_check.ps1 -Verbose       Run with verbose output
    .\health_check.ps1 -Json          Output results in JSON format
    .\health_check.ps1 -Verbose -Timeout 60    Run with verbose output and 60s timeout

EXIT CODES:
    0   All checks passed (healthy)
    1   Critical issues found (unhealthy)
    2   Warnings found (degraded)
"@
}

# Main function
function Main {
    if ($Help) {
        Show-Usage
        exit 0
    }
    
    # Start health check
    Write-Log "INFO" "Starting AEGIS Framework health check..."
    
    # Run all checks
    Test-DockerServices
    Test-SystemResources
    Test-Network
    Test-AegisServices
    Test-Logs
    Test-Security
    
    # Generate report
    New-Report
    
    # Exit with appropriate code
    switch ($HealthStatus) {
        "healthy" {
            Write-Log "SUCCESS" "Health check completed - All systems healthy"
            exit 0
        }
        "degraded" {
            Write-Log "WARN" "Health check completed - System degraded"
            exit 2
        }
        "unhealthy" {
            Write-Log "ERROR" "Health check completed - Critical issues found"
            exit 1
        }
    }
}

# Run main function
Main