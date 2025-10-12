# AEGIS Framework - Maintenance Script (PowerShell)
# Automated system maintenance and optimization for Windows

param(
    [switch]$Verbose,
    [switch]$DryRun,
    [switch]$Quick,
    [switch]$Full,
    [switch]$Help
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogFile = Join-Path $ProjectRoot "logs\maintenance.log"
$BackupDir = Join-Path $ProjectRoot "backups"
$TempDir = Join-Path $env:TEMP "aegis_maintenance"
$MaxLogSizeMB = 100
$MaxBackupAgeDays = 30

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Maintenance statistics
$CleanedFiles = 0
$FreedSpaceBytes = 0
$RotatedLogs = 0
$OptimizedDatabases = 0
$FixedPermissions = 0

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

# Convert bytes to human readable format
function ConvertTo-HumanReadable {
    param([long]$Bytes)
    
    if ($Bytes -lt 1KB) {
        return "$($Bytes)B"
    }
    elseif ($Bytes -lt 1MB) {
        return "$([math]::Round($Bytes / 1KB, 2))KB"
    }
    elseif ($Bytes -lt 1GB) {
        return "$([math]::Round($Bytes / 1MB, 2))MB"
    }
    else {
        return "$([math]::Round($Bytes / 1GB, 2))GB"
    }
}

# Clean temporary files
function Clear-TempFiles {
    Write-Log "INFO" "Cleaning temporary files..."
    
    $TempDirs = @(
        $env:TEMP,
        (Join-Path $ProjectRoot "temp"),
        (Join-Path $ProjectRoot ".cache"),
        (Join-Path $ProjectRoot "node_modules\.cache")
    )
    
    $CleanedSize = 0
    
    foreach ($TempDir in $TempDirs) {
        if (Test-Path $TempDir) {
            try {
                # Find and clean AEGIS-related temp files
                $TempFiles = Get-ChildItem -Path $TempDir -Recurse -File | Where-Object { 
                    $_.Name -like "*aegis*" -and $_.LastWriteTime -lt (Get-Date).AddDays(-1)
                }
                
                foreach ($File in $TempFiles) {
                    if ($DryRun) {
                        Write-Log "INFO" "Would delete: $($File.FullName)"
                    }
                    else {
                        $FileSize = $File.Length
                        Remove-Item $File.FullName -Force -ErrorAction SilentlyContinue
                        $CleanedSize += $FileSize
                        $script:CleanedFiles++
                    }
                }
            }
            catch {
                Write-Log "WARN" "Could not clean temp directory: $TempDir"
            }
        }
    }
    
    # Clean Python cache files
    if (Test-Path $ProjectRoot) {
        try {
            $PythonCacheFiles = Get-ChildItem -Path $ProjectRoot -Recurse -File | Where-Object { 
                $_.Extension -in @(".pyc", ".pyo") -or $_.Name -eq "__pycache__"
            }
            
            foreach ($File in $PythonCacheFiles) {
                if ($DryRun) {
                    Write-Log "INFO" "Would delete: $($File.FullName)"
                }
                else {
                    $FileSize = $File.Length
                    Remove-Item $File.FullName -Force -ErrorAction SilentlyContinue
                    $CleanedSize += $FileSize
                    $script:CleanedFiles++
                }
            }
        }
        catch {
            Write-Log "WARN" "Could not clean Python cache files"
        }
    }
    
    $script:FreedSpaceBytes += $CleanedSize
    Write-Log "SUCCESS" "Cleaned $CleanedFiles temporary files, freed $(ConvertTo-HumanReadable $CleanedSize)"
}

# Rotate log files
function Invoke-LogRotation {
    Write-Log "INFO" "Rotating log files..."
    
    $LogDir = Join-Path $ProjectRoot "logs"
    
    if (!(Test-Path $LogDir)) {
        Write-Log "WARN" "Log directory does not exist: $LogDir"
        return
    }
    
    # Find large log files
    $LogFiles = Get-ChildItem -Path $LogDir -Filter "*.log" -File
    $MaxSizeBytes = $MaxLogSizeMB * 1MB
    
    foreach ($LogFile in $LogFiles) {
        if ($LogFile.Length -gt $MaxSizeBytes) {
            $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $RotatedName = "$($LogFile.BaseName).$Timestamp$($LogFile.Extension)"
            $RotatedPath = Join-Path $LogDir $RotatedName
            
            if ($DryRun) {
                Write-Log "INFO" "Would rotate: $($LogFile.Name) -> $RotatedName"
            }
            else {
                try {
                    Move-Item $LogFile.FullName $RotatedPath
                    New-Item -ItemType File -Path $LogFile.FullName -Force | Out-Null
                    
                    # Compress rotated log if possible
                    if (Test-Command "Compress-Archive") {
                        $CompressedPath = "$RotatedPath.zip"
                        Compress-Archive -Path $RotatedPath -DestinationPath $CompressedPath
                        Remove-Item $RotatedPath -Force
                    }
                    
                    $script:RotatedLogs++
                    Write-Log "SUCCESS" "Rotated log file: $($LogFile.Name)"
                }
                catch {
                    Write-Log "ERROR" "Failed to rotate log file: $($LogFile.Name)"
                }
            }
        }
    }
    
    # Clean old rotated logs
    if (!$DryRun) {
        try {
            $OldLogs = Get-ChildItem -Path $LogDir -File | Where-Object { 
                $_.Name -match "\.log\.\d{8}_\d{6}" -and $_.LastWriteTime -lt (Get-Date).AddDays(-$MaxBackupAgeDays)
            }
            
            foreach ($OldLog in $OldLogs) {
                Remove-Item $OldLog.FullName -Force
            }
        }
        catch {
            Write-Log "WARN" "Could not clean old rotated logs"
        }
    }
    
    Write-Log "SUCCESS" "Rotated $RotatedLogs log files"
}

# Clean old backups
function Clear-OldBackups {
    Write-Log "INFO" "Cleaning old backups..."
    
    if (!(Test-Path $BackupDir)) {
        Write-Log "INFO" "Backup directory does not exist: $BackupDir"
        return
    }
    
    $CleanedBackups = 0
    $CleanedSize = 0
    
    try {
        # Clean backups older than MaxBackupAgeDays
        $OldBackups = Get-ChildItem -Path $BackupDir -File -Recurse | Where-Object { 
            $_.LastWriteTime -lt (Get-Date).AddDays(-$MaxBackupAgeDays)
        }
        
        foreach ($Backup in $OldBackups) {
            if ($DryRun) {
                Write-Log "INFO" "Would delete old backup: $($Backup.FullName)"
            }
            else {
                $FileSize = $Backup.Length
                Remove-Item $Backup.FullName -Force
                $CleanedSize += $FileSize
                $CleanedBackups++
            }
        }
    }
    catch {
        Write-Log "WARN" "Could not clean old backups"
    }
    
    $script:FreedSpaceBytes += $CleanedSize
    Write-Log "SUCCESS" "Cleaned $CleanedBackups old backups, freed $(ConvertTo-HumanReadable $CleanedSize)"
}

# Optimize databases
function Optimize-Databases {
    Write-Log "INFO" "Optimizing databases..."
    
    # Check if Docker is available
    if (!(Test-Command "docker")) {
        Write-Log "WARN" "Docker not available, skipping database optimization"
        return
    }
    
    try {
        # Optimize PostgreSQL
        $PostgresContainer = docker ps --filter "name=postgres" --format "{{.ID}}" 2>$null
        if ($PostgresContainer) {
            if ($DryRun) {
                Write-Log "INFO" "Would optimize PostgreSQL database"
            }
            else {
                Write-Log "INFO" "Running PostgreSQL VACUUM and ANALYZE..."
                docker exec $PostgresContainer psql -U postgres -d aegis -c "VACUUM ANALYZE;" 2>$null
                $script:OptimizedDatabases++
                Write-Log "SUCCESS" "Optimized PostgreSQL database"
            }
        }
        
        # Optimize Redis
        $RedisContainer = docker ps --filter "name=redis" --format "{{.ID}}" 2>$null
        if ($RedisContainer) {
            if ($DryRun) {
                Write-Log "INFO" "Would optimize Redis database"
            }
            else {
                Write-Log "INFO" "Running Redis BGREWRITEAOF..."
                docker exec $RedisContainer redis-cli BGREWRITEAOF 2>$null
                $script:OptimizedDatabases++
                Write-Log "SUCCESS" "Optimized Redis database"
            }
        }
    }
    catch {
        Write-Log "WARN" "Could not optimize databases"
    }
    
    Write-Log "SUCCESS" "Optimized $OptimizedDatabases databases"
}

# Clean Docker resources
function Clear-DockerResources {
    Write-Log "INFO" "Cleaning Docker resources..."
    
    if (!(Test-Command "docker")) {
        Write-Log "WARN" "Docker not available, skipping Docker cleanup"
        return
    }
    
    if ($DryRun) {
        Write-Log "INFO" "Would clean Docker resources"
        try {
            docker system df 2>$null
        }
        catch {
            Write-Log "WARN" "Could not get Docker system info"
        }
    }
    else {
        try {
            # Remove unused containers, networks, images, and build cache
            docker system prune -f --volumes 2>$null
            Write-Log "SUCCESS" "Cleaned Docker resources"
        }
        catch {
            Write-Log "WARN" "Could not clean Docker resources"
        }
    }
}

# Update system packages
function Update-SystemPackages {
    Write-Log "INFO" "Updating system packages..."
    
    if ($DryRun) {
        Write-Log "INFO" "Would update system packages"
        return
    }
    
    try {
        # Check for Chocolatey
        if (Test-Command "choco") {
            Write-Log "INFO" "Updating packages using Chocolatey..."
            choco upgrade all -y --limit-output 2>$null
            Write-Log "SUCCESS" "Updated packages using Chocolatey"
        }
        # Check for Scoop
        elseif (Test-Command "scoop") {
            Write-Log "INFO" "Updating packages using Scoop..."
            scoop update * 2>$null
            Write-Log "SUCCESS" "Updated packages using Scoop"
        }
        # Check for winget
        elseif (Test-Command "winget") {
            Write-Log "INFO" "Updating packages using winget..."
            winget upgrade --all --silent 2>$null
            Write-Log "SUCCESS" "Updated packages using winget"
        }
        else {
            Write-Log "WARN" "No supported package manager found"
        }
        
        # Update Windows if possible
        if (Get-Module -ListAvailable -Name PSWindowsUpdate) {
            Import-Module PSWindowsUpdate
            Get-WUInstall -AcceptAll -AutoReboot:$false -Silent
            Write-Log "SUCCESS" "Updated Windows packages"
        }
    }
    catch {
        Write-Log "WARN" "Could not update system packages: $($_.Exception.Message)"
    }
}

# Check and repair file permissions
function Repair-FilePermissions {
    Write-Log "INFO" "Checking and fixing file permissions..."
    
    $FixedFiles = 0
    
    try {
        # Fix script permissions
        $ScriptFiles = Get-ChildItem -Path (Join-Path $ProjectRoot "scripts") -Filter "*.ps1" -File -ErrorAction SilentlyContinue
        
        foreach ($Script in $ScriptFiles) {
            # Check if script is executable (not applicable on Windows, but we can check if it's not blocked)
            $Zone = Get-Content -Path "$($Script.FullName):Zone.Identifier" -ErrorAction SilentlyContinue
            if ($Zone) {
                if ($DryRun) {
                    Write-Log "INFO" "Would unblock script: $($Script.Name)"
                }
                else {
                    Unblock-File -Path $Script.FullName
                    $FixedFiles++
                    Write-Log "INFO" "Unblocked script: $($Script.Name)"
                }
            }
        }
        
        # Check sensitive files
        $SensitiveFiles = @(
            (Join-Path $ProjectRoot ".env"),
            (Join-Path $ProjectRoot "config\ssl"),
            (Join-Path $ProjectRoot "config\keys")
        )
        
        foreach ($File in $SensitiveFiles) {
            if (Test-Path $File) {
                try {
                    $Acl = Get-Acl $File
                    $AccessRules = $Acl.Access | Where-Object { $_.IdentityReference -ne $env:USERNAME -and $_.IdentityReference -notlike "*SYSTEM*" -and $_.IdentityReference -notlike "*Administrators*" }
                    
                    if ($AccessRules) {
                        if ($DryRun) {
                            Write-Log "INFO" "Would fix permissions for: $File"
                        }
                        else {
                            # Remove unnecessary access rules
                            foreach ($Rule in $AccessRules) {
                                $Acl.RemoveAccessRule($Rule)
                            }
                            Set-Acl -Path $File -AclObject $Acl
                            $FixedFiles++
                            Write-Log "INFO" "Fixed permissions for: $File"
                        }
                    }
                }
                catch {
                    Write-Log "WARN" "Could not check permissions for: $File"
                }
            }
        }
    }
    catch {
        Write-Log "WARN" "Could not fix file permissions"
    }
    
    $script:FixedPermissions = $FixedFiles
    Write-Log "SUCCESS" "Fixed permissions for $FixedFiles files"
}

# Generate maintenance report
function New-MaintenanceReport {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $FreedSpaceHuman = ConvertTo-HumanReadable $FreedSpaceBytes
    
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor $Colors.White
    Write-Host "AEGIS Framework Maintenance Report" -ForegroundColor $Colors.White
    Write-Host "=========================================" -ForegroundColor $Colors.White
    Write-Host "Timestamp: $Timestamp" -ForegroundColor $Colors.White
    Write-Host "Mode: $(if ($DryRun) { "DRY RUN" } else { "LIVE" })" -ForegroundColor $Colors.White
    Write-Host ""
    Write-Host "Summary:" -ForegroundColor $Colors.White
    Write-Host "  - Cleaned files: $CleanedFiles" -ForegroundColor $Colors.White
    Write-Host "  - Freed space: $FreedSpaceHuman" -ForegroundColor $Colors.White
    Write-Host "  - Rotated logs: $RotatedLogs" -ForegroundColor $Colors.White
    Write-Host "  - Optimized databases: $OptimizedDatabases" -ForegroundColor $Colors.White
    Write-Host "  - Fixed permissions: $FixedPermissions" -ForegroundColor $Colors.White
    Write-Host ""
    Write-Host "Maintenance completed successfully!" -ForegroundColor $Colors.Green
    Write-Host "=========================================" -ForegroundColor $Colors.White
}

# Show usage
function Show-Usage {
    Write-Host @"
Usage: .\maintenance.ps1 [OPTIONS]

AEGIS Framework Maintenance Script

OPTIONS:
    -Verbose        Enable verbose output
    -DryRun         Show what would be done without making changes
    -Quick          Run quick maintenance (skip package updates)
    -Full           Run full maintenance including system updates
    -Help           Show this help message

MAINTENANCE TASKS:
    - Clean temporary files and caches
    - Rotate large log files
    - Clean old backups
    - Optimize databases (PostgreSQL, Redis)
    - Clean Docker resources
    - Update system packages (with -Full)
    - Fix file permissions

EXAMPLES:
    .\maintenance.ps1                Run standard maintenance
    .\maintenance.ps1 -Verbose       Run with verbose output
    .\maintenance.ps1 -DryRun        Dry run (show what would be done)
    .\maintenance.ps1 -Full          Run full maintenance including updates
    .\maintenance.ps1 -Quick         Run quick maintenance only

SCHEDULING:
    Add to Task Scheduler for automated maintenance:
    schtasks /create /tn "AEGIS Maintenance" /tr "powershell.exe -File C:\path\to\maintenance.ps1 -Quick" /sc weekly /d SUN /st 02:00
"@
}

# Main function
function Main {
    if ($Help) {
        Show-Usage
        exit 0
    }
    
    # Create necessary directories
    $LogDir = Split-Path -Parent $LogFile
    if (!(Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    
    if (!(Test-Path $TempDir)) {
        New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    }
    
    # Start maintenance
    Write-Log "INFO" "Starting AEGIS Framework maintenance..."
    if ($DryRun) {
        Write-Log "INFO" "Running in DRY RUN mode - no changes will be made"
    }
    
    # Run maintenance tasks
    Clear-TempFiles
    Invoke-LogRotation
    Clear-OldBackups
    Optimize-Databases
    Clear-DockerResources
    Repair-FilePermissions
    
    # Run additional tasks based on mode
    if ($Full -and !$Quick) {
        Update-SystemPackages
    }
    
    # Generate report
    New-MaintenanceReport
    
    # Cleanup
    if (Test-Path $TempDir) {
        Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    Write-Log "SUCCESS" "Maintenance completed successfully"
}

# Run main function
Main