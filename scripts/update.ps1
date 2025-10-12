# AEGIS Framework - Update Script (PowerShell)
# Automated system updates and dependency management for Windows

param(
    [switch]$Verbose,
    [switch]$DryRun,
    [switch]$Force,
    [switch]$SkipBackup,
    [switch]$SkipTests,
    [switch]$UpdateSystem,
    [switch]$UpdateDocker,
    [switch]$UpdatePython,
    [switch]$UpdateNode,
    [switch]$UpdateAll,
    [switch]$Help
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogFile = Join-Path $ProjectRoot "logs\update.log"
$BackupDir = Join-Path $ProjectRoot "backups\pre-update"
$TempDir = Join-Path $env:TEMP "aegis_update_$(Get-Random)"

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Update statistics
$UpdatedPackages = 0
$UpdatedServices = 0
$FailedUpdates = 0

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

# Check system requirements
function Test-SystemRequirements {
    Write-Log "INFO" "Checking system requirements..."
    
    $MissingDeps = @()
    
    # Check essential commands
    $RequiredCommands = @("git", "powershell")
    foreach ($Cmd in $RequiredCommands) {
        if (!(Test-Command $Cmd)) {
            $MissingDeps += $Cmd
        }
    }
    
    if ($MissingDeps.Count -gt 0) {
        Write-Log "ERROR" "Missing required dependencies: $($MissingDeps -join ', ')"
        exit 1
    }
    
    # Check disk space (require at least 1GB free)
    try {
        $Drive = (Get-Item $ProjectRoot).PSDrive
        $FreeSpace = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='$($Drive.Name):'").FreeSpace
        $FreeSpaceGB = [math]::Round($FreeSpace / 1GB, 2)
        
        if ($FreeSpaceGB -lt 1) {
            Write-Log "WARN" "Low disk space detected. Available: ${FreeSpaceGB}GB"
        }
    }
    catch {
        Write-Log "WARN" "Could not check disk space"
    }
    
    Write-Log "SUCCESS" "System requirements check passed"
}

# Create pre-update backup
function New-PreUpdateBackup {
    if ($SkipBackup) {
        Write-Log "INFO" "Skipping backup as requested"
        return
    }
    
    Write-Log "INFO" "Creating pre-update backup..."
    
    $BackupTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BackupPath = Join-Path $BackupDir "backup_$BackupTimestamp"
    
    if ($DryRun) {
        Write-Log "INFO" "Would create backup at: $BackupPath"
        return
    }
    
    New-Item -ItemType Directory -Path $BackupPath -Force | Out-Null
    
    # Backup configuration files
    $ConfigDir = Join-Path $ProjectRoot "config"
    if (Test-Path $ConfigDir) {
        Copy-Item -Path $ConfigDir -Destination $BackupPath -Recurse -Force
    }
    
    # Backup environment files
    $EnvFiles = @(".env", ".env.local", ".env.production")
    foreach ($EnvFile in $EnvFiles) {
        $EnvPath = Join-Path $ProjectRoot $EnvFile
        if (Test-Path $EnvPath) {
            Copy-Item -Path $EnvPath -Destination $BackupPath -Force
        }
    }
    
    # Backup Docker compose files
    $ComposeFiles = @("docker-compose.yml", "docker-compose.override.yml")
    foreach ($ComposeFile in $ComposeFiles) {
        $ComposePath = Join-Path $ProjectRoot $ComposeFile
        if (Test-Path $ComposePath) {
            Copy-Item -Path $ComposePath -Destination $BackupPath -Force
        }
    }
    
    # Backup package files
    $PackageFiles = @("package.json", "requirements.txt", "Pipfile", "pyproject.toml")
    foreach ($PackageFile in $PackageFiles) {
        $PackagePath = Join-Path $ProjectRoot $PackageFile
        if (Test-Path $PackagePath) {
            Copy-Item -Path $PackagePath -Destination $BackupPath -Force
        }
    }
    
    Write-Log "SUCCESS" "Backup created at: $BackupPath"
}

# Update system packages
function Update-SystemPackages {
    if (!$UpdateSystem -and !$UpdateAll) {
        return
    }
    
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
            $script:UpdatedPackages++
            Write-Log "SUCCESS" "Updated packages using Chocolatey"
        }
        # Check for Scoop
        elseif (Test-Command "scoop") {
            Write-Log "INFO" "Updating packages using Scoop..."
            scoop update * 2>$null
            $script:UpdatedPackages++
            Write-Log "SUCCESS" "Updated packages using Scoop"
        }
        # Check for winget
        elseif (Test-Command "winget") {
            Write-Log "INFO" "Updating packages using winget..."
            winget upgrade --all --silent 2>$null
            $script:UpdatedPackages++
            Write-Log "SUCCESS" "Updated packages using winget"
        }
        else {
            Write-Log "WARN" "No supported package manager found"
        }
        
        # Update Windows if PSWindowsUpdate module is available
        if (Get-Module -ListAvailable -Name PSWindowsUpdate) {
            Write-Log "INFO" "Updating Windows packages..."
            Import-Module PSWindowsUpdate
            Get-WUInstall -AcceptAll -AutoReboot:$false -Silent
            Write-Log "SUCCESS" "Updated Windows packages"
        }
    }
    catch {
        Write-Log "ERROR" "Failed to update system packages: $($_.Exception.Message)"
        $script:FailedUpdates++
    }
}

# Update Docker and Docker Compose
function Update-DockerComponents {
    if (!$UpdateDocker -and !$UpdateAll) {
        return
    }
    
    Write-Log "INFO" "Updating Docker components..."
    
    if (!(Test-Command "docker")) {
        Write-Log "WARN" "Docker not installed, skipping Docker update"
        return
    }
    
    if ($DryRun) {
        Write-Log "INFO" "Would update Docker components"
        return
    }
    
    try {
        # Update Docker images
        Write-Log "INFO" "Pulling latest Docker images..."
        
        # Get list of images used in docker-compose
        $ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"
        if (Test-Path $ComposeFile) {
            $ComposeContent = Get-Content $ComposeFile
            $Images = $ComposeContent | Select-String "^\s*image:" | ForEach-Object {
                ($_ -split ":")[1].Trim()
            } | Sort-Object -Unique
            
            foreach ($Image in $Images) {
                Write-Log "INFO" "Pulling image: $Image"
                try {
                    docker pull $Image 2>$null
                    Write-Log "SUCCESS" "Updated image: $Image"
                    $script:UpdatedServices++
                }
                catch {
                    Write-Log "ERROR" "Failed to update image: $Image"
                    $script:FailedUpdates++
                }
            }
        }
        
        # Update Docker Compose if installed via pip
        if ((Test-Command "pip") -and (pip list | Select-String "docker-compose")) {
            Write-Log "INFO" "Updating Docker Compose..."
            pip install --upgrade docker-compose 2>$null
            $script:UpdatedPackages++
        }
    }
    catch {
        Write-Log "ERROR" "Failed to update Docker components: $($_.Exception.Message)"
        $script:FailedUpdates++
    }
    
    Write-Log "SUCCESS" "Docker components updated"
}

# Update Python dependencies
function Update-PythonDependencies {
    if (!$UpdatePython -and !$UpdateAll) {
        return
    }
    
    Write-Log "INFO" "Updating Python dependencies..."
    
    if ($DryRun) {
        Write-Log "INFO" "Would update Python dependencies"
        return
    }
    
    Push-Location $ProjectRoot
    
    try {
        # Update pip itself
        if (Test-Command "pip") {
            Write-Log "INFO" "Updating pip..."
            pip install --upgrade pip 2>$null
        }
        
        # Update requirements.txt dependencies
        $RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
        if (Test-Path $RequirementsFile) {
            Write-Log "INFO" "Updating requirements.txt dependencies..."
            try {
                pip install --upgrade -r requirements.txt 2>$null
                Write-Log "SUCCESS" "Updated requirements.txt dependencies"
                $script:UpdatedPackages++
            }
            catch {
                Write-Log "ERROR" "Failed to update requirements.txt dependencies"
                $script:FailedUpdates++
            }
        }
        
        # Update Pipfile dependencies
        $PipfileFile = Join-Path $ProjectRoot "Pipfile"
        if ((Test-Path $PipfileFile) -and (Test-Command "pipenv")) {
            Write-Log "INFO" "Updating Pipfile dependencies..."
            try {
                pipenv update 2>$null
                Write-Log "SUCCESS" "Updated Pipfile dependencies"
                $script:UpdatedPackages++
            }
            catch {
                Write-Log "ERROR" "Failed to update Pipfile dependencies"
                $script:FailedUpdates++
            }
        }
        
        # Update pyproject.toml dependencies
        $PyprojectFile = Join-Path $ProjectRoot "pyproject.toml"
        if ((Test-Path $PyprojectFile) -and (Test-Command "poetry")) {
            Write-Log "INFO" "Updating pyproject.toml dependencies..."
            try {
                poetry update 2>$null
                Write-Log "SUCCESS" "Updated pyproject.toml dependencies"
                $script:UpdatedPackages++
            }
            catch {
                Write-Log "ERROR" "Failed to update pyproject.toml dependencies"
                $script:FailedUpdates++
            }
        }
    }
    catch {
        Write-Log "ERROR" "Failed to update Python dependencies: $($_.Exception.Message)"
        $script:FailedUpdates++
    }
    finally {
        Pop-Location
    }
    
    Write-Log "SUCCESS" "Python dependencies updated"
}

# Update Node.js dependencies
function Update-NodeDependencies {
    if (!$UpdateNode -and !$UpdateAll) {
        return
    }
    
    Write-Log "INFO" "Updating Node.js dependencies..."
    
    if (!(Test-Command "npm")) {
        Write-Log "WARN" "Node.js/npm not installed, skipping Node.js update"
        return
    }
    
    if ($DryRun) {
        Write-Log "INFO" "Would update Node.js dependencies"
        return
    }
    
    Push-Location $ProjectRoot
    
    try {
        # Update npm itself
        Write-Log "INFO" "Updating npm..."
        npm install -g npm@latest 2>$null
        
        # Update package.json dependencies
        $PackageJsonFile = Join-Path $ProjectRoot "package.json"
        if (Test-Path $PackageJsonFile) {
            Write-Log "INFO" "Updating package.json dependencies..."
            try {
                npm update 2>$null
                Write-Log "SUCCESS" "Updated package.json dependencies"
                $script:UpdatedPackages++
            }
            catch {
                Write-Log "ERROR" "Failed to update package.json dependencies"
                $script:FailedUpdates++
            }
            
            # Audit and fix vulnerabilities
            Write-Log "INFO" "Auditing and fixing npm vulnerabilities..."
            try {
                npm audit fix 2>$null
            }
            catch {
                # npm audit fix can fail but it's not critical
            }
        }
        
        # Update yarn dependencies if yarn.lock exists
        $YarnLockFile = Join-Path $ProjectRoot "yarn.lock"
        if ((Test-Path $YarnLockFile) -and (Test-Command "yarn")) {
            Write-Log "INFO" "Updating yarn dependencies..."
            try {
                yarn upgrade 2>$null
                Write-Log "SUCCESS" "Updated yarn dependencies"
                $script:UpdatedPackages++
            }
            catch {
                Write-Log "ERROR" "Failed to update yarn dependencies"
                $script:FailedUpdates++
            }
        }
    }
    catch {
        Write-Log "ERROR" "Failed to update Node.js dependencies: $($_.Exception.Message)"
        $script:FailedUpdates++
    }
    finally {
        Pop-Location
    }
    
    Write-Log "SUCCESS" "Node.js dependencies updated"
}

# Update AEGIS codebase
function Update-AegisCodebase {
    Write-Log "INFO" "Updating AEGIS codebase..."
    
    Push-Location $ProjectRoot
    
    try {
        # Check if we're in a git repository
        if (!(Test-Path ".git")) {
            Write-Log "WARN" "Not a git repository, skipping code update"
            return
        }
        
        if ($DryRun) {
            Write-Log "INFO" "Would update AEGIS codebase from git"
            return
        }
        
        # Check for uncommitted changes
        $GitStatus = git status --porcelain 2>$null
        $StashCreated = $false
        
        if ($GitStatus) {
            Write-Log "INFO" "Stashing local changes..."
            git stash push -m "Pre-update stash $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" 2>$null
            $StashCreated = $true
        }
        
        # Fetch latest changes
        Write-Log "INFO" "Fetching latest changes..."
        git fetch origin 2>$null
        
        # Get current branch
        $CurrentBranch = git branch --show-current 2>$null
        
        # Pull latest changes
        Write-Log "INFO" "Pulling latest changes for branch: $CurrentBranch"
        try {
            git pull origin $CurrentBranch 2>$null
            Write-Log "SUCCESS" "Updated AEGIS codebase"
            $script:UpdatedServices++
        }
        catch {
            Write-Log "ERROR" "Failed to update AEGIS codebase"
            $script:FailedUpdates++
            
            # Restore stashed changes if any
            if ($StashCreated) {
                git stash pop 2>$null
            }
            return
        }
        
        # Restore stashed changes if any
        if ($StashCreated) {
            Write-Log "INFO" "Restoring stashed changes..."
            try {
                git stash pop 2>$null
            }
            catch {
                Write-Log "WARN" "Could not restore stashed changes automatically"
            }
        }
    }
    catch {
        Write-Log "ERROR" "Failed to update AEGIS codebase: $($_.Exception.Message)"
        $script:FailedUpdates++
    }
    finally {
        Pop-Location
    }
    
    Write-Log "SUCCESS" "AEGIS codebase updated"
}

# Run post-update tasks
function Invoke-PostUpdateTasks {
    Write-Log "INFO" "Running post-update tasks..."
    
    if ($DryRun) {
        Write-Log "INFO" "Would run post-update tasks"
        return
    }
    
    Push-Location $ProjectRoot
    
    try {
        # Rebuild Docker containers if docker-compose.yml exists
        $ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"
        if ((Test-Path $ComposeFile) -and (Test-Command "docker-compose")) {
            Write-Log "INFO" "Rebuilding Docker containers..."
            try {
                docker-compose build --no-cache 2>$null
            }
            catch {
                Write-Log "WARN" "Failed to rebuild containers"
            }
        }
        
        # Install/update pre-commit hooks if .pre-commit-config.yaml exists
        $PreCommitConfig = Join-Path $ProjectRoot ".pre-commit-config.yaml"
        if ((Test-Path $PreCommitConfig) -and (Test-Command "pre-commit")) {
            Write-Log "INFO" "Updating pre-commit hooks..."
            pre-commit install 2>$null
            pre-commit autoupdate 2>$null
        }
        
        # Clear Python cache
        Get-ChildItem -Path $ProjectRoot -Recurse -Directory -Name "__pycache__" -ErrorAction SilentlyContinue | ForEach-Object {
            Remove-Item -Path $_ -Recurse -Force -ErrorAction SilentlyContinue
        }
        Get-ChildItem -Path $ProjectRoot -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue | ForEach-Object {
            Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
        }
        
        # Clear Node.js cache if node_modules exists
        $NodeModulesDir = Join-Path $ProjectRoot "node_modules"
        if ((Test-Path $NodeModulesDir) -and (Test-Command "npm")) {
            try {
                npm cache clean --force 2>$null
            }
            catch {
                # npm cache clean can fail but it's not critical
            }
        }
    }
    catch {
        Write-Log "WARN" "Some post-update tasks failed: $($_.Exception.Message)"
    }
    finally {
        Pop-Location
    }
    
    Write-Log "SUCCESS" "Post-update tasks completed"
}

# Run tests after update
function Invoke-PostUpdateTests {
    if ($SkipTests) {
        Write-Log "INFO" "Skipping tests as requested"
        return
    }
    
    Write-Log "INFO" "Running tests after update..."
    
    if ($DryRun) {
        Write-Log "INFO" "Would run tests"
        return
    }
    
    Push-Location $ProjectRoot
    
    try {
        # Run Python tests
        $PytestConfig = Join-Path $ProjectRoot "pytest.ini"
        $TestsDir = Join-Path $ProjectRoot "tests"
        if ((Test-Path $PytestConfig) -or (Test-Path $TestsDir)) {
            Write-Log "INFO" "Running Python tests..."
            if (Test-Command "pytest") {
                try {
                    pytest --tb=short -q 2>$null
                }
                catch {
                    Write-Log "WARN" "Some Python tests failed"
                }
            }
        }
        
        # Run Node.js tests
        $PackageJsonFile = Join-Path $ProjectRoot "package.json"
        if ((Test-Path $PackageJsonFile) -and (Test-Command "npm")) {
            try {
                npm run test --if-present 2>$null
                Write-Log "SUCCESS" "Node.js tests passed"
            }
            catch {
                Write-Log "WARN" "Some Node.js tests failed"
            }
        }
        
        # Run linting
        if (Test-Command "flake8") {
            try {
                flake8 . 2>$null
            }
            catch {
                Write-Log "WARN" "Linting issues found"
            }
        }
    }
    catch {
        Write-Log "WARN" "Some tests failed: $($_.Exception.Message)"
    }
    finally {
        Pop-Location
    }
    
    Write-Log "SUCCESS" "Tests completed"
}

# Generate update report
function New-UpdateReport {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor $Colors.White
    Write-Host "AEGIS Framework Update Report" -ForegroundColor $Colors.White
    Write-Host "=========================================" -ForegroundColor $Colors.White
    Write-Host "Timestamp: $Timestamp" -ForegroundColor $Colors.White
    Write-Host "Mode: $(if ($DryRun) { "DRY RUN" } else { "LIVE" })" -ForegroundColor $Colors.White
    Write-Host ""
    Write-Host "Summary:" -ForegroundColor $Colors.White
    Write-Host "  - Updated packages: $UpdatedPackages" -ForegroundColor $Colors.White
    Write-Host "  - Updated services: $UpdatedServices" -ForegroundColor $Colors.White
    Write-Host "  - Failed updates: $FailedUpdates" -ForegroundColor $Colors.White
    Write-Host ""
    
    if ($FailedUpdates -gt 0) {
        Write-Host "Update completed with some failures!" -ForegroundColor $Colors.Yellow
        Write-Host "Check the log file for details: $LogFile" -ForegroundColor $Colors.White
    }
    else {
        Write-Host "Update completed successfully!" -ForegroundColor $Colors.Green
    }
    
    Write-Host "=========================================" -ForegroundColor $Colors.White
}

# Cleanup function
function Remove-TempFiles {
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Show usage
function Show-Usage {
    Write-Host @"
Usage: .\update.ps1 [OPTIONS]

AEGIS Framework Update Script

OPTIONS:
    -Verbose        Enable verbose output
    -DryRun         Show what would be done without making changes
    -Force          Force update even if checks fail
    -SkipBackup     Skip creating pre-update backup
    -SkipTests      Skip running tests after update
    -UpdateSystem   Update system packages
    -UpdateDocker   Update Docker components
    -UpdatePython   Update Python dependencies
    -UpdateNode     Update Node.js dependencies
    -UpdateAll      Update everything (system, docker, python, node)
    -Help           Show this help message

UPDATE COMPONENTS:
    - AEGIS codebase (git pull)
    - System packages (choco, scoop, winget, Windows Update)
    - Docker images and containers
    - Python dependencies (pip, pipenv, poetry)
    - Node.js dependencies (npm, yarn)
    - Post-update tasks (rebuild, cache clear)

EXAMPLES:
    .\update.ps1                    Update AEGIS codebase only
    .\update.ps1 -UpdateAll         Update everything
    .\update.ps1 -UpdatePython -UpdateNode  Update Python and Node.js dependencies
    .\update.ps1 -DryRun -UpdateAll Dry run of full update
    .\update.ps1 -Verbose -UpdateSystem  Update system packages with verbose output

SCHEDULING:
    Add to Task Scheduler for automated updates:
    schtasks /create /tn "AEGIS Update" /tr "powershell.exe -File C:\path\to\update.ps1 -UpdatePython -UpdateNode" /sc weekly /d SUN /st 02:00
"@
}

# Main function
function Main {
    if ($Help) {
        Show-Usage
        exit 0
    }
    
    # Create temporary directory
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    
    # Create log directory
    $LogDir = Split-Path -Parent $LogFile
    if (!(Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    
    # Start update process
    Write-Log "INFO" "Starting AEGIS Framework update..."
    if ($DryRun) {
        Write-Log "INFO" "Running in DRY RUN mode - no changes will be made"
    }
    
    try {
        # Run update steps
        Test-SystemRequirements
        New-PreUpdateBackup
        Update-AegisCodebase
        Update-SystemPackages
        Update-DockerComponents
        Update-PythonDependencies
        Update-NodeDependencies
        Invoke-PostUpdateTasks
        Invoke-PostUpdateTests
        
        # Generate report
        New-UpdateReport
        
        Write-Log "SUCCESS" "Update process completed"
        
        # Exit with error code if there were failures
        if ($FailedUpdates -gt 0) {
            exit 1
        }
    }
    finally {
        # Cleanup
        Remove-TempFiles
    }
}

# Run main function
Main