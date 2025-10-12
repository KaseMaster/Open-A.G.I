# AEGIS Framework - Deployment Script (PowerShell)
# Advanced Encrypted Governance and Intelligence System
# This script handles deployment to various environments on Windows

param(
    [Parameter(Position=0)]
    [ValidateSet("local", "staging", "production", "kubernetes")]
    [string]$Environment = "production",
    
    [switch]$BuildOnly,
    [switch]$Push,
    [switch]$SkipTests,
    [switch]$SkipBackup,
    [switch]$Rollback,
    [string]$Version,
    [switch]$Help
)

# Configuration
$AegisDir = Split-Path -Parent $PSScriptRoot
$DockerRegistry = "ghcr.io/aegis"
$ImageName = "aegis-framework"
$DefaultVersion = if (Test-Path "$AegisDir\VERSION") { Get-Content "$AegisDir\VERSION" } else { "1.0.0" }
$Version = if ($Version) { $Version } else { $DefaultVersion }
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$HealthCheckTimeout = 300

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
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

function Write-Header {
    param([string]$Message)
    Write-Host "[AEGIS DEPLOY] $Message" -ForegroundColor $Colors.Magenta
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

# Show help
function Show-Help {
    @"
AEGIS Framework Deployment Script (PowerShell)

Usage: .\deploy.ps1 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    local       Deploy to local Docker environment
    staging     Deploy to staging environment
    production  Deploy to production environment
    kubernetes  Deploy to Kubernetes cluster

OPTIONS:
    -BuildOnly      Only build images, don't deploy
    -Push           Push images to registry
    -SkipTests      Skip running tests before deployment
    -SkipBackup     Skip database backup before deployment
    -Rollback       Rollback to previous version
    -Version        Specify version to deploy
    -Help           Show this help message

EXAMPLES:
    .\deploy.ps1 local                    # Deploy to local environment
    .\deploy.ps1 production -Push         # Deploy to production and push images
    .\deploy.ps1 staging -SkipTests       # Deploy to staging without tests
    .\deploy.ps1 -Rollback production     # Rollback production deployment

"@
}

# Check prerequisites
function Test-Prerequisites {
    Write-Header "Checking deployment prerequisites..."
    
    # Check required commands
    $RequiredCommands = @("docker", "docker-compose")
    
    if ($Environment -eq "kubernetes") {
        $RequiredCommands += @("kubectl", "helm")
    }
    
    foreach ($cmd in $RequiredCommands) {
        if (-not (Test-Command $cmd)) {
            Write-Error "$cmd is not installed or not in PATH"
            exit 1
        }
    }
    
    # Check Docker daemon
    try {
        docker info | Out-Null
    }
    catch {
        Write-Error "Docker daemon is not running"
        exit 1
    }
    
    # Check environment file
    if (-not (Test-Path "$AegisDir\.env")) {
        Write-Error ".env file not found. Run setup script first."
        exit 1
    }
    
    # Load environment variables
    Get-Content "$AegisDir\.env" | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    
    Write-Success "Prerequisites check completed"
}

# Run tests
function Invoke-Tests {
    if ($SkipTests) {
        Write-Warning "Skipping tests as requested"
        return
    }
    
    Write-Header "Running tests before deployment..."
    
    Set-Location $AegisDir
    
    # Activate virtual environment if it exists
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    }
    
    # Run unit tests
    if (Test-Command pytest) {
        Write-Info "Running unit tests..."
        pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Unit tests failed"
            exit 1
        }
    }
    
    # Run security tests
    if (Test-Command bandit) {
        Write-Info "Running security tests..."
        bandit -r src/ -f json -o security-report.json
    }
    
    # Run linting
    if (Test-Command ruff) {
        Write-Info "Running code quality checks..."
        ruff check src/
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Code quality checks failed"
            exit 1
        }
    }
    
    Write-Success "Tests completed successfully"
}

# Build Docker images
function Build-Images {
    Write-Header "Building Docker images..."
    
    Set-Location $AegisDir
    
    # Get Git commit hash
    $GitCommit = try { git rev-parse HEAD } catch { "unknown" }
    $BuildDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    
    # Build main application image
    Write-Info "Building main application image..."
    docker build `
        --target production `
        --build-arg BUILD_DATE="$BuildDate" `
        --build-arg VCS_REF="$GitCommit" `
        --build-arg VERSION="$Version" `
        -t "$DockerRegistry/$ImageName`:$Version" `
        -t "$DockerRegistry/$ImageName`:latest" `
        .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build main application image"
        exit 1
    }
    
    # Build development image if needed
    if ($Environment -eq "local") {
        Write-Info "Building development image..."
        docker build `
            --target development `
            --build-arg BUILD_DATE="$BuildDate" `
            --build-arg VCS_REF="$GitCommit" `
            --build-arg VERSION="$Version" `
            -t "$DockerRegistry/$ImageName`:dev" `
            .
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build development image"
            exit 1
        }
    }
    
    Write-Success "Docker images built successfully"
}

# Push images to registry
function Push-Images {
    if (-not $Push) {
        Write-Info "Skipping image push"
        return
    }
    
    Write-Header "Pushing images to registry..."
    
    # Login to registry if credentials are available
    $RegistryUsername = [Environment]::GetEnvironmentVariable("DOCKER_REGISTRY_USERNAME")
    $RegistryPassword = [Environment]::GetEnvironmentVariable("DOCKER_REGISTRY_PASSWORD")
    
    if ($RegistryUsername -and $RegistryPassword) {
        $RegistryPassword | docker login $DockerRegistry -u $RegistryUsername --password-stdin
    }
    
    # Push images
    docker push "$DockerRegistry/$ImageName`:$Version"
    docker push "$DockerRegistry/$ImageName`:latest"
    
    if ($Environment -eq "local") {
        docker push "$DockerRegistry/$ImageName`:dev"
    }
    
    Write-Success "Images pushed to registry"
}

# Create backup
function New-Backup {
    if ($SkipBackup) {
        Write-Warning "Skipping backup as requested"
        return
    }
    
    Write-Header "Creating backup before deployment..."
    
    $BackupDir = "$AegisDir\backups\$Timestamp"
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    
    # Backup database
    $PostgresHost = [Environment]::GetEnvironmentVariable("POSTGRES_HOST")
    if ($PostgresHost) {
        Write-Info "Backing up PostgreSQL database..."
        $env:PGPASSWORD = [Environment]::GetEnvironmentVariable("POSTGRES_PASSWORD")
        pg_dump `
            -h $PostgresHost `
            -p ([Environment]::GetEnvironmentVariable("POSTGRES_PORT")) `
            -U ([Environment]::GetEnvironmentVariable("POSTGRES_USER")) `
            -d ([Environment]::GetEnvironmentVariable("POSTGRES_DB")) `
            --no-password `
            --verbose `
            --clean `
            --if-exists `
            > "$BackupDir\postgres_backup.sql"
    }
    
    # Backup Redis data
    $RedisHost = [Environment]::GetEnvironmentVariable("REDIS_HOST")
    if ($RedisHost) {
        Write-Info "Backing up Redis data..."
        redis-cli -h $RedisHost `
            -p ([Environment]::GetEnvironmentVariable("REDIS_PORT")) `
            -a ([Environment]::GetEnvironmentVariable("REDIS_PASSWORD")) `
            --rdb "$BackupDir\redis_backup.rdb"
    }
    
    # Backup configuration files
    Write-Info "Backing up configuration files..."
    Copy-Item -Path "$AegisDir\config" -Destination "$BackupDir\config" -Recurse -Force
    Copy-Item -Path "$AegisDir\.env" -Destination "$BackupDir\.env" -Force
    
    # Backup certificates and keys
    if (Test-Path "$AegisDir\certs") {
        Copy-Item -Path "$AegisDir\certs" -Destination "$BackupDir\certs" -Recurse -Force
    }
    
    if (Test-Path "$AegisDir\keys") {
        Copy-Item -Path "$AegisDir\keys" -Destination "$BackupDir\keys" -Recurse -Force
    }
    
    # Create backup manifest
    $Manifest = @{
        timestamp = $Timestamp
        version = $Version
        environment = $Environment
        git_commit = try { git rev-parse HEAD } catch { "unknown" }
        backup_type = "pre-deployment"
    }
    
    $Manifest | ConvertTo-Json | Set-Content "$BackupDir\manifest.json"
    
    Write-Success "Backup created at $BackupDir"
}

# Deploy to local environment
function Deploy-Local {
    Write-Header "Deploying to local environment..."
    
    Set-Location $AegisDir
    
    # Stop existing services
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
    
    # Start services
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    
    # Wait for services to be ready
    Wait-HealthCheck "http://localhost:8080/health"
    
    Write-Success "Local deployment completed"
}

# Deploy to staging environment
function Deploy-Staging {
    Write-Header "Deploying to staging environment..."
    
    Set-Location $AegisDir
    
    # Update environment variables for staging
    $env:AEGIS_ENV = "staging"
    $env:AEGIS_DEBUG = "true"
    
    # Stop existing services
    docker-compose down
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    Wait-HealthCheck "http://staging.aegis.local:8080/health"
    
    Write-Success "Staging deployment completed"
}

# Deploy to production environment
function Deploy-Production {
    Write-Header "Deploying to production environment..."
    
    # Additional safety checks for production
    if (-not $Rollback) {
        $Confirm = Read-Host "Are you sure you want to deploy to PRODUCTION? (yes/no)"
        if ($Confirm -ne "yes") {
            Write-Info "Production deployment cancelled"
            exit 0
        }
    }
    
    Set-Location $AegisDir
    
    # Update environment variables for production
    $env:AEGIS_ENV = "production"
    $env:AEGIS_DEBUG = "false"
    
    # Rolling update strategy
    Write-Info "Performing rolling update..."
    
    # Scale up new instances
    docker-compose up -d --scale aegis-core=2
    
    # Wait for new instances to be healthy
    Start-Sleep 30
    Wait-HealthCheck "http://production.aegis.local:8080/health"
    
    # Scale down old instances
    docker-compose up -d --scale aegis-core=1
    
    Write-Success "Production deployment completed"
}

# Deploy to Kubernetes
function Deploy-Kubernetes {
    Write-Header "Deploying to Kubernetes..."
    
    $K8sDir = "$AegisDir\k8s"
    
    if (-not (Test-Path $K8sDir)) {
        Write-Error "Kubernetes manifests directory not found: $K8sDir"
        exit 1
    }
    
    # Check cluster connectivity
    try {
        kubectl cluster-info | Out-Null
    }
    catch {
        Write-Error "Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    # Create namespace if it doesn't exist
    kubectl create namespace aegis --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ConfigMaps and Secrets
    if (Test-Path "$K8sDir\configmap.yaml") {
        kubectl apply -f "$K8sDir\configmap.yaml" -n aegis
    }
    
    if (Test-Path "$K8sDir\secrets.yaml") {
        kubectl apply -f "$K8sDir\secrets.yaml" -n aegis
    }
    
    # Deploy using Helm if chart exists
    if (Test-Path "$K8sDir\helm\aegis") {
        Write-Info "Deploying using Helm..."
        helm upgrade --install aegis "$K8sDir\helm\aegis" `
            --namespace aegis `
            --set image.tag="$Version" `
            --set environment="$Environment"
    }
    else {
        # Deploy using kubectl
        Write-Info "Deploying using kubectl..."
        kubectl apply -f "$K8sDir\" -n aegis
    }
    
    # Wait for rollout to complete
    kubectl rollout status deployment/aegis-core -n aegis --timeout=600s
    
    Write-Success "Kubernetes deployment completed"
}

# Wait for health check
function Wait-HealthCheck {
    param(
        [string]$Url,
        [int]$Timeout = $HealthCheckTimeout
    )
    
    $Interval = 10
    $Elapsed = 0
    
    Write-Info "Waiting for health check at $Url..."
    
    while ($Elapsed -lt $Timeout) {
        try {
            $Response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($Response.StatusCode -eq 200) {
                Write-Success "Health check passed"
                return
            }
        }
        catch {
            # Continue trying
        }
        
        Start-Sleep $Interval
        $Elapsed += $Interval
        Write-Info "Health check failed, retrying in ${Interval}s... (${Elapsed}/${Timeout}s)"
    }
    
    Write-Error "Health check timeout after ${Timeout}s"
    exit 1
}

# Rollback deployment
function Invoke-Rollback {
    Write-Header "Rolling back deployment..."
    
    $BackupDirs = Get-ChildItem "$AegisDir\backups" -Directory | Sort-Object Name -Descending
    
    if (-not $BackupDirs) {
        Write-Error "No backup found for rollback"
        exit 1
    }
    
    $BackupDir = $BackupDirs[0].FullName
    Write-Info "Rolling back to backup: $BackupDir"
    
    # Restore configuration
    Copy-Item -Path "$BackupDir\.env" -Destination "$AegisDir\.env" -Force
    Copy-Item -Path "$BackupDir\config\*" -Destination "$AegisDir\config\" -Recurse -Force
    
    # Restore database
    if (Test-Path "$BackupDir\postgres_backup.sql") {
        Write-Info "Restoring PostgreSQL database..."
        $env:PGPASSWORD = [Environment]::GetEnvironmentVariable("POSTGRES_PASSWORD")
        psql -h ([Environment]::GetEnvironmentVariable("POSTGRES_HOST")) `
            -p ([Environment]::GetEnvironmentVariable("POSTGRES_PORT")) `
            -U ([Environment]::GetEnvironmentVariable("POSTGRES_USER")) `
            -d ([Environment]::GetEnvironmentVariable("POSTGRES_DB")) `
            -f "$BackupDir\postgres_backup.sql"
    }
    
    # Restore Redis data
    if (Test-Path "$BackupDir\redis_backup.rdb") {
        Write-Info "Restoring Redis data..."
        redis-cli -h ([Environment]::GetEnvironmentVariable("REDIS_HOST")) `
            -p ([Environment]::GetEnvironmentVariable("REDIS_PORT")) `
            -a ([Environment]::GetEnvironmentVariable("REDIS_PASSWORD")) `
            --rdb "$BackupDir\redis_backup.rdb"
    }
    
    # Restart services
    switch ($Environment) {
        "local" { Deploy-Local }
        "staging" { Deploy-Staging }
        "production" { Deploy-Production }
        "kubernetes" { Deploy-Kubernetes }
    }
    
    Write-Success "Rollback completed"
}

# Post-deployment tasks
function Invoke-PostDeployment {
    Write-Header "Running post-deployment tasks..."
    
    # Run database migrations
    if (Test-Path "$AegisDir\migrations\migrate.py") {
        Write-Info "Running database migrations..."
        python "$AegisDir\migrations\migrate.py"
    }
    
    # Clear caches
    $RedisHost = [Environment]::GetEnvironmentVariable("REDIS_HOST")
    if ($RedisHost) {
        Write-Info "Clearing Redis cache..."
        redis-cli -h $RedisHost `
            -p ([Environment]::GetEnvironmentVariable("REDIS_PORT")) `
            -a ([Environment]::GetEnvironmentVariable("REDIS_PASSWORD")) `
            FLUSHALL
    }
    
    # Send deployment notification
    $SlackWebhook = [Environment]::GetEnvironmentVariable("SLACK_WEBHOOK_URL")
    if ($SlackWebhook) {
        Write-Info "Sending deployment notification..."
        $Body = @{
            text = "AEGIS deployment completed: $Environment v$Version"
        } | ConvertTo-Json
        
        Invoke-RestMethod -Uri $SlackWebhook -Method Post -Body $Body -ContentType "application/json"
    }
    
    Write-Success "Post-deployment tasks completed"
}

# Cleanup old images and containers
function Invoke-Cleanup {
    Write-Header "Cleaning up old images and containers..."
    
    # Remove old images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove old backups (keep last 10)
    $BackupDirs = Get-ChildItem "$AegisDir\backups" -Directory | Sort-Object Name -Descending
    if ($BackupDirs.Count -gt 10) {
        $BackupDirs[10..($BackupDirs.Count-1)] | Remove-Item -Recurse -Force
    }
    
    Write-Success "Cleanup completed"
}

# Main function
function Main {
    if ($Help) {
        Show-Help
        return
    }
    
    Write-Header "AEGIS Framework Deployment"
    Write-Host "Advanced Encrypted Governance and Intelligence System" -ForegroundColor $Colors.Cyan
    Write-Host "=======================================================" -ForegroundColor $Colors.Cyan
    Write-Host "Environment: $Environment" -ForegroundColor $Colors.Cyan
    Write-Host "Version: $Version" -ForegroundColor $Colors.Cyan
    Write-Host "Timestamp: $Timestamp" -ForegroundColor $Colors.Cyan
    Write-Host ""
    
    # Check prerequisites
    Test-Prerequisites
    
    # Handle rollback
    if ($Rollback) {
        Invoke-Rollback
        return
    }
    
    # Run tests
    Invoke-Tests
    
    # Create backup
    New-Backup
    
    # Build images
    Build-Images
    
    # Push images if requested
    Push-Images
    
    # Exit if build-only
    if ($BuildOnly) {
        Write-Success "Build completed, exiting as requested"
        return
    }
    
    # Deploy based on environment
    switch ($Environment) {
        "local" { Deploy-Local }
        "staging" { Deploy-Staging }
        "production" { Deploy-Production }
        "kubernetes" { Deploy-Kubernetes }
        default {
            Write-Error "Unknown environment: $Environment"
            exit 1
        }
    }
    
    # Post-deployment tasks
    Invoke-PostDeployment
    
    # Cleanup
    Invoke-Cleanup
    
    Write-Success "Deployment completed successfully!"
    Write-Host ""
    Write-Host "Environment: $Environment" -ForegroundColor $Colors.Green
    Write-Host "Version: $Version" -ForegroundColor $Colors.Green
    
    $HealthUrl = switch ($Environment) {
        "local" { "http://localhost:8080/health" }
        "staging" { "http://staging.aegis.local:8080/health" }
        "production" { "http://production.aegis.local:8080/health" }
        "kubernetes" { "kubectl get pods -n aegis" }
    }
    Write-Host "Health Check: $HealthUrl" -ForegroundColor $Colors.Green
}

# Run main function
Main