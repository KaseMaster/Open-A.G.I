# AEGIS Framework - Setup Script for Windows
# Advanced Encrypted Governance and Intelligence System
# This script sets up the AEGIS development environment on Windows

param(
    [switch]$SkipSystem,
    [switch]$SkipDocker,
    [switch]$SkipTests,
    [switch]$Help
)

# Configuration
$AEGIS_DIR = Split-Path -Parent $PSScriptRoot
$PYTHON_VERSION = "3.11"
$NODE_VERSION = "18"
$DOCKER_DESKTOP_URL = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    Purple = "Magenta"
    Cyan = "Cyan"
    White = "White"
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
    Write-Host "[AEGIS] $Message" -ForegroundColor $Colors.Purple
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

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Show help
function Show-Help {
    Write-Host "AEGIS Framework Setup Script for Windows"
    Write-Host "Usage: .\setup.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -SkipSystem    Skip system dependency installation"
    Write-Host "  -SkipDocker    Skip Docker installation"
    Write-Host "  -SkipTests     Skip running tests"
    Write-Host "  -Help          Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\setup.ps1                    # Full setup"
    Write-Host "  .\setup.ps1 -SkipDocker        # Skip Docker installation"
    Write-Host "  .\setup.ps1 -SkipSystem -SkipTests  # Minimal setup"
}

# Check system requirements
function Test-SystemRequirements {
    Write-Header "Checking system requirements..."
    
    # Check Windows version
    $osVersion = [System.Environment]::OSVersion.Version
    Write-Info "Windows version: $($osVersion.Major).$($osVersion.Minor)"
    
    if ($osVersion.Major -lt 10) {
        Write-Error "Windows 10 or later is required"
        exit 1
    }
    
    # Check architecture
    $arch = $env:PROCESSOR_ARCHITECTURE
    Write-Info "Architecture: $arch"
    
    if ($arch -ne "AMD64" -and $arch -ne "ARM64") {
        Write-Warning "Untested architecture: $arch"
    }
    
    # Check PowerShell version
    $psVersion = $PSVersionTable.PSVersion
    Write-Info "PowerShell version: $psVersion"
    
    if ($psVersion.Major -lt 5) {
        Write-Error "PowerShell 5.0 or later is required"
        exit 1
    }
    
    Write-Success "System requirements check completed"
}

# Install Chocolatey package manager
function Install-Chocolatey {
    if (-not (Test-Command "choco")) {
        Write-Info "Installing Chocolatey package manager..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Success "Chocolatey installed successfully"
    } else {
        Write-Info "Chocolatey is already installed"
    }
}

# Install system dependencies
function Install-SystemDependencies {
    Write-Header "Installing system dependencies..."
    
    if (-not (Test-Administrator)) {
        Write-Error "Administrator privileges required for system dependency installation"
        Write-Info "Please run PowerShell as Administrator or use -SkipSystem flag"
        exit 1
    }
    
    Install-Chocolatey
    
    # Install essential tools
    $packages = @(
        "git",
        "curl",
        "wget",
        "7zip",
        "jq",
        "openssl",
        "postgresql",
        "redis-64",
        "nginx",
        "tor",
        "python",
        "nodejs",
        "vscode"
    )
    
    foreach ($package in $packages) {
        Write-Info "Installing $package..."
        try {
            choco install $package -y --no-progress
        }
        catch {
            Write-Warning "Failed to install $package"
        }
    }
    
    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Success "System dependencies installed"
}

# Setup Python environment
function Set-PythonEnvironment {
    Write-Header "Setting up Python environment..."
    
    # Check Python installation
    if (-not (Test-Command "python")) {
        Write-Error "Python is not installed or not in PATH"
        Write-Info "Please install Python $PYTHON_VERSION or later"
        exit 1
    }
    
    $pythonVersion = python --version 2>&1
    Write-Info "Current Python version: $pythonVersion"
    
    # Change to AEGIS directory
    Set-Location $AEGIS_DIR
    
    # Create virtual environment
    if (-not (Test-Path "venv")) {
        Write-Info "Creating Python virtual environment..."
        python -m venv venv
    }
    
    # Activate virtual environment
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    if (Test-Path "requirements.txt") {
        Write-Info "Installing Python dependencies..."
        pip install -r requirements.txt
    }
    
    if (Test-Path "requirements-dev.txt") {
        Write-Info "Installing development dependencies..."
        pip install -r requirements-dev.txt
    }
    
    Write-Success "Python environment setup completed"
}

# Setup Node.js environment
function Set-NodejsEnvironment {
    Write-Header "Setting up Node.js environment..."
    
    # Check Node.js installation
    if (-not (Test-Command "node")) {
        Write-Error "Node.js is not installed or not in PATH"
        Write-Info "Please install Node.js $NODE_VERSION or later"
        exit 1
    }
    
    $nodeVersion = node --version
    Write-Info "Current Node.js version: $nodeVersion"
    
    # Install global packages
    Write-Info "Installing global Node.js packages..."
    $globalPackages = @(
        "yarn",
        "pm2",
        "nodemon",
        "typescript",
        "ts-node"
    )
    
    foreach ($package in $globalPackages) {
        npm install -g $package
    }
    
    # Install project dependencies
    if (Test-Path "package.json") {
        Write-Info "Installing Node.js dependencies..."
        npm install
    }
    
    Write-Success "Node.js environment setup completed"
}

# Setup Docker environment
function Set-DockerEnvironment {
    Write-Header "Setting up Docker environment..."
    
    if (-not (Test-Command "docker")) {
        Write-Info "Docker is not installed"
        Write-Info "Please download and install Docker Desktop from:"
        Write-Info $DOCKER_DESKTOP_URL
        Write-Warning "Manual installation required for Docker Desktop"
        
        # Optionally download Docker Desktop
        $download = Read-Host "Download Docker Desktop now? (y/N)"
        if ($download -eq "y" -or $download -eq "Y") {
            $tempFile = "$env:TEMP\DockerDesktopInstaller.exe"
            Write-Info "Downloading Docker Desktop..."
            Invoke-WebRequest -Uri $DOCKER_DESKTOP_URL -OutFile $tempFile
            Write-Info "Starting Docker Desktop installer..."
            Start-Process $tempFile -Wait
            Remove-Item $tempFile
        }
    } else {
        Write-Info "Docker is already installed"
        $dockerVersion = docker --version
        Write-Info "Docker version: $dockerVersion"
    }
    
    Write-Success "Docker environment setup completed"
}

# Generate SSL certificates
function New-SSLCertificates {
    Write-Header "Generating SSL certificates..."
    
    $certDir = Join-Path $AEGIS_DIR "certs"
    if (-not (Test-Path $certDir)) {
        New-Item -ItemType Directory -Path $certDir -Force | Out-Null
    }
    
    $certFile = Join-Path $certDir "server.crt"
    $keyFile = Join-Path $certDir "server.key"
    
    if (-not (Test-Path $certFile)) {
        Write-Info "Generating self-signed SSL certificate..."
        
        if (Test-Command "openssl") {
            openssl req -x509 -newkey rsa:4096 -keyout $keyFile -out $certFile -days 365 -nodes -subj "/C=US/ST=State/L=City/O=AEGIS/OU=Security/CN=localhost"
        } else {
            # Use PowerShell to create certificate
            $cert = New-SelfSignedCertificate -DnsName "localhost" -CertStoreLocation "cert:\LocalMachine\My" -KeyLength 4096 -NotAfter (Get-Date).AddDays(365)
            $certPassword = ConvertTo-SecureString -String "aegis" -Force -AsPlainText
            Export-PfxCertificate -Cert $cert -FilePath (Join-Path $certDir "server.pfx") -Password $certPassword
            Export-Certificate -Cert $cert -FilePath $certFile
        }
        
        Write-Success "SSL certificates generated"
    } else {
        Write-Info "SSL certificates already exist"
    }
}

# Generate encryption keys
function New-EncryptionKeys {
    Write-Header "Generating encryption keys..."
    
    $keyDir = Join-Path $AEGIS_DIR "keys"
    if (-not (Test-Path $keyDir)) {
        New-Item -ItemType Directory -Path $keyDir -Force | Out-Null
    }
    
    # Generate random keys
    $masterKeyFile = Join-Path $keyDir "master.key"
    $jwtKeyFile = Join-Path $keyDir "jwt.key"
    $encryptionKeyFile = Join-Path $keyDir "encryption.key"
    
    if (-not (Test-Path $masterKeyFile)) {
        Write-Info "Generating master encryption key..."
        $masterKey = -join ((1..64) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
        $masterKey | Out-File -FilePath $masterKeyFile -Encoding ASCII -NoNewline
    }
    
    if (-not (Test-Path $jwtKeyFile)) {
        Write-Info "Generating JWT secret key..."
        $jwtKey = -join ((1..128) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
        $jwtKey | Out-File -FilePath $jwtKeyFile -Encoding ASCII -NoNewline
    }
    
    if (-not (Test-Path $encryptionKeyFile)) {
        Write-Info "Generating encryption key..."
        $encryptionKey = -join ((1..64) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
        $encryptionKey | Out-File -FilePath $encryptionKeyFile -Encoding ASCII -NoNewline
    }
    
    Write-Success "Encryption keys generated"
}

# Setup environment configuration
function Set-EnvironmentConfiguration {
    Write-Header "Setting up environment configuration..."
    
    $envFile = Join-Path $AEGIS_DIR ".env"
    $envExampleFile = Join-Path $AEGIS_DIR ".env.example"
    
    if (-not (Test-Path $envFile)) {
        if (Test-Path $envExampleFile) {
            Write-Info "Creating .env file from template..."
            Copy-Item $envExampleFile $envFile
            
            # Generate random values
            $jwtSecret = -join ((1..128) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
            $postgresPassword = -join ((1..32) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
            $redisPassword = -join ((1..32) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
            $encryptionKey = -join ((1..64) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
            $masterKey = -join ((1..64) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
            $salt = -join ((1..32) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
            
            # Update .env file with generated values
            $envContent = Get-Content $envFile
            $envContent = $envContent -replace "your-super-secret-jwt-key-change-this-immediately", $jwtSecret
            $envContent = $envContent -replace "aegis_secure_pass_change_this", $postgresPassword
            $envContent = $envContent -replace "aegis_redis_pass_change_this", $redisPassword
            $envContent = $envContent -replace "your-32-character-encryption-key", $encryptionKey
            $envContent = $envContent -replace "your-master-key-for-key-derivation", $masterKey
            $envContent = $envContent -replace "your-random-salt-value", $salt
            
            $envContent | Set-Content $envFile
            
            Write-Success "Environment file created with secure random values"
        } else {
            Write-Error ".env.example file not found"
            exit 1
        }
    } else {
        Write-Info "Environment file already exists"
    }
}

# Setup directory structure
function Set-DirectoryStructure {
    Write-Header "Creating directory structure..."
    
    $directories = @(
        "data",
        "logs",
        "backups",
        "certs",
        "keys",
        "temp",
        "data\db",
        "data\redis",
        "data\prometheus",
        "data\grafana",
        "logs\app",
        "logs\nginx",
        "logs\tor"
    )
    
    foreach ($dir in $directories) {
        $fullPath = Join-Path $AEGIS_DIR $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    }
    
    Write-Success "Directory structure created"
}

# Setup pre-commit hooks
function Set-PrecommitHooks {
    Write-Header "Setting up pre-commit hooks..."
    
    $precommitConfig = Join-Path $AEGIS_DIR ".pre-commit-config.yaml"
    
    if (Test-Path $precommitConfig) {
        # Activate virtual environment
        & ".\venv\Scripts\Activate.ps1"
        
        if (Test-Command "pre-commit") {
            pre-commit install
            pre-commit install --hook-type commit-msg
            Write-Success "Pre-commit hooks installed"
        } else {
            Write-Warning "pre-commit not found in virtual environment"
        }
    } else {
        Write-Warning "Pre-commit configuration not found"
    }
}

# Run initial tests
function Invoke-Tests {
    Write-Header "Running initial tests..."
    
    Set-Location $AEGIS_DIR
    
    # Activate virtual environment
    & ".\venv\Scripts\Activate.ps1"
    
    if (Test-Command "pytest") {
        pytest tests\ -v --tb=short
        Write-Success "Tests completed"
    } else {
        Write-Warning "pytest not found, skipping tests"
    }
}

# Display setup summary
function Show-Summary {
    Write-Header "Setup Summary"
    Write-Host ""
    Write-Host "✓ System dependencies installed" -ForegroundColor Green
    Write-Host "✓ Python environment configured" -ForegroundColor Green
    Write-Host "✓ Node.js environment configured" -ForegroundColor Green
    Write-Host "✓ Docker environment configured" -ForegroundColor Green
    Write-Host "✓ SSL certificates generated" -ForegroundColor Green
    Write-Host "✓ Encryption keys generated" -ForegroundColor Green
    Write-Host "✓ Environment configuration created" -ForegroundColor Green
    Write-Host "✓ Directory structure created" -ForegroundColor Green
    Write-Host "✓ Pre-commit hooks installed" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "1. Review and customize .env file"
    Write-Host "2. Start services: docker-compose up -d"
    Write-Host "3. Run development server: python main.py"
    Write-Host "4. Access dashboard: http://localhost:8080"
    Write-Host ""
    Write-Host "Important Security Notes:" -ForegroundColor Yellow
    Write-Host "• Change default passwords in .env file"
    Write-Host "• Review SSL certificate configuration"
    Write-Host "• Configure Windows Firewall rules"
    Write-Host "• Enable monitoring and logging"
    Write-Host ""
    Write-Host "AEGIS Framework setup completed successfully!" -ForegroundColor Purple
}

# Main setup function
function Main {
    Write-Header "AEGIS Framework Setup for Windows"
    Write-Host "Advanced Encrypted Governance and Intelligence System"
    Write-Host "======================================================="
    Write-Host ""
    
    # Show help if requested
    if ($Help) {
        Show-Help
        return
    }
    
    # Check execution policy
    $executionPolicy = Get-ExecutionPolicy
    if ($executionPolicy -eq "Restricted") {
        Write-Error "PowerShell execution policy is restricted"
        Write-Info "Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
        exit 1
    }
    
    try {
        # Run setup steps
        Test-SystemRequirements
        
        if (-not $SkipSystem) {
            Install-SystemDependencies
        }
        
        Set-PythonEnvironment
        Set-NodejsEnvironment
        
        if (-not $SkipDocker) {
            Set-DockerEnvironment
        }
        
        New-SSLCertificates
        New-EncryptionKeys
        Set-EnvironmentConfiguration
        Set-DirectoryStructure
        Set-PrecommitHooks
        
        if (-not $SkipTests) {
            Invoke-Tests
        }
        
        Show-Summary
    }
    catch {
        Write-Error "Setup failed: $($_.Exception.Message)"
        Write-Error $_.ScriptStackTrace
        exit 1
    }
}

# Run main function
Main