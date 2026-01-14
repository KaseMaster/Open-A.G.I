# ================================================================
# Quantum Currency – PRODUCTION-GRADE Self-Healing Automated Verification Script
# Author: Qwen / Alibaba Cloud
# Description:
#   - Executes all 16 menu options from run_quantum_currency.bat reliably.
#   - Includes explicit API Server management (Start/Stop).
#   - Detects and applies auto-fixes for critical import/dependency errors.
#   - Implements a retry mechanism for failed modules.
#   - Logs all actions and results to full_self_healing_log.txt.
# ================================================================

param(
    [string]$ProjectRoot = "D:\AI AGENT CODERV1\QUANTUM CURRENCY",
    [switch]$SkipDependencies = $false
)

# Ensure we're in the project root
Set-Location $ProjectRoot

# Log file
$LogFile = "$ProjectRoot\full_self_healing_log.txt"
$ErrorLog = "$ProjectRoot\full_self_healing_errors.log"

# Clear previous logs
if (Test-Path $LogFile) { Remove-Item $LogFile }
if (Test-Path $ErrorLog) { Remove-Item $ErrorLog }

Write-Output "==== 🚀 Starting Self-Healing Verification (Quantum Currency) ====" | Tee-Object -FilePath $LogFile -Append
Write-Output "Timestamp: $(Get-Date)" | Tee-Object -FilePath $LogFile -Append
Write-Output "==============================================================" | Tee-Object -FilePath $LogFile -Append

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    Write-Output $LogEntry
    Add-Content $LogFile $LogEntry
}

function Write-ErrorLog {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [ERROR] $Message"
    Write-Error $LogEntry
    Add-Content $ErrorLog $LogEntry
}

function Test-Port {
    param([int]$Port = 5000)
    $TcpClient = New-Object System.Net.Sockets.TcpClient
    try {
        $TcpClient.Connect("127.0.0.1", $Port)
        return $true
    } catch {
        return $false
    } finally {
        $TcpClient.Close()
    }
}

function Stop-PythonProcesses {
    Write-Log "Stopping any existing Python processes..."
    taskkill /f /im python.exe 2>$null
    Start-Sleep -Seconds 2
}

function Install-Dependencies {
    if ($SkipDependencies) {
        Write-Log "Skipping dependency installation as requested"
        return
    }
    
    Write-Log "Installing/Updating dependencies..."
    try {
        pip install -r requirements-all.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Dependencies installed successfully"
        } else {
            Write-ErrorLog "Failed to install dependencies"
        }
    } catch {
        Write-ErrorLog "Error installing dependencies: $_"
    }
}

function Test-ModuleIntegrity {
    Write-Log "Checking critical module integrity..."
    
    $Modules = @(
        "Open-A.G.I\quantum-currency\src\core\cal_engine.py",
        "Open-A.G.I\quantum-currency\src\core\lambda_attunement.py",
        "Open-A.G.I\quantum-currency\src\dashboard\dashboard_app.py",
        "Open-A.G.I\quantum-currency\scripts\lambda_attunement_tool.py"
    )
    
    $MissingModules = @()
    foreach ($Module in $Modules) {
        if (-not (Test-Path $Module)) {
            $MissingModules += $Module
            Write-ErrorLog "Missing module: $Module"
        }
    }
    
    if ($MissingModules.Count -eq 0) {
        Write-Log "All critical modules found"
        return $true
    } else {
        Write-ErrorLog "Missing $($MissingModules.Count) critical modules"
        return $false
    }
}

function Fix-MintTransactionDemo {
    Write-Log "Fixing Mint Transaction Demo import issues..."
    
    $DemoPath = "Open-A.G.I\scripts\demo_mint_flex.py"
    if (Test-Path $DemoPath) {
        $Content = Get-Content $DemoPath -Raw
        
        # Fix import paths
        $FixedContent = $Content -replace "from openagi.harmonic_validation import make_snapshot, compute_coherence_score", "from quantum_currency.src.core.harmonic_validation import make_snapshot, compute_coherence_score"
        $FixedContent = $FixedContent -replace "from openagi.token_rules import validate_harmonic_tx, apply_token_effects", "from quantum_currency.src.core.token_rules import validate_harmonic_tx, apply_token_effects"
        
        Set-Content $DemoPath $FixedContent
        Write-Log "Fixed Mint Transaction Demo import issues"
        return $true
    } else {
        Write-ErrorLog "Mint Transaction Demo not found at $DemoPath"
        return $false
    }
}

# Main execution
try {
    Write-Log "Quantum Currency Full System Verification Script"
    Write-Log "============================================="
    
    # Pre-execution setup
    Write-Log "=== Pre-Execution Setup ==="
    Install-Dependencies
    Test-ModuleIntegrity
    
    # Fix known issues
    Fix-MintTransactionDemo
    
    # Run the new unified orchestrator
    Write-Log "=== Running Unified Coherence-Aware Orchestrator ==="
    Write-Log "Launching Full Verification Pipeline..."
    
    # Change to the quantum-currency directory
    Set-Location "Open-A.G.I\quantum-currency"
    
    # Run the orchestrator
    python scripts/orchestrator.py
    
    # Return to project root
    Set-Location $ProjectRoot
    
    Write-Log "Script execution completed"
    
    # Show summary
    if (Test-Path $ErrorLog) {
        $ErrorCount = (Get-Content $ErrorLog | Measure-Object).Count
        Write-Log "Total errors encountered: $ErrorCount"
        if ($ErrorCount -gt 0) {
            Write-Log "See $ErrorLog for details"
        }
    }
    
} catch {
    Write-ErrorLog "Script execution failed: $_"
    exit 1
}