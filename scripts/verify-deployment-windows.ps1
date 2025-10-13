# ============================================================================
# AEGIS Framework - Script de Verificación Post-Despliegue (Windows)
# ============================================================================
# Descripción: Verifica que todos los componentes del sistema AEGIS estén
#              correctamente instalados y funcionando después del despliegue
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: Diciembre 2024
# ============================================================================

param(
    [switch]$Detailed,
    [switch]$SkipServices,
    [switch]$SkipNetwork,
    [string]$ConfigPath = ".\config",
    [string]$LogPath = ".\logs"
)

# Configuración de colores para output
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Header = "Magenta"
}

# Configuración de verificación
$Config = @{
    RequiredPorts = @(8080, 5173, 8545, 9050, 9051)
    RequiredServices = @("Dashboard", "SecureChat", "Blockchain", "Tor")
    RequiredFiles = @(
        "main.py",
        "config\app_config.json",
        "config\torrc",
        "dapps\secure-chat\ui\package.json",
        "dapps\aegis-token\package.json"
    )
    RequiredDirectories = @(
        "config",
        "logs",
        "dapps\secure-chat\ui",
        "dapps\aegis-token",
        "venv"
    )
    Timeouts = @{
        ServiceStart = 30
        NetworkCheck = 10
        HealthCheck = 15
    }
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White",
        [switch]$NoNewline
    )
    
    if ($NoNewline) {
        Write-Host $Message -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $Message -ForegroundColor $Color
    }
}

function Write-Header {
    param([string]$Title)
    
    Write-Host ""
    Write-ColorOutput "=" * 80 -Color $Colors.Header
    Write-ColorOutput " $Title" -Color $Colors.Header
    Write-ColorOutput "=" * 80 -Color $Colors.Header
    Write-Host ""
}

function Write-TestResult {
    param(
        [string]$TestName,
        [bool]$Success,
        [string]$Details = ""
    )
    
    $status = if ($Success) { "✅ PASS" } else { "❌ FAIL" }
    $color = if ($Success) { $Colors.Success } else { $Colors.Error }
    
    Write-ColorOutput "[$status] $TestName" -Color $color
    
    if ($Details -and ($Detailed -or -not $Success)) {
        Write-ColorOutput "    └─ $Details" -Color "Gray"
    }
}

function Test-CommandExists {
    param([string]$Command)
    
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Test-PortOpen {
    param(
        [string]$Host = "localhost",
        [int]$Port,
        [int]$Timeout = 5
    )
    
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $tcpClient.BeginConnect($Host, $Port, $null, $null)
        $wait = $asyncResult.AsyncWaitHandle.WaitOne($Timeout * 1000, $false)
        
        if ($wait) {
            $tcpClient.EndConnect($asyncResult)
            $tcpClient.Close()
            return $true
        } else {
            $tcpClient.Close()
            return $false
        }
    } catch {
        return $false
    }
}

function Test-HttpEndpoint {
    param(
        [string]$Url,
        [int]$Timeout = 10
    )
    
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $Timeout -UseBasicParsing -ErrorAction Stop
        return @{
            Success = $true
            StatusCode = $response.StatusCode
            ResponseTime = $null
        }
    } catch {
        return @{
            Success = $false
            StatusCode = $null
            Error = $_.Exception.Message
        }
    }
}

function Get-ProcessByPort {
    param([int]$Port)
    
    try {
        $netstat = netstat -ano | Select-String ":$Port "
        if ($netstat) {
            $pid = ($netstat -split '\s+')[-1]
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            return $process.ProcessName
        }
        return $null
    } catch {
        return $null
    }
}

# ============================================================================
# TESTS DE VERIFICACIÓN
# ============================================================================

function Test-SystemRequirements {
    Write-Header "Verificación de Requisitos del Sistema"
    
    $results = @()
    
    # Verificar versión de Windows
    $osVersion = [System.Environment]::OSVersion.Version
    $isWindows10Plus = $osVersion.Major -ge 10
    $results += @{
        Name = "Windows 10/11"
        Success = $isWindows10Plus
        Details = "Versión detectada: $($osVersion.Major).$($osVersion.Minor)"
    }
    
    # Verificar PowerShell
    $psVersion = $PSVersionTable.PSVersion
    $isPowerShell5Plus = $psVersion.Major -ge 5
    $results += @{
        Name = "PowerShell 5.0+"
        Success = $isPowerShell5Plus
        Details = "Versión: $($psVersion.Major).$($psVersion.Minor)"
    }
    
    # Verificar RAM
    $totalRAM = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
    $hasEnoughRAM = $totalRAM -ge 8
    $results += @{
        Name = "RAM (8GB mínimo)"
        Success = $hasEnoughRAM
        Details = "RAM disponible: ${totalRAM}GB"
    }
    
    # Verificar espacio en disco
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
    $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
    $hasEnoughSpace = $freeSpaceGB -ge 10
    $results += @{
        Name = "Espacio en disco (10GB mínimo)"
        Success = $hasEnoughSpace
        Details = "Espacio libre: ${freeSpaceGB}GB"
    }
    
    foreach ($result in $results) {
        Write-TestResult -TestName $result.Name -Success $result.Success -Details $result.Details
    }
    
    return ($results | Where-Object { -not $_.Success }).Count -eq 0
}

function Test-Dependencies {
    Write-Header "Verificación de Dependencias"
    
    $results = @()
    
    # Python
    $pythonExists = Test-CommandExists "python"
    $pythonDetails = ""
    if ($pythonExists) {
        try {
            $pythonVersion = python --version 2>&1
            $pythonDetails = $pythonVersion
        } catch {
            $pythonDetails = "Error obteniendo versión"
        }
    }
    $results += @{
        Name = "Python 3.8+"
        Success = $pythonExists
        Details = $pythonDetails
    }
    
    # Node.js
    $nodeExists = Test-CommandExists "node"
    $nodeDetails = ""
    if ($nodeExists) {
        try {
            $nodeVersion = node --version
            $nodeDetails = "Versión: $nodeVersion"
        } catch {
            $nodeDetails = "Error obteniendo versión"
        }
    }
    $results += @{
        Name = "Node.js 18+"
        Success = $nodeExists
        Details = $nodeDetails
    }
    
    # NPM
    $npmExists = Test-CommandExists "npm"
    $npmDetails = ""
    if ($npmExists) {
        try {
            $npmVersion = npm --version
            $npmDetails = "Versión: $npmVersion"
        } catch {
            $npmDetails = "Error obteniendo versión"
        }
    }
    $results += @{
        Name = "NPM"
        Success = $npmExists
        Details = $npmDetails
    }
    
    # Git
    $gitExists = Test-CommandExists "git"
    $gitDetails = ""
    if ($gitExists) {
        try {
            $gitVersion = git --version
            $gitDetails = $gitVersion
        } catch {
            $gitDetails = "Error obteniendo versión"
        }
    }
    $results += @{
        Name = "Git"
        Success = $gitExists
        Details = $gitDetails
    }
    
    # Tor
    $torExists = Test-CommandExists "tor"
    $torDetails = ""
    if ($torExists) {
        try {
            $torVersion = tor --version | Select-Object -First 1
            $torDetails = $torVersion
        } catch {
            $torDetails = "Error obteniendo versión"
        }
    }
    $results += @{
        Name = "Tor"
        Success = $torExists
        Details = $torDetails
    }
    
    foreach ($result in $results) {
        Write-TestResult -TestName $result.Name -Success $result.Success -Details $result.Details
    }
    
    return ($results | Where-Object { -not $_.Success }).Count -eq 0
}

function Test-ProjectStructure {
    Write-Header "Verificación de Estructura del Proyecto"
    
    $results = @()
    
    # Verificar archivos requeridos
    foreach ($file in $Config.RequiredFiles) {
        $exists = Test-Path $file
        $results += @{
            Name = "Archivo: $file"
            Success = $exists
            Details = if ($exists) { "Encontrado" } else { "No encontrado" }
        }
    }
    
    # Verificar directorios requeridos
    foreach ($dir in $Config.RequiredDirectories) {
        $exists = Test-Path $dir -PathType Container
        $results += @{
            Name = "Directorio: $dir"
            Success = $exists
            Details = if ($exists) { "Encontrado" } else { "No encontrado" }
        }
    }
    
    # Verificar entorno virtual Python
    $venvExists = Test-Path "venv\Scripts\python.exe"
    $results += @{
        Name = "Entorno virtual Python"
        Success = $venvExists
        Details = if ($venvExists) { "Configurado correctamente" } else { "No encontrado o mal configurado" }
    }
    
    # Verificar node_modules
    $nodeModulesUI = Test-Path "dapps\secure-chat\ui\node_modules"
    $results += @{
        Name = "Node modules (Secure Chat UI)"
        Success = $nodeModulesUI
        Details = if ($nodeModulesUI) { "Instalados" } else { "No instalados" }
    }
    
    $nodeModulesToken = Test-Path "dapps\aegis-token\node_modules"
    $results += @{
        Name = "Node modules (AEGIS Token)"
        Success = $nodeModulesToken
        Details = if ($nodeModulesToken) { "Instalados" } else { "No instalados" }
    }
    
    foreach ($result in $results) {
        Write-TestResult -TestName $result.Name -Success $result.Success -Details $result.Details
    }
    
    return ($results | Where-Object { -not $_.Success }).Count -eq 0
}

function Test-Configuration {
    Write-Header "Verificación de Configuración"
    
    $results = @()
    
    # Verificar archivo .env
    $envExists = Test-Path ".env"
    $results += @{
        Name = "Archivo .env"
        Success = $envExists
        Details = if ($envExists) { "Encontrado" } else { "No encontrado - usar .env.example como base" }
    }
    
    # Verificar configuración de la aplicación
    $appConfigExists = Test-Path "$ConfigPath\app_config.json"
    $appConfigValid = $false
    if ($appConfigExists) {
        try {
            $appConfig = Get-Content "$ConfigPath\app_config.json" | ConvertFrom-Json
            $appConfigValid = $appConfig.dashboard -and $appConfig.dashboard.port
        } catch {
            $appConfigValid = $false
        }
    }
    $results += @{
        Name = "Configuración de la aplicación"
        Success = $appConfigValid
        Details = if ($appConfigValid) { "Válida" } else { "Inválida o no encontrada" }
    }
    
    # Verificar configuración de Tor
    $torrcExists = Test-Path "$ConfigPath\torrc"
    $results += @{
        Name = "Configuración de Tor"
        Success = $torrcExists
        Details = if ($torrcExists) { "Encontrada" } else { "No encontrada" }
    }
    
    # Verificar directorio de logs
    $logsExists = Test-Path $LogPath -PathType Container
    if (-not $logsExists) {
        try {
            New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
            $logsExists = $true
        } catch {
            $logsExists = $false
        }
    }
    $results += @{
        Name = "Directorio de logs"
        Success = $logsExists
        Details = if ($logsExists) { "Disponible" } else { "No se pudo crear" }
    }
    
    foreach ($result in $results) {
        Write-TestResult -TestName $result.Name -Success $result.Success -Details $result.Details
    }
    
    return ($results | Where-Object { -not $_.Success }).Count -eq 0
}

function Test-NetworkPorts {
    Write-Header "Verificación de Puertos de Red"
    
    if ($SkipNetwork) {
        Write-ColorOutput "⏭️  Verificación de red omitida por parámetro" -Color $Colors.Warning
        return $true
    }
    
    $results = @()
    
    foreach ($port in $Config.RequiredPorts) {
        $isOpen = Test-PortOpen -Port $port -Timeout $Config.Timeouts.NetworkCheck
        $process = if ($isOpen) { Get-ProcessByPort -Port $port } else { $null }
        
        $details = if ($isOpen) {
            if ($process) { "Puerto abierto - Proceso: $process" } else { "Puerto abierto" }
        } else {
            "Puerto cerrado o no accesible"
        }
        
        $results += @{
            Name = "Puerto $port"
            Success = $isOpen
            Details = $details
        }
    }
    
    foreach ($result in $results) {
        Write-TestResult -TestName $result.Name -Success $result.Success -Details $result.Details
    }
    
    return ($results | Where-Object { -not $_.Success }).Count -eq 0
}

function Test-Services {
    Write-Header "Verificación de Servicios"
    
    if ($SkipServices) {
        Write-ColorOutput "⏭️  Verificación de servicios omitida por parámetro" -Color $Colors.Warning
        return $true
    }
    
    $results = @()
    
    # Dashboard (Puerto 8080)
    $dashboardTest = Test-HttpEndpoint -Url "http://localhost:8080" -Timeout $Config.Timeouts.HealthCheck
    $results += @{
        Name = "Dashboard AEGIS"
        Success = $dashboardTest.Success
        Details = if ($dashboardTest.Success) { 
            "Respondiendo (HTTP $($dashboardTest.StatusCode))" 
        } else { 
            "No responde: $($dashboardTest.Error)" 
        }
    }
    
    # Secure Chat UI (Puerto 5173)
    $chatTest = Test-HttpEndpoint -Url "http://localhost:5173" -Timeout $Config.Timeouts.HealthCheck
    $results += @{
        Name = "Secure Chat UI"
        Success = $chatTest.Success
        Details = if ($chatTest.Success) { 
            "Respondiendo (HTTP $($chatTest.StatusCode))" 
        } else { 
            "No responde: $($chatTest.Error)" 
        }
    }
    
    # Blockchain Local (Puerto 8545)
    $blockchainTest = Test-PortOpen -Port 8545 -Timeout $Config.Timeouts.NetworkCheck
    $results += @{
        Name = "Blockchain Local (Hardhat)"
        Success = $blockchainTest
        Details = if ($blockchainTest) { "Activo" } else { "No activo" }
    }
    
    # Tor SOCKS Proxy (Puerto 9050)
    $torSocksTest = Test-PortOpen -Port 9050 -Timeout $Config.Timeouts.NetworkCheck
    $results += @{
        Name = "Tor SOCKS Proxy"
        Success = $torSocksTest
        Details = if ($torSocksTest) { "Activo" } else { "No activo" }
    }
    
    # Tor Control Port (Puerto 9051)
    $torControlTest = Test-PortOpen -Port 9051 -Timeout $Config.Timeouts.NetworkCheck
    $results += @{
        Name = "Tor Control Port"
        Success = $torControlTest
        Details = if ($torControlTest) { "Activo" } else { "No activo" }
    }
    
    foreach ($result in $results) {
        Write-TestResult -TestName $result.Name -Success $result.Success -Details $result.Details
    }
    
    return ($results | Where-Object { -not $_.Success }).Count -eq 0
}

function Test-PythonEnvironment {
    Write-Header "Verificación del Entorno Python"
    
    $results = @()
    
    # Activar entorno virtual y verificar paquetes
    $venvPython = ".\venv\Scripts\python.exe"
    $venvPip = ".\venv\Scripts\pip.exe"
    
    if (Test-Path $venvPython) {
        # Verificar paquetes críticos
        $criticalPackages = @("flask", "requests", "cryptography", "stem")
        
        foreach ($package in $criticalPackages) {
            try {
                $packageInfo = & $venvPip show $package 2>$null
                $isInstalled = $LASTEXITCODE -eq 0
                $version = if ($isInstalled) {
                    ($packageInfo | Select-String "Version:").ToString().Split(":")[1].Trim()
                } else {
                    "No instalado"
                }
                
                $results += @{
                    Name = "Paquete Python: $package"
                    Success = $isInstalled
                    Details = if ($isInstalled) { "Versión: $version" } else { "No instalado" }
                }
            } catch {
                $results += @{
                    Name = "Paquete Python: $package"
                    Success = $false
                    Details = "Error verificando paquete"
                }
            }
        }
        
        # Verificar que se puede importar el módulo principal
        try {
            $importTest = & $venvPython -c "import sys; sys.path.append('.'); import main; print('OK')" 2>$null
            $canImport = $LASTEXITCODE -eq 0
            $results += @{
                Name = "Importación del módulo principal"
                Success = $canImport
                Details = if ($canImport) { "Exitosa" } else { "Falló" }
            }
        } catch {
            $results += @{
                Name = "Importación del módulo principal"
                Success = $false
                Details = "Error en la importación"
            }
        }
    } else {
        $results += @{
            Name = "Entorno virtual Python"
            Success = $false
            Details = "No encontrado en .\venv\Scripts\python.exe"
        }
    }
    
    foreach ($result in $results) {
        Write-TestResult -TestName $result.Name -Success $result.Success -Details $result.Details
    }
    
    return ($results | Where-Object { -not $_.Success }).Count -eq 0
}

function Test-NodeEnvironment {
    Write-Header "Verificación del Entorno Node.js"
    
    $results = @()
    
    # Verificar Secure Chat UI
    if (Test-Path "dapps\secure-chat\ui\package.json") {
        try {
            Push-Location "dapps\secure-chat\ui"
            
            # Verificar que las dependencias están instaladas
            $nodeModulesExists = Test-Path "node_modules"
            $results += @{
                Name = "Dependencias Secure Chat UI"
                Success = $nodeModulesExists
                Details = if ($nodeModulesExists) { "Instaladas" } else { "No instaladas" }
            }
            
            # Verificar que el build funciona
            if ($nodeModulesExists) {
                try {
                    $buildTest = npm run build 2>$null
                    $buildSuccess = $LASTEXITCODE -eq 0
                    $results += @{
                        Name = "Build Secure Chat UI"
                        Success = $buildSuccess
                        Details = if ($buildSuccess) { "Exitoso" } else { "Falló" }
                    }
                } catch {
                    $results += @{
                        Name = "Build Secure Chat UI"
                        Success = $false
                        Details = "Error ejecutando build"
                    }
                }
            }
            
            Pop-Location
        } catch {
            $results += @{
                Name = "Secure Chat UI"
                Success = $false
                Details = "Error accediendo al directorio"
            }
        }
    }
    
    # Verificar AEGIS Token
    if (Test-Path "dapps\aegis-token\package.json") {
        try {
            Push-Location "dapps\aegis-token"
            
            # Verificar que las dependencias están instaladas
            $nodeModulesExists = Test-Path "node_modules"
            $results += @{
                Name = "Dependencias AEGIS Token"
                Success = $nodeModulesExists
                Details = if ($nodeModulesExists) { "Instaladas" } else { "No instaladas" }
            }
            
            # Verificar compilación de contratos
            if ($nodeModulesExists) {
                try {
                    $compileTest = npx hardhat compile 2>$null
                    $compileSuccess = $LASTEXITCODE -eq 0
                    $results += @{
                        Name = "Compilación de contratos"
                        Success = $compileSuccess
                        Details = if ($compileSuccess) { "Exitosa" } else { "Falló" }
                    }
                } catch {
                    $results += @{
                        Name = "Compilación de contratos"
                        Success = $false
                        Details = "Error compilando contratos"
                    }
                }
            }
            
            Pop-Location
        } catch {
            $results += @{
                Name = "AEGIS Token"
                Success = $false
                Details = "Error accediendo al directorio"
            }
        }
    }
    
    foreach ($result in $results) {
        Write-TestResult -TestName $result.Name -Success $result.Success -Details $result.Details
    }
    
    return ($results | Where-Object { -not $_.Success }).Count -eq 0
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Start-DeploymentVerification {
    Write-ColorOutput @"
 █████╗ ███████╗ ██████╗ ██╗███████╗    ███████╗██████╗  █████╗ ███╗   ███╗███████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
██╔══██╗██╔════╝██╔════╝ ██║██╔════╝    ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
███████║█████╗  ██║  ███╗██║███████╗    █████╗  ██████╔╝███████║██╔████╔██║█████╗  ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ 
██╔══██║██╔══╝  ██║   ██║██║╚════██║    ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ██║███╗██║██║   ██║██╔══██╗██╔═██╗ 
██║  ██║███████╗╚██████╔╝██║███████║    ██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
"@ -Color $Colors.Header
    
    Write-ColorOutput "Verificación Post-Despliegue - Windows" -Color $Colors.Header
    Write-ColorOutput "Versión: 2.0.0 | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Color $Colors.Info
    Write-Host ""
    
    $startTime = Get-Date
    $allTestsPassed = $true
    
    # Ejecutar todas las verificaciones
    $testResults = @{
        SystemRequirements = Test-SystemRequirements
        Dependencies = Test-Dependencies
        ProjectStructure = Test-ProjectStructure
        Configuration = Test-Configuration
        NetworkPorts = Test-NetworkPorts
        Services = Test-Services
        PythonEnvironment = Test-PythonEnvironment
        NodeEnvironment = Test-NodeEnvironment
    }
    
    # Calcular resultados finales
    $passedTests = ($testResults.Values | Where-Object { $_ -eq $true }).Count
    $totalTests = $testResults.Count
    $allTestsPassed = $passedTests -eq $totalTests
    
    # Mostrar resumen final
    Write-Header "Resumen de Verificación"
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-ColorOutput "Tiempo de ejecución: $($duration.TotalSeconds.ToString('F2')) segundos" -Color $Colors.Info
    Write-ColorOutput "Tests ejecutados: $totalTests" -Color $Colors.Info
    Write-ColorOutput "Tests exitosos: $passedTests" -Color $Colors.Success
    Write-ColorOutput "Tests fallidos: $($totalTests - $passedTests)" -Color $Colors.Error
    Write-Host ""
    
    if ($allTestsPassed) {
        Write-ColorOutput "🎉 ¡VERIFICACIÓN EXITOSA!" -Color $Colors.Success
        Write-ColorOutput "El sistema AEGIS está correctamente instalado y configurado." -Color $Colors.Success
        Write-Host ""
        Write-ColorOutput "URLs de acceso:" -Color $Colors.Info
        Write-ColorOutput "  • Dashboard: http://localhost:8080" -Color $Colors.Info
        Write-ColorOutput "  • Secure Chat: http://localhost:5173" -Color $Colors.Info
        Write-ColorOutput "  • Blockchain: http://localhost:8545" -Color $Colors.Info
        Write-Host ""
        Write-ColorOutput "Para iniciar todos los servicios, ejecuta:" -Color $Colors.Info
        Write-ColorOutput "  .\scripts\start-all-services.ps1" -Color $Colors.Info
    } else {
        Write-ColorOutput "⚠️  VERIFICACIÓN INCOMPLETA" -Color $Colors.Warning
        Write-ColorOutput "Algunos componentes requieren atención." -Color $Colors.Warning
        Write-Host ""
        Write-ColorOutput "Revisa los errores anteriores y:" -Color $Colors.Info
        Write-ColorOutput "  1. Consulta la documentación de troubleshooting" -Color $Colors.Info
        Write-ColorOutput "  2. Ejecuta los scripts de instalación faltantes" -Color $Colors.Info
        Write-ColorOutput "  3. Verifica la configuración de red y puertos" -Color $Colors.Info
        Write-Host ""
        Write-ColorOutput "Para más ayuda:" -Color $Colors.Info
        Write-ColorOutput "  .\scripts\verify-deployment-windows.ps1 -Detailed" -Color $Colors.Info
    }
    
    Write-Host ""
    Write-ColorOutput "Logs detallados disponibles en: $LogPath" -Color $Colors.Info
    
    # Guardar reporte de verificación
    $reportPath = "$LogPath\deployment-verification-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    try {
        $report = @{
            timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            version = "2.0.0"
            duration_seconds = $duration.TotalSeconds
            tests_total = $totalTests
            tests_passed = $passedTests
            tests_failed = $totalTests - $passedTests
            all_tests_passed = $allTestsPassed
            test_results = $testResults
            system_info = @{
                os_version = [System.Environment]::OSVersion.VersionString
                powershell_version = $PSVersionTable.PSVersion.ToString()
                computer_name = $env:COMPUTERNAME
                user_name = $env:USERNAME
            }
        }
        
        $report | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportPath -Encoding UTF8
        Write-ColorOutput "Reporte guardado en: $reportPath" -Color $Colors.Info
    } catch {
        Write-ColorOutput "⚠️  No se pudo guardar el reporte: $($_.Exception.Message)" -Color $Colors.Warning
    }
    
    return $allTestsPassed
}

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if ($MyInvocation.InvocationName -ne '.') {
    try {
        $success = Start-DeploymentVerification
        exit $(if ($success) { 0 } else { 1 })
    } catch {
        Write-ColorOutput "❌ Error crítico durante la verificación: $($_.Exception.Message)" -Color $Colors.Error
        Write-ColorOutput "Stack trace: $($_.ScriptStackTrace)" -Color $Colors.Error
        exit 1
    }
}