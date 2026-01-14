# Script de implementacion de correcciones MetaMask
# OpenAGI Secure Chat+ - Implementacion automatizada

Write-Host "INICIANDO IMPLEMENTACION DE CORRECCIONES METAMASK" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Cyan

# Configuracion del servidor
$SERVER = "77.237.235.224"
$USER = "root"
$REMOTE_PATH = "/opt/openagi/web/advanced-chat-php/public"

# Archivos a implementar
$FILES = @{
    "app_simple_metamask.js" = "app_fixed.js"
    "debug_console_metamask.html" = "debug_console.html"
}

Write-Host "Archivos a implementar:" -ForegroundColor Yellow
foreach ($file in $FILES.Keys) {
    Write-Host "  $file -> $($FILES[$file])" -ForegroundColor White
}

# Funcion para ejecutar comando SSH con reintentos
function Invoke-SSHWithRetry {
    param(
        [string]$Command,
        [int]$MaxRetries = 3,
        [int]$DelaySeconds = 5
    )
    
    for ($i = 1; $i -le $MaxRetries; $i++) {
        Write-Host "Intento $i de $MaxRetries..." -ForegroundColor Yellow
        
        try {
            $result = ssh -o ConnectTimeout=30 -o ServerAliveInterval=10 "$USER@$SERVER" $Command
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Comando ejecutado exitosamente" -ForegroundColor Green
                return $result
            }
        }
        catch {
            Write-Host "Error en intento $i : $_" -ForegroundColor Red
        }
        
        if ($i -lt $MaxRetries) {
            Write-Host "Esperando $DelaySeconds segundos..." -ForegroundColor Yellow
            Start-Sleep -Seconds $DelaySeconds
        }
    }
    
    Write-Host "Fallo despues de $MaxRetries intentos" -ForegroundColor Red
    return $null
}

# Funcion para subir archivo con reintentos
function Copy-FileWithRetry {
    param(
        [string]$LocalFile,
        [string]$RemoteFile,
        [int]$MaxRetries = 3
    )
    
    for ($i = 1; $i -le $MaxRetries; $i++) {
        Write-Host "Subiendo $LocalFile (intento $i)..." -ForegroundColor Yellow
        
        try {
            scp -o ConnectTimeout=30 "$LocalFile" "$USER@$SERVER`:$REMOTE_PATH/$RemoteFile"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Archivo $LocalFile subido exitosamente" -ForegroundColor Green
                return $true
            }
        }
        catch {
            Write-Host "Error subiendo archivo: $_" -ForegroundColor Red
        }
        
        if ($i -lt $MaxRetries) {
            Start-Sleep -Seconds 5
        }
    }
    
    Write-Host "Fallo la subida de $LocalFile despues de $MaxRetries intentos" -ForegroundColor Red
    return $false
}

# Paso 1: Verificar conexion al servidor
Write-Host ""
Write-Host "PASO 1: Verificando conexion al servidor..." -ForegroundColor Cyan
$connectionTest = Invoke-SSHWithRetry "echo 'Conexion SSH exitosa'; whoami; pwd"

if ($null -eq $connectionTest) {
    Write-Host "No se pudo establecer conexion SSH. Abortando implementacion." -ForegroundColor Red
    exit 1
}

# Paso 2: Crear backup del sistema actual
Write-Host ""
Write-Host "PASO 2: Creando backup del sistema actual..." -ForegroundColor Cyan
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupResult = Invoke-SSHWithRetry "cd $REMOTE_PATH; cp app_fixed.js app_fixed.js.backup.$timestamp; echo 'Backup creado exitosamente'"

# Paso 3: Implementar JavaScript simplificado
Write-Host ""
Write-Host "PASO 3: Implementando JavaScript simplificado..." -ForegroundColor Cyan
$jsSuccess = Copy-FileWithRetry "G:\Open A.G.I\app_simple_metamask.js" "app_fixed.js"

if ($jsSuccess) {
    # Verificar que el archivo se subio correctamente
    $fileCheck = Invoke-SSHWithRetry "cd $REMOTE_PATH; ls -la app_fixed.js; echo 'Archivo JavaScript implementado'"
    Write-Host "Verificacion de archivo: $fileCheck" -ForegroundColor White
}

# Paso 4: Implementar consola de debug
Write-Host ""
Write-Host "PASO 4: Implementando consola de debug..." -ForegroundColor Cyan
$debugSuccess = Copy-FileWithRetry "G:\Open A.G.I\debug_console_metamask.html" "debug_console.html"

# Paso 5: Verificar implementacion
Write-Host ""
Write-Host "PASO 5: Verificando implementacion..." -ForegroundColor Cyan
$verificationCmd = "cd $REMOTE_PATH; echo '=== ARCHIVOS IMPLEMENTADOS ==='; ls -la app_fixed.js debug_console.html; echo '=== SERVIDOR PHP ACTIVO ==='; ps aux | grep 'php -S' | grep -v grep"
$verificationResult = Invoke-SSHWithRetry $verificationCmd

Write-Host ""
Write-Host "Resultado de verificacion:" -ForegroundColor Yellow
Write-Host $verificationResult -ForegroundColor White

# Paso 6: Probar acceso web
Write-Host ""
Write-Host "PASO 6: Probando acceso web..." -ForegroundColor Cyan
$webTestCmd = "curl -s -I http://127.0.0.1:8087/ | head -1; echo 'Consola debug:'; curl -s -I http://127.0.0.1:8087/debug_console.html | head -1"
$webTest = Invoke-SSHWithRetry $webTestCmd

Write-Host ""
Write-Host "Resultado de prueba web:" -ForegroundColor Yellow
Write-Host $webTest -ForegroundColor White

# Resumen final
Write-Host ""
Write-Host "IMPLEMENTACION COMPLETADA" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host "JavaScript simplificado: $(if($jsSuccess){'IMPLEMENTADO'}else{'FALLO'})" -ForegroundColor $(if($jsSuccess){'Green'}else{'Red'})
Write-Host "Consola de debug: $(if($debugSuccess){'IMPLEMENTADA'}else{'FALLO'})" -ForegroundColor $(if($debugSuccess){'Green'}else{'Red'})
Write-Host ""
Write-Host "URLs para probar:" -ForegroundColor Yellow
Write-Host "   Sistema principal: http://77.237.235.224:8087/" -ForegroundColor White
Write-Host "   Consola de debug: http://77.237.235.224:8087/debug_console.html" -ForegroundColor White

Write-Host ""
Write-Host "Proximos pasos recomendados:" -ForegroundColor Yellow
Write-Host "1. Abrir la consola de debug para monitorear errores" -ForegroundColor White
Write-Host "2. Probar la conexion MetaMask en el sistema principal" -ForegroundColor White
Write-Host "3. Revisar los logs detallados en la consola" -ForegroundColor White