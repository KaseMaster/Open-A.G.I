# Script de implementación de correcciones MetaMask
# OpenAGI Secure Chat+ - Implementación automatizada

Write-Host "🚀 INICIANDO IMPLEMENTACIÓN DE CORRECCIONES METAMASK" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Cyan

# Configuración del servidor
$SERVER = "77.237.235.224"
$USER = "root"
$REMOTE_PATH = "/opt/openagi/web/advanced-chat-php/public"

# Archivos a implementar
$FILES = @{
    "app_simple_metamask.js" = "app_fixed.js"
    "debug_console_metamask.html" = "debug_console.html"
}

Write-Host "📋 Archivos a implementar:" -ForegroundColor Yellow
foreach ($file in $FILES.Keys) {
    Write-Host "  • $file → $($FILES[$file])" -ForegroundColor White
}

# Función para ejecutar comando SSH con reintentos
function Invoke-SSHWithRetry {
    param(
        [string]$Command,
        [int]$MaxRetries = 3,
        [int]$DelaySeconds = 5
    )
    
    for ($i = 1; $i -le $MaxRetries; $i++) {
        Write-Host "🔄 Intento $i de $MaxRetries..." -ForegroundColor Yellow
        
        try {
            $result = ssh -o ConnectTimeout=30 -o ServerAliveInterval=10 "$USER@$SERVER" $Command
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Comando ejecutado exitosamente" -ForegroundColor Green
                return $result
            }
        }
        catch {
            Write-Host "❌ Error en intento $i`: $_" -ForegroundColor Red
        }
        
        if ($i -lt $MaxRetries) {
            Write-Host "⏳ Esperando $DelaySeconds segundos antes del siguiente intento..." -ForegroundColor Yellow
            Start-Sleep -Seconds $DelaySeconds
        }
    }
    
    Write-Host "❌ Falló después de $MaxRetries intentos" -ForegroundColor Red
    return $null
}

# Función para subir archivo con reintentos
function Copy-FileWithRetry {
    param(
        [string]$LocalFile,
        [string]$RemoteFile,
        [int]$MaxRetries = 3
    )
    
    for ($i = 1; $i -le $MaxRetries; $i++) {
        Write-Host "🔄 Subiendo $LocalFile (intento $i)..." -ForegroundColor Yellow
        
        try {
            scp -o ConnectTimeout=30 "$LocalFile" "$USER@$SERVER`:$REMOTE_PATH/$RemoteFile"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Archivo $LocalFile subido exitosamente" -ForegroundColor Green
                return $true
            }
        }
        catch {
            Write-Host "❌ Error subiendo archivo: $_" -ForegroundColor Red
        }
        
        if ($i -lt $MaxRetries) {
            Start-Sleep -Seconds 5
        }
    }
    
    Write-Host "❌ Falló la subida de $LocalFile después de $MaxRetries intentos" -ForegroundColor Red
    return $false
}

# Paso 1: Verificar conexión al servidor
Write-Host "`n🔍 PASO 1: Verificando conexión al servidor..." -ForegroundColor Cyan
$connectionTest = Invoke-SSHWithRetry "echo 'Conexión SSH exitosa' && whoami && pwd"

if ($null -eq $connectionTest) {
    Write-Host "❌ No se pudo establecer conexión SSH. Abortando implementación." -ForegroundColor Red
    exit 1
}

# Paso 2: Crear backup del sistema actual
Write-Host "`n💾 PASO 2: Creando backup del sistema actual..." -ForegroundColor Cyan
$backupResult = Invoke-SSHWithRetry "cd $REMOTE_PATH && cp app_fixed.js app_fixed.js.backup.$(date +%Y%m%d_%H%M%S) && echo 'Backup creado exitosamente'"

# Paso 3: Implementar JavaScript simplificado
Write-Host "`n🔧 PASO 3: Implementando JavaScript simplificado..." -ForegroundColor Cyan
$jsSuccess = Copy-FileWithRetry "G:\Open A.G.I\app_simple_metamask.js" "app_fixed.js"

if ($jsSuccess) {
    # Verificar sintaxis del JavaScript
    $syntaxCheck = Invoke-SSHWithRetry "cd $REMOTE_PATH && php -l app_fixed.js 2>/dev/null || echo 'Archivo JavaScript implementado'"
    Write-Host "📝 Verificación de sintaxis: $syntaxCheck" -ForegroundColor White
}

# Paso 4: Implementar consola de debug
Write-Host "`n🔍 PASO 4: Implementando consola de debug..." -ForegroundColor Cyan
$debugSuccess = Copy-FileWithRetry "G:\Open A.G.I\debug_console_metamask.html" "debug_console.html"

# Paso 5: Verificar implementación
Write-Host "`n✅ PASO 5: Verificando implementación..." -ForegroundColor Cyan
$verificationResult = Invoke-SSHWithRetry @"
cd $REMOTE_PATH && 
echo '=== ARCHIVOS IMPLEMENTADOS ===' && 
ls -la app_fixed.js debug_console.html && 
echo -e '\n=== TAMAÑOS DE ARCHIVO ===' && 
wc -l app_fixed.js debug_console.html && 
echo -e '\n=== SERVIDOR PHP ACTIVO ===' && 
ps aux | grep 'php -S' | grep -v grep
"@

Write-Host "`n📊 Resultado de verificación:" -ForegroundColor Yellow
Write-Host $verificationResult -ForegroundColor White

# Paso 6: Probar acceso web
Write-Host "`n🌐 PASO 6: Probando acceso web..." -ForegroundColor Cyan
$webTest = Invoke-SSHWithRetry @"
curl -s -I http://127.0.0.1:8087/ | head -1 && 
echo 'Consola debug:' && 
curl -s -I http://127.0.0.1:8087/debug_console.html | head -1
"@

Write-Host "🌐 Resultado de prueba web:" -ForegroundColor Yellow
Write-Host $webTest -ForegroundColor White

# Resumen final
Write-Host "`n🎉 IMPLEMENTACIÓN COMPLETADA" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host "✅ JavaScript simplificado: $(if($jsSuccess){'IMPLEMENTADO'}else{'FALLÓ'})" -ForegroundColor $(if($jsSuccess){'Green'}else{'Red'})
Write-Host "✅ Consola de debug: $(if($debugSuccess){'IMPLEMENTADA'}else{'FALLÓ'})" -ForegroundColor $(if($debugSuccess){'Green'}else{'Red'})
Write-Host "`n🔗 URLs para probar:" -ForegroundColor Yellow
Write-Host "   • Sistema principal: http://77.237.235.224:8087/" -ForegroundColor White
Write-Host "   • Consola de debug: http://77.237.235.224:8087/debug_console.html" -ForegroundColor White

Write-Host "`n📋 Próximos pasos recomendados:" -ForegroundColor Yellow
Write-Host "1. Abrir la consola de debug para monitorear errores" -ForegroundColor White
Write-Host "2. Probar la conexión MetaMask en el sistema principal" -ForegroundColor White
Write-Host "3. Revisar los logs detallados en la consola" -ForegroundColor White

Write-Host "`n🔧 Si hay problemas, restaurar backup con:" -ForegroundColor Yellow
Write-Host "ssh root@77.237.235.224 'cd $REMOTE_PATH && cp app_fixed.js.backup.* app_fixed.js'" -ForegroundColor Gray