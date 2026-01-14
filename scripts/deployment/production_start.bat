@echo off
REM AEGIS Framework - Script de Inicio de Producción (Windows)
REM Script para iniciar todos los componentes en producción

setlocal enabledelayedexpansion

REM Configuración
set "AEGIS_HOME=%~dp0"
set "LOG_DIR=%AEGIS_HOME%logs"
set "PID_DIR=%AEGIS_HOME%pids"
set "CONFIG_FILE=%AEGIS_HOME%production_config_v3.json"

REM Colores (si está disponible)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Funciones de utilidad
:log_info
echo %BLUE%[INFO]%NC% %~1
exit /b

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
exit /b

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
exit /b

:log_error
echo %RED%[ERROR]%NC% %~1
exit /b

:check_requirements
call :log_info "Verificando requisitos del sistema..."

REM Verificar Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    call :log_error "Python no está instalado o no está en PATH"
    exit /b 1
)

REM Verificar versión de Python
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo Python version: !python_version!

REM Verificar módulos de Python
set "modules=gunicorn uvicorn flask fastapi redis"
for %%m in (%modules%) do (
    python -c "import %%m" 2>nul
    if !errorlevel! neq 0 (
        call :log_error "Módulo Python faltante: %%m"
        call :log_info "Instala con: pip install %%m"
        exit /b 1
    )
)

call :log_success "Requisitos verificados"
exit /b 0

:create_directories
call :log_info "Creando directorios necesarios..."

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%PID_DIR%" mkdir "%PID_DIR%"

REM Crear logs individuales para cada servicio
if not exist "%LOG_DIR%\node.log" echo. > "%LOG_DIR%\node.log"
if not exist "%LOG_DIR%\api.log" echo. > "%LOG_DIR%\api.log"
if not exist "%LOG_DIR%\dashboard.log" echo. > "%LOG_DIR%\dashboard.log"
if not exist "%LOG_DIR%\admin.log" echo. > "%LOG_DIR%\admin.log"
if not exist "%LOG_DIR%\deploy.log" echo. > "%LOG_DIR%\deploy.log"

call :log_success "Directorios creados"
exit /b 0

:check_config
call :log_info "Verificando configuración..."

if not exist "%CONFIG_FILE%" (
    call :log_error "Archivo de configuración no encontrado: %CONFIG_FILE%"
    call :log_info "Crea production_config_v3.json con la configuración de producción"
    exit /b 1
)

REM Validar JSON (simple)
python -c "import json; json.load(open('%CONFIG_FILE%'))" 2>nul
if !errorlevel! neq 0 (
    call :log_error "Archivo de configuración JSON inválido"
    exit /b 1
)

call :log_success "Configuración válida"
exit /b 0

:start_service
set "service_name=%~1"
set "port=%~2"
set "workers=%~3"
set "module=%~4"
set "server_type=%~5"
set "log_file=%LOG_DIR%\%service_name%.log"
set "pid_file=%PID_DIR%\%service_name%.pid"

call :log_info "Iniciando %service_name% en puerto %port%..."

REM Verificar si ya está corriendo
if exist "%pid_file%" (
    REM Verificar si el proceso está activo
    set /p pid=<"%pid_file%"
    tasklist /FI "PID eq !pid!" 2>nul | find "!pid!" >nul
    if !errorlevel! equ 0 (
        call :log_warning "%service_name% ya está corriendo (PID: !pid!)"
        exit /b 0
    )
)

REM Construir comando
if "%server_type%"=="gunicorn" (
    set "cmd=python -m gunicorn --bind 127.0.0.1:%port% --workers %workers% --worker-class sync --max-requests 1000 --timeout 30 --keepalive 2 --log-level info --preload --daemon --pid "%pid_file%" --access-logfile "%log_file%" --error-logfile "%log_file%" %module%"
) else (
    set "cmd=python -m uvicorn %module% --host 127.0.0.1 --port %port% --workers %workers% --log-level info"
)

REM Iniciar servicio en segundo plano
start "" /B cmd /c "!cmd! > "%log_file%" 2>&1"

REM Esperar un momento y verificar
ping -n 3 127.0.0.1 >nul

REM Verificar si se inició correctamente (simplificado)
tasklist | find "python" >nul
if !errorlevel! equ 0 (
    REM Obtener PID del proceso recién iniciado
    for /f "tokens=2" %%i in ('tasklist ^| find "python"') do (
        echo %%i > "%pid_file%"
        call :log_success "%service_name% iniciado (PID: %%i)"
        exit /b 0
    )
) else (
    call :log_error "%service_name% falló al iniciar"
    exit /b 1
)

:start_all_services
call :log_info "Iniciando todos los servicios AEGIS..."

REM Leer configuración (simplificado)
for /f "tokens=2 delims=:" %%i in ('python -c "import json; c=json.load(open('%CONFIG_FILE%')); print(c['node']['port'])" 2^>nul') do set node_port=%%i
for /f "tokens=2 delims=:" %%i in ('python -c "import json; c=json.load(open('%CONFIG_FILE%')); print(c['api']['port'])" 2^>nul') do set api_port=%%i
for /f "tokens=2 delims=:" %%i in ('python -c "import json; c=json.load(open('%CONFIG_FILE%')); print(c['dashboard']['port'])" 2^>nul') do set dashboard_port=%%i
for /f "tokens=2 delims=:" %%i in ('python -c "import json; c=json.load(open('%CONFIG_FILE%')); print(c['admin']['port'])" 2^>nul') do set admin_port=%%i

REM Valores por defecto si no se pueden leer
if not defined node_port set node_port=8080
if not defined api_port set api_port=8000
if not defined dashboard_port set dashboard_port=3000
if not defined admin_port set admin_port=8081

REM Iniciar servicios
call :start_service "aegis-node" %node_port% 4 "node:app" "gunicorn"
call :start_service "aegis-api" %api_port% 4 "api:app" "gunicorn"
call :start_service "aegis-dashboard" %dashboard_port% 2 "dashboard:app" "uvicorn"
call :start_service "aegis-admin" %admin_port% 2 "admin:app" "uvicorn"

call :log_success "Todos los servicios iniciados"
exit /b 0

:stop_service
set "service_name=%~1"
set "pid_file=%PID_DIR%\%service_name%.pid"

if exist "%pid_file%" (
    set /p pid=<"%pid_file%"
    call :log_info "Deteniendo %service_name% (PID: !pid!)..."
    
    REM Intentar terminación graceful
    taskkill /PID !pid! /T /F >nul 2>nul
    
    if !errorlevel! equ 0 (
        del "%pid_file%"
        call :log_success "%service_name% detenido"
    ) else (
        call :log_warning "No se pudo detener %service_name%"
    )
) else (
    call :log_warning "%service_name% no está corriendo"
)
exit /b 0

:stop_all_services
call :log_info "Deteniendo todos los servicios..."

set services=aegis-node aegis-api aegis-dashboard aegis-admin
for %%s in (%services%) do (
    call :stop_service %%s
)

call :log_success "Todos los servicios detenidos"
exit /b 0

:show_status
call :log_info "Estado de servicios AEGIS:"

set services=aegis-node aegis-api aegis-dashboard aegis-admin
set running_count=0

for %%s in (%services%) do (
    set "status=🔴 DETENIDO"
    set "pid="
    
    if exist "%PID_DIR%\%%s.pid" (
        set /p pid=<"%PID_DIR%\%%s.pid"
        tasklist /FI "PID eq !pid!" 2>nul | find "!pid!" >nul
        if !errorlevel! equ 0 (
            set "status=🟢 CORRIENDO (PID: !pid!)"
            set /a running_count+=1
        )
    )
    
    echo   !status! %%s
)

echo.
echo 📊 Resumen: !running_count!/4 servicios activos
exit /b 0

:show_logs
set "service=%~1"
set "log_file=%LOG_DIR%\%service%.log"

if exist "%log_file%" (
    call :log_info "Mostrando logs de %service% (ultimas 50 lineas):"
    if exist "%SystemRoot%\System32\more.com" (
        more +50 "%log_file%"
    ) else (
        type "%log_file%"
    )
) else (
    call :log_error "No se encontro log para %service%"
)
exit /b 0

:show_help
echo Uso: %~nx0 {start^|stop^|restart^|status^|logs^|help}
echo.
echo Comandos:
echo   start   - Iniciar todos los servicios
echo   stop    - Detener todos los servicios
echo   restart - Reiniciar todos los servicios
echo   status  - Mostrar estado de los servicios
echo   logs    - Mostrar logs [servicio]
echo   help    - Mostrar esta ayuda
echo.
echo Servicios disponibles:
echo   aegis-node, aegis-api, aegis-dashboard, aegis-admin
echo.
echo Ejemplos:
echo   %~nx0 start
echo   %~nx0 logs aegis-node
echo   %~nx0 restart
exit /b 0

:main
REM Verificar que estamos en el directorio correcto
if not exist "production_config_v3.json" (
    call :log_error "Este script debe ejecutarse desde el directorio raiz de AEGIS"
    call :log_info "Asegurate de estar en el directorio que contiene production_config_v3.json"
    exit /b 1
)

REM Ejecutar función según argumento
if "%1"=="start" (
    call :check_requirements
    if !errorlevel! neq 0 exit /b 1
    call :create_directories
    call :check_config
    call :start_all_services
    call :show_status
    call :log_success "Despliegue de produccion completado"
    call :log_info "URLs de acceso:"
    call :log_info "  • Node: http://127.0.0.1:8080"
    call :log_info "  • API: http://127.0.0.1:8000"
    call :log_info "  • Dashboard: http://127.0.0.1:3000"
    call :log_info "  • Admin: http://127.0.0.1:8081"
) else if "%1"=="stop" (
    call :stop_all_services
) else if "%1"=="restart" (
    call :stop_all_services
    timeout /t 2 /nobreak >nul
    call :check_requirements
    if !errorlevel! neq 0 exit /b 1
    call :create_directories
    call :check_config
    call :start_all_services
    call :show_status
) else if "%1"=="status" (
    call :show_status
) else if "%1"=="logs" (
    if "%2"=="" (
        call :log_error "Especifica un servicio: %~nx0 logs [servicio]"
        echo Servicios: aegis-node, aegis-api, aegis-dashboard, aegis-admin
        exit /b 1
    )
    call :show_logs %2
) else if "%1"=="help" (
    call :show_help
) else (
    call :log_error "Comando desconocido: %1"
    call :show_help
    exit /b 1
)

exit /b 0

REM Ejecutar función principal
call :main %*