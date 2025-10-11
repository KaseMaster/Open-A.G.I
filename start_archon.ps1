Param(
  [switch]$WithoutTor,
  [int]$Port = 8090,
  [string]$TorExe = "C:\\ProgramData\\chocolatey\\bin\\tor.exe",
  [string]$TorrcPath = "$PSScriptRoot\\torrc"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "== Iniciando ARCHON (Dashboard puerto $Port) ==" -ForegroundColor Cyan

# 1) Verificar puerto del dashboard
function Test-PortFree([int]$p){
  try {
    $res = Test-NetConnection -ComputerName 127.0.0.1 -Port $p -WarningAction SilentlyContinue
    return -not $res.TcpTestSucceeded
  } catch { return $true }
}

if(-not (Test-PortFree -p $Port)){
  Write-Host "El puerto $Port está ocupado. Cambia el parámetro -Port o libera el puerto." -ForegroundColor Yellow
  exit 1
}

# 2) Actualizar app_config.json con nuevo puerto
$configPath = Join-Path $PSScriptRoot 'app_config.json'
if(Test-Path $configPath){
  $cfg = Get-Content $configPath -Raw | ConvertFrom-Json
  if(-not $cfg.monitoring){ $cfg | Add-Member -NotePropertyName monitoring -NotePropertyValue (@{}) }
  $cfg.monitoring.dashboard_port = $Port
  ($cfg | ConvertTo-Json -Depth 6) | Set-Content $configPath -Encoding UTF8
  Write-Host "Actualizado app_config.json -> dashboard_port=$Port" -ForegroundColor Green
}

# 3) Si Tor está habilitado, actualizar torrc HiddenServicePort
if(-not $WithoutTor){
  if(Test-Path $TorrcPath){
    $content = Get-Content $TorrcPath -Raw
    # Reemplazar la línea de HiddenServicePort
    $new = $content -replace '(?m)^HiddenServicePort\s+80\s+.*$', "HiddenServicePort 80 127.0.0.1:$Port"
    if($new -ne $content){
      $new | Set-Content $TorrcPath -Encoding UTF8
      Write-Host "Actualizado torrc -> HiddenServicePort 80 127.0.0.1:$Port" -ForegroundColor Green
    } else {
      Write-Host "No se encontró línea HiddenServicePort para actualizar; se mantiene configuración actual." -ForegroundColor Yellow
    }

    # Lanzar Tor en segundo plano
    if(Test-Path $TorExe){
      Write-Host "Lanzando Tor..." -ForegroundColor Cyan
      Start-Process -FilePath $TorExe -ArgumentList "-f `"$TorrcPath`"" -WindowStyle Minimized
      # Esperar listeners 9050/9051
      $timeout = (Get-Date).AddSeconds(15)
      do {
        $s9050 = Test-NetConnection -ComputerName 127.0.0.1 -Port 9050 -WarningAction SilentlyContinue
        $s9051 = Test-NetConnection -ComputerName 127.0.0.1 -Port 9051 -WarningAction SilentlyContinue
        if($s9050.TcpTestSucceeded -and $s9051.TcpTestSucceeded){ break }
        Start-Sleep -Milliseconds 800
      } while((Get-Date) -lt $timeout)
      if(-not ($s9050.TcpTestSucceeded -and $s9051.TcpTestSucceeded)){
        Write-Host "Advertencia: No se detectan listeners de Tor (9050/9051) aún." -ForegroundColor Yellow
      } else { Write-Host "Tor activo en 9050/9051." -ForegroundColor Green }
    } else {
      Write-Host "No se encontró tor.exe en $TorExe. Usa -WithoutTor o corrige la ruta." -ForegroundColor Yellow
    }
  } else {
    Write-Host "No se encontró torrc en $TorrcPath. Usa -WithoutTor o crea el archivo." -ForegroundColor Yellow
  }
}

# 4) Lanzar el nodo (dashboard)
Write-Host "Lanzando nodo ARCHON..." -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList "main.py start-node" -WorkingDirectory $PSScriptRoot
Write-Host "Nodo iniciado. Abre http://127.0.0.1:$Port" -ForegroundColor Green