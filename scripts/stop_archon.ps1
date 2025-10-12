Param(
  [int]$Port = 8090
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "== Deteniendo ARCHON (Dashboard puerto $Port) ==" -ForegroundColor Cyan

# 1) Detener proceso Python que ejecuta main.py start-node
try {
  $procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -like '*main.py start-node*' }
  foreach($p in $procs){
    Write-Host "Deteniendo proceso Python PID=$($p.ProcessId)" -ForegroundColor Yellow
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
  }
} catch { Write-Host "No se pudo consultar procesos Python: $_" -ForegroundColor Red }

# 2) Detener proceso que escucha en el puerto del dashboard (si queda alguno)
try {
  $tcp = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
  foreach($c in $tcp){
    Write-Host "Deteniendo proceso en puerto $Port PID=$($c.OwningProcess)" -ForegroundColor Yellow
    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
  }
} catch { Write-Host "No se pudo consultar conexiones TCP: $_" -ForegroundColor Red }

# 3) Detener Tor
try {
  Get-Process -Name tor -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Deteniendo Tor PID=$($_.Id)" -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
  }
} catch { Write-Host "No se pudo detener Tor: $_" -ForegroundColor Red }

Write-Host "Validando puertos..." -ForegroundColor Cyan
$r1 = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue
$r2 = Test-NetConnection -ComputerName 127.0.0.1 -Port 9050 -WarningAction SilentlyContinue
$r3 = Test-NetConnection -ComputerName 127.0.0.1 -Port 9051 -WarningAction SilentlyContinue

Write-Host "8090 (dashboard): $($r1.TcpTestSucceeded)" -ForegroundColor Green
Write-Host "9050 (Tor SOCKS): $($r2.TcpTestSucceeded)" -ForegroundColor Green
Write-Host "9051 (Tor Control): $($r3.TcpTestSucceeded)" -ForegroundColor Green

Write-Host "== Entorno detenido ==" -ForegroundColor Cyan