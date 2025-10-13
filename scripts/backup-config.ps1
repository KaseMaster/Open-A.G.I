# ============================================================================
# AEGIS Framework - Script de Respaldo de Configuración para Windows
# ============================================================================
# Descripción: Script para crear respaldos de la configuración del sistema
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

param(
    [string]$BackupPath = ".\backups",
    [switch]$Restore,
    [string]$RestoreFrom,
    [switch]$List,
    [switch]$Clean,
    [int]$KeepDays = 30,
    [switch]$Help
)

# Configuración de colores
$Colors = @{
    Red = 'Red'
    Green = 'Green'
    Yellow = 'Yellow'
    Blue = 'Blue'
    Cyan = 'Cyan'
    Magenta = 'Magenta'
    White = 'White'
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Show-Help {
    Write-ColorOutput "🛡️  AEGIS Framework - Gestor de Respaldos de Configuración" "Cyan"
    Write-ColorOutput "=========================================================" "Cyan"
    Write-Host ""
    Write-ColorOutput "DESCRIPCIÓN:" "Yellow"
    Write-Host "  Crea y gestiona respaldos de la configuración del sistema AEGIS"
    Write-Host ""
    Write-ColorOutput "USO:" "Yellow"
    Write-Host "  .\backup-config.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput "OPCIONES:" "Yellow"
    Write-Host "  -BackupPath PATH    Directorio para almacenar respaldos (default: .\backups)"
    Write-Host "  -Restore            Restaurar desde un respaldo"
    Write-Host "  -RestoreFrom PATH   Ruta específica del respaldo a restaurar"
    Write-Host "  -List               Listar respaldos disponibles"
    Write-Host "  -Clean              Limpiar respaldos antiguos"
    Write-Host "  -KeepDays N         Días a mantener respaldos (default: 30)"
    Write-Host "  -Help               Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput "EJEMPLOS:" "Yellow"
    Write-Host "  .\backup-config.ps1                                    # Crear respaldo"
    Write-Host "  .\backup-config.ps1 -BackupPath D:\Backups            # Respaldo en ubicación específica"
    Write-Host "  .\backup-config.ps1 -List                             # Listar respaldos"
    Write-Host "  .\backup-config.ps1 -Restore                          # Restaurar último respaldo"
    Write-Host "  .\backup-config.ps1 -RestoreFrom .\backups\backup_20241201_120000.zip"
    Write-Host "  .\backup-config.ps1 -Clean -KeepDays 7                # Limpiar respaldos > 7 días"
    Write-Host ""
    exit 0
}

function Test-Prerequisites {
    Write-ColorOutput "🔍 Verificando prerrequisitos..." "Blue"
    
    # Verificar si estamos en el directorio correcto
    if (-not (Test-Path "main.py")) {
        Write-ColorOutput "❌ Error: No se encontró main.py. Ejecuta este script desde el directorio raíz del proyecto AEGIS." "Red"
        exit 1
    }
    
    Write-ColorOutput "✅ Prerrequisitos verificados" "Green"
}

function New-BackupDirectory {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        try {
            New-Item -ItemType Directory -Path $Path -Force | Out-Null
            Write-ColorOutput "📁 Directorio de respaldos creado: $Path" "Green"
        }
        catch {
            Write-ColorOutput "❌ Error creando directorio de respaldos: $($_.Exception.Message)" "Red"
            exit 1
        }
    }
}

function Get-BackupItems {
    $items = @()
    
    # Archivos de configuración principales
    $configFiles = @(
        ".env",
        "config\app_config.json",
        "config\torrc"
    )
    
    foreach ($file in $configFiles) {
        if (Test-Path $file) {
            $items += @{
                Type = "File"
                Source = $file
                Destination = $file
            }
        }
    }
    
    # Directorios de configuración
    $configDirs = @(
        "config",
        "logs",
        "tor_data"
    )
    
    foreach ($dir in $configDirs) {
        if (Test-Path $dir) {
            $items += @{
                Type = "Directory"
                Source = $dir
                Destination = $dir
            }
        }
    }
    
    # Archivos de dependencias
    $depFiles = @(
        "requirements.txt",
        "dapps\secure-chat\ui\package.json",
        "dapps\secure-chat\ui\package-lock.json",
        "dapps\aegis-token\package.json",
        "dapps\aegis-token\package-lock.json",
        "dapps\aegis-token\hardhat.config.js"
    )
    
    foreach ($file in $depFiles) {
        if (Test-Path $file) {
            $items += @{
                Type = "File"
                Source = $file
                Destination = $file
            }
        }
    }
    
    return $items
}

function New-Backup {
    param([string]$BackupPath)
    
    Write-ColorOutput "🔄 Iniciando proceso de respaldo..." "Blue"
    
    # Crear directorio de respaldos
    New-BackupDirectory -Path $BackupPath
    
    # Generar nombre del respaldo
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupName = "aegis_backup_$timestamp"
    $tempDir = Join-Path $env:TEMP $backupName
    $zipPath = Join-Path $BackupPath "$backupName.zip"
    
    try {
        # Crear directorio temporal
        New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
        
        # Obtener elementos a respaldar
        $items = Get-BackupItems
        
        if ($items.Count -eq 0) {
            Write-ColorOutput "⚠️  No se encontraron elementos para respaldar" "Yellow"
            return
        }
        
        Write-ColorOutput "📦 Copiando $($items.Count) elementos..." "Blue"
        
        # Copiar elementos
        foreach ($item in $items) {
            $sourcePath = $item.Source
            $destPath = Join-Path $tempDir $item.Destination
            $destDir = Split-Path $destPath -Parent
            
            # Crear directorio de destino si no existe
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            
            if ($item.Type -eq "File") {
                Copy-Item -Path $sourcePath -Destination $destPath -Force
                Write-ColorOutput "  ✅ Archivo: $sourcePath" "Green"
            }
            elseif ($item.Type -eq "Directory") {
                Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
                Write-ColorOutput "  ✅ Directorio: $sourcePath" "Green"
            }
        }
        
        # Crear archivo de metadatos
        $metadata = @{
            BackupDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            BackupVersion = "2.0.0"
            SystemInfo = @{
                OS = (Get-CimInstance -ClassName Win32_OperatingSystem).Caption
                PowerShell = $PSVersionTable.PSVersion.ToString()
                ComputerName = $env:COMPUTERNAME
                UserName = $env:USERNAME
            }
            Items = $items
        }
        
        $metadataPath = Join-Path $tempDir "backup_metadata.json"
        $metadata | ConvertTo-Json -Depth 10 | Out-File -FilePath $metadataPath -Encoding UTF8
        
        # Crear archivo ZIP
        Write-ColorOutput "🗜️  Comprimiendo respaldo..." "Blue"
        Compress-Archive -Path "$tempDir\*" -DestinationPath $zipPath -Force
        
        # Limpiar directorio temporal
        Remove-Item -Path $tempDir -Recurse -Force
        
        # Verificar respaldo
        if (Test-Path $zipPath) {
            $backupSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
            Write-ColorOutput "✅ Respaldo creado exitosamente:" "Green"
            Write-ColorOutput "   📁 Archivo: $zipPath" "White"
            Write-ColorOutput "   📊 Tamaño: $backupSize MB" "White"
            Write-ColorOutput "   🕒 Fecha: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "White"
        }
        else {
            Write-ColorOutput "❌ Error: No se pudo crear el archivo de respaldo" "Red"
            exit 1
        }
    }
    catch {
        Write-ColorOutput "❌ Error durante el respaldo: $($_.Exception.Message)" "Red"
        
        # Limpiar en caso de error
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path $zipPath) {
            Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
        }
        
        exit 1
    }
}

function Get-BackupList {
    param([string]$BackupPath)
    
    if (-not (Test-Path $BackupPath)) {
        Write-ColorOutput "⚠️  No se encontró el directorio de respaldos: $BackupPath" "Yellow"
        return @()
    }
    
    $backups = Get-ChildItem -Path $BackupPath -Filter "aegis_backup_*.zip" | Sort-Object LastWriteTime -Descending
    
    return $backups
}

function Show-BackupList {
    param([string]$BackupPath)
    
    Write-ColorOutput "📋 Lista de Respaldos Disponibles" "Cyan"
    Write-ColorOutput "=================================" "Cyan"
    Write-Host ""
    
    $backups = Get-BackupList -BackupPath $BackupPath
    
    if ($backups.Count -eq 0) {
        Write-ColorOutput "⚠️  No se encontraron respaldos en: $BackupPath" "Yellow"
        return
    }
    
    $index = 1
    foreach ($backup in $backups) {
        $size = [math]::Round($backup.Length / 1MB, 2)
        $age = (Get-Date) - $backup.LastWriteTime
        $ageText = if ($age.Days -gt 0) { "$($age.Days) días" } else { "$($age.Hours) horas" }
        
        Write-ColorOutput "$index. $($backup.Name)" "White"
        Write-ColorOutput "   📊 Tamaño: $size MB" "Blue"
        Write-ColorOutput "   🕒 Creado: $($backup.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" "Blue"
        Write-ColorOutput "   ⏰ Antigüedad: $ageText" "Blue"
        Write-Host ""
        
        $index++
    }
}

function Restore-Backup {
    param(
        [string]$BackupPath,
        [string]$RestoreFrom
    )
    
    $backupFile = ""
    
    if ($RestoreFrom) {
        if (Test-Path $RestoreFrom) {
            $backupFile = $RestoreFrom
        }
        else {
            Write-ColorOutput "❌ Error: No se encontró el archivo de respaldo: $RestoreFrom" "Red"
            exit 1
        }
    }
    else {
        # Usar el respaldo más reciente
        $backups = Get-BackupList -BackupPath $BackupPath
        
        if ($backups.Count -eq 0) {
            Write-ColorOutput "❌ Error: No se encontraron respaldos para restaurar" "Red"
            exit 1
        }
        
        $backupFile = $backups[0].FullName
    }
    
    Write-ColorOutput "🔄 Iniciando restauración desde: $backupFile" "Blue"
    
    # Confirmar restauración
    Write-ColorOutput "⚠️  ADVERTENCIA: Esta operación sobrescribirá la configuración actual." "Yellow"
    $confirm = Read-Host "¿Deseas continuar? (s/N)"
    
    if ($confirm -ne "s" -and $confirm -ne "S") {
        Write-ColorOutput "❌ Restauración cancelada por el usuario" "Yellow"
        exit 0
    }
    
    $tempDir = Join-Path $env:TEMP "aegis_restore_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    
    try {
        # Extraer respaldo
        Write-ColorOutput "📦 Extrayendo respaldo..." "Blue"
        Expand-Archive -Path $backupFile -DestinationPath $tempDir -Force
        
        # Verificar metadatos
        $metadataPath = Join-Path $tempDir "backup_metadata.json"
        if (Test-Path $metadataPath) {
            $metadata = Get-Content $metadataPath | ConvertFrom-Json
            Write-ColorOutput "📋 Información del respaldo:" "Blue"
            Write-ColorOutput "   🕒 Fecha: $($metadata.BackupDate)" "White"
            Write-ColorOutput "   🖥️  Sistema: $($metadata.SystemInfo.OS)" "White"
            Write-ColorOutput "   📊 Elementos: $($metadata.Items.Count)" "White"
        }
        
        # Crear respaldo de la configuración actual
        Write-ColorOutput "💾 Creando respaldo de seguridad de la configuración actual..." "Blue"
        $safetyBackupPath = Join-Path $BackupPath "safety_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
        New-Backup -BackupPath (Split-Path $safetyBackupPath -Parent)
        
        # Restaurar archivos
        Write-ColorOutput "🔄 Restaurando configuración..." "Blue"
        
        $restoredItems = 0
        Get-ChildItem -Path $tempDir -Recurse | ForEach-Object {
            if ($_.Name -ne "backup_metadata.json") {
                $relativePath = $_.FullName.Substring($tempDir.Length + 1)
                $targetPath = Join-Path (Get-Location) $relativePath
                $targetDir = Split-Path $targetPath -Parent
                
                if (-not (Test-Path $targetDir)) {
                    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
                }
                
                if ($_.PSIsContainer) {
                    if (-not (Test-Path $targetPath)) {
                        New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
                    }
                }
                else {
                    Copy-Item -Path $_.FullName -Destination $targetPath -Force
                    Write-ColorOutput "  ✅ Restaurado: $relativePath" "Green"
                    $restoredItems++
                }
            }
        }
        
        # Limpiar directorio temporal
        Remove-Item -Path $tempDir -Recurse -Force
        
        Write-ColorOutput "✅ Restauración completada exitosamente" "Green"
        Write-ColorOutput "   📊 Elementos restaurados: $restoredItems" "White"
        Write-ColorOutput "   💾 Respaldo de seguridad: $safetyBackupPath" "White"
        Write-Host ""
        Write-ColorOutput "💡 Recomendación: Reinicia los servicios para aplicar la configuración restaurada" "Yellow"
    }
    catch {
        Write-ColorOutput "❌ Error durante la restauración: $($_.Exception.Message)" "Red"
        
        # Limpiar en caso de error
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        exit 1
    }
}

function Remove-OldBackups {
    param(
        [string]$BackupPath,
        [int]$KeepDays
    )
    
    Write-ColorOutput "🧹 Limpiando respaldos antiguos (> $KeepDays días)..." "Blue"
    
    if (-not (Test-Path $BackupPath)) {
        Write-ColorOutput "⚠️  No se encontró el directorio de respaldos: $BackupPath" "Yellow"
        return
    }
    
    $cutoffDate = (Get-Date).AddDays(-$KeepDays)
    $backups = Get-ChildItem -Path $BackupPath -Filter "aegis_backup_*.zip" | Where-Object { $_.LastWriteTime -lt $cutoffDate }
    
    if ($backups.Count -eq 0) {
        Write-ColorOutput "✅ No se encontraron respaldos antiguos para eliminar" "Green"
        return
    }
    
    Write-ColorOutput "🗑️  Eliminando $($backups.Count) respaldos antiguos:" "Yellow"
    
    $totalSize = 0
    foreach ($backup in $backups) {
        $size = [math]::Round($backup.Length / 1MB, 2)
        $totalSize += $size
        
        Write-ColorOutput "  🗑️  $($backup.Name) ($size MB)" "Red"
        Remove-Item -Path $backup.FullName -Force
    }
    
    Write-ColorOutput "✅ Limpieza completada" "Green"
    Write-ColorOutput "   📊 Archivos eliminados: $($backups.Count)" "White"
    Write-ColorOutput "   💾 Espacio liberado: $([math]::Round($totalSize, 2)) MB" "White"
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function Main {
    if ($Help) {
        Show-Help
    }
    
    Test-Prerequisites
    
    if ($List) {
        Show-BackupList -BackupPath $BackupPath
    }
    elseif ($Clean) {
        Remove-OldBackups -BackupPath $BackupPath -KeepDays $KeepDays
    }
    elseif ($Restore) {
        Restore-Backup -BackupPath $BackupPath -RestoreFrom $RestoreFrom
    }
    else {
        New-Backup -BackupPath $BackupPath
    }
}

# Ejecutar función principal
Main