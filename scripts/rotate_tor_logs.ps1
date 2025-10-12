Param(
    [string]$LogPath = 'G:\OpenAGI_logs\tor.log',
    [string]$ArchiveDir = 'G:\OpenAGI_logs\archive',
    [int]$MaxSizeMB = 50,
    [int]$RetentionDays = 90
)

Write-Verbose "Rotating Tor log if size >= $MaxSizeMB MB: $LogPath"

if (Test-Path -LiteralPath $LogPath) {
    $sizeMB = [math]::Round((Get-Item -LiteralPath $LogPath).Length / 1MB, 2)
    if ($sizeMB -ge $MaxSizeMB) {
        if (-not (Test-Path -LiteralPath $ArchiveDir)) { New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null }
        $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $dest = Join-Path $ArchiveDir "tor_$timestamp.log"
        Write-Output "[Rotate] Size=$sizeMB MB >= $MaxSizeMB MB. Moving to: $dest"
        Move-Item -LiteralPath $LogPath -Destination $dest -Force
        New-Item -ItemType File -Path $LogPath -Force | Out-Null
    } else {
        Write-Output "[Skip] Current size: $sizeMB MB (< $MaxSizeMB MB)"
    }

    # Cleanup old archives
    if (Test-Path -LiteralPath $ArchiveDir) {
        $cutoff = (Get-Date).AddDays(-$RetentionDays)
        Get-ChildItem -LiteralPath $ArchiveDir -File -Filter '*.log' | Where-Object { $_.LastWriteTime -lt $cutoff } | ForEach-Object {
            Write-Output "[Cleanup] Removing old archive: $($_.FullName)"
            Remove-Item -LiteralPath $_.FullName -Force
        }
    }
} else {
    Write-Output "[Info] Log file not found: $LogPath"
}