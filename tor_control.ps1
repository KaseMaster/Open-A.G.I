Param(
    [string]$ControlHost = '127.0.0.1',
    [int]$ControlPort = 9051,
    [string]$CookiePath = 'G:\Open A.G.I\tor_data\control_auth_cookie',
    [string]$Command = 'SIGNAL HUP'
)

function Send-TorCommand {
    param(
        [string]$TargetHost,
        [int]$TargetPort,
        [string]$CookieFile,
        [string]$Cmd
    )

    if (-not (Test-Path -LiteralPath $CookieFile)) {
        throw "Cookie file not found: $CookieFile"
    }

    $cookieBytes = [System.IO.File]::ReadAllBytes($CookieFile)
    $cookieHex = ($cookieBytes | ForEach-Object { $_.ToString('X2') }) -join ''

    $client = New-Object System.Net.Sockets.TcpClient
    $client.Connect($TargetHost, $TargetPort)
    $stream = $client.GetStream()
    $writer = New-Object System.IO.StreamWriter($stream)
    $writer.NewLine = "`r`n"
    $writer.AutoFlush = $true
    $reader = New-Object System.IO.StreamReader($stream)

    # Authenticate using cookie
    $writer.WriteLine("AUTHENTICATE $cookieHex")
    $resp = $reader.ReadLine()
    Write-Output "AUTH RESP: $resp"
    if ($resp -notlike '250*') {
        $client.Close()
        throw "Authentication failed: $resp"
    }

    # Send command (e.g., SIGNAL HUP)
    Write-Output "Sending command: $Cmd"
    $writer.WriteLine($Cmd)
    $resp2 = $reader.ReadLine()
    Write-Output "CMD RESP: $resp2"

    # Quit control connection
    $writer.WriteLine('QUIT')
    $client.Close()
}

try {
    Send-TorCommand -TargetHost $ControlHost -TargetPort $ControlPort -CookieFile $CookiePath -Cmd $Command
} catch {
    Write-Error $_
    exit 1
}