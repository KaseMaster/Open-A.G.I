<?php
header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('Access-Control-Allow-Origin: *');

echo "event: connection\n";
echo "data: {\"type\":\"connected\",\"message\":\"WebSocket conectado\"}\n\n";
ob_flush();
flush();

for ($i = 0; $i < 10; $i++) {
    sleep(1);
    echo "event: heartbeat\n";
    echo "data: {\"type\":\"heartbeat\",\"timestamp\":" . time() . "}\n\n";
    ob_flush();
    flush();
}
?>