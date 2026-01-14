<?php
header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('Access-Control-Allow-Origin: *');

function sendSSE($data, $event = 'message') {
    echo "event: " . $event . "\n";
    echo "data: " . json_encode($data) . "\n\n";
    ob_flush();
    flush();
}

$room = isset($_GET['room']) ? $_GET['room'] : 'general';

sendSSE(array(
    'type' => 'connected',
    'room' => $room,
    'timestamp' => time(),
    'message' => 'WebSocket conectado'
), 'connection');

for ($i = 0; $i < 30; $i++) {
    sleep(1);
    sendSSE(array(
        'type' => 'heartbeat',
        'room' => $room,
        'timestamp' => time()
    ), 'heartbeat');
}
?>