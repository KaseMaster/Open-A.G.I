<?php
/**
 * WebSocket Simple para OpenAGI Secure Chat+
 * Implementación básica de WebSocket usando Server-Sent Events
 */

header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Headers: Cache-Control');

// Función para enviar datos SSE
function sendSSE($data, $event = 'message') {
    echo "event: $event\n";
    echo "data: " . json_encode($data) . "\n\n";
    ob_flush();
    flush();
}

// Obtener room_id de la URL
$roomId = $_GET['room'] ?? 'general';
$token = $_GET['token'] ?? '';

// Archivo de mensajes para la sala
$dataDir = __DIR__ . '/../../data/chat_php';
if (!is_dir($dataDir)) {
    @mkdir($dataDir, 0777, true);
}

function messagesFile($roomId) {
    $safe = preg_replace('/[^a-zA-Z0-9_\-]/', '_', $roomId);
    return __DIR__ . '/../../data/chat_php/messages_' . $safe . '.json';
}

function readJson($file, $default) {
    if (!file_exists($file)) return $default;
    $raw = file_get_contents($file);
    if ($raw === false || $raw === '') return $default;
    $json = json_decode($raw, true);
    return $json ?: $default;
}

// Enviar mensaje inicial de conexión
sendSSE([
    'type' => 'connected',
    'room' => $roomId,
    'timestamp' => time(),
    'message' => 'Conectado al chat en tiempo real'
], 'connection');

// Obtener timestamp del último mensaje conocido
$lastCheck = $_GET['since'] ?? 0;
$messagesFile = messagesFile($roomId);

// Loop principal para verificar nuevos mensajes
$maxTime = 30; // 30 segundos máximo
$startTime = time();

while ((time() - $startTime) < $maxTime) {
    $messages = readJson($messagesFile, []);
    
    // Filtrar mensajes nuevos
    $newMessages = array_filter($messages, function($msg) use ($lastCheck) {
        return isset($msg['timestamp']) && $msg['timestamp'] > $lastCheck;
    });
    
    if (!empty($newMessages)) {
        foreach ($newMessages as $message) {
            sendSSE([
                'type' => 'new_message',
                'room' => $roomId,
                'message' => $message
            ], 'message');
            $lastCheck = max($lastCheck, $message['timestamp']);
        }
    }
    
    // Verificar cada 2 segundos
    sleep(2);
}

// Enviar mensaje de desconexión
sendSSE([
    'type' => 'disconnected',
    'room' => $roomId,
    'timestamp' => time()
], 'connection');
?>