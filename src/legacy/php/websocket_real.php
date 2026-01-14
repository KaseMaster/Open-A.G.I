<?php
/**
 * OpenAGI Secure Chat+ - WebSocket Real con Comunicación Bidireccional
 * Programador Principal: Jose Gómez alias KaseMaster
 * Contacto: kasemaster@protonmail.com
 * Versión: 2.1.0 - AEGIS Enhanced WebSocket
 * Licencia: MIT
 */

header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Headers: Cache-Control');

$dataDir = __DIR__ . '/../../data/chat_php';
$eventsFile = $dataDir . '/events.json';
$lastEventFile = $dataDir . '/last_event_id.json';

function readJson($file, $default) {
  if (!file_exists($file)) return $default;
  $raw = file_get_contents($file);
  if ($raw === false || $raw === '') return $default;
  $json = json_decode($raw, true);
  return $json ?: $default;
}

function writeJson($file, $data) {
  file_put_contents($file, json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
}

function messagesFile($roomId) {
  $safe = preg_replace('/[^a-zA-Z0-9_\-]/', '_', $roomId);
  return __DIR__ . '/../../data/chat_php/messages_' . $safe . '.json';
}

function sendSSE($event, $data, $id = null) {
  if ($id !== null) {
    echo "id: $id\n";
  }
  echo "event: $event\n";
  echo "data: " . json_encode($data) . "\n\n";
  
  if (ob_get_level()) {
    ob_flush();
  }
  flush();
}

// Obtener último ID de evento del cliente
$lastEventId = isset($_SERVER['HTTP_LAST_EVENT_ID']) ? (int)$_SERVER['HTTP_LAST_EVENT_ID'] : 0;
$roomId = $_GET['room'] ?? 'general';

// Inicializar archivo de eventos si no existe
if (!file_exists($eventsFile)) {
  writeJson($eventsFile, []);
}

// Obtener último ID de evento del servidor
$lastServerEventId = readJson($lastEventFile, ['id' => 0])['id'];

// Enviar evento de conexión
sendSSE('connected', [
  'status' => 'connected',
  'room' => $roomId,
  'server_time' => time(),
  'last_event_id' => $lastServerEventId
], ++$lastServerEventId);

// Actualizar último ID de evento
writeJson($lastEventFile, ['id' => $lastServerEventId]);

// Función para detectar nuevos mensajes
function getNewMessages($roomId, $lastCheck) {
  $messages = readJson(messagesFile($roomId), []);
  $newMessages = [];
  
  foreach ($messages as $message) {
    if ($message['ts'] > $lastCheck) {
      $newMessages[] = $message;
    }
  }
  
  return $newMessages;
}

// Función para detectar cambios en salas
function getRoomUpdates($lastCheck) {
  $roomsFile = __DIR__ . '/../../data/chat_php/rooms.json';
  $rooms = readJson($roomsFile, []);
  $updates = [];
  
  foreach ($rooms as $room) {
    if (isset($room['updated_at']) && $room['updated_at'] > $lastCheck) {
      $updates[] = [
        'type' => 'room_updated',
        'room' => $room
      ];
    }
  }
  
  return $updates;
}

// Función para detectar usuarios conectados
function getActiveUsers($roomId) {
  $sessionsFile = __DIR__ . '/../../data/chat_php/sessions.json';
  $sessions = readJson($sessionsFile, []);
  $activeUsers = [];
  $now = time();
  
  foreach ($sessions as $session) {
    // Considerar activo si la última actividad fue hace menos de 5 minutos
    if (($now - $session['last_activity']) < 300) {
      $activeUsers[] = [
        'wallet_address' => $session['wallet_address'],
        'last_activity' => $session['last_activity']
      ];
    }
  }
  
  return $activeUsers;
}

$lastCheck = time();
$heartbeatInterval = 30; // 30 segundos
$maxDuration = 300; // 5 minutos máximo de conexión
$startTime = time();

// Loop principal del WebSocket
while ((time() - $startTime) < $maxDuration) {
  // Verificar si la conexión sigue activa
  if (connection_aborted()) {
    break;
  }
  
  // Detectar nuevos mensajes
  $newMessages = getNewMessages($roomId, $lastCheck);
  foreach ($newMessages as $message) {
    sendSSE('new_message', [
      'room_id' => $roomId,
      'message' => $message
    ], ++$lastServerEventId);
  }
  
  // Detectar actualizaciones de salas
  $roomUpdates = getRoomUpdates($lastCheck);
  foreach ($roomUpdates as $update) {
    sendSSE('room_update', $update, ++$lastServerEventId);
  }
  
  // Enviar lista de usuarios activos cada 60 segundos
  if ((time() - $startTime) % 60 === 0) {
    $activeUsers = getActiveUsers($roomId);
    sendSSE('active_users', [
      'room_id' => $roomId,
      'users' => $activeUsers,
      'count' => count($activeUsers)
    ], ++$lastServerEventId);
  }
  
  // Heartbeat cada 30 segundos
  if ((time() - $startTime) % $heartbeatInterval === 0) {
    sendSSE('heartbeat', [
      'server_time' => time(),
      'uptime' => time() - $startTime,
      'room' => $roomId
    ], ++$lastServerEventId);
  }
  
  // Actualizar último check
  $lastCheck = time();
  
  // Actualizar último ID de evento en el servidor
  writeJson($lastEventFile, ['id' => $lastServerEventId]);
  
  // Esperar 1 segundo antes del siguiente ciclo
  sleep(1);
}

// Enviar evento de desconexión
sendSSE('disconnected', [
  'reason' => 'timeout',
  'duration' => time() - $startTime,
  'server_time' => time()
], ++$lastServerEventId);

// Actualizar último ID de evento final
writeJson($lastEventFile, ['id' => $lastServerEventId]);
?>