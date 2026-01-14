<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, X-Session-Token');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    exit(0);
}

function readJson($file, $default) {
  if (!file_exists($file)) return $default;
  $raw = file_get_contents($file);
  if ($raw === false || $raw === '') return $default;
  $json = json_decode($raw, true);
  return $json ?: $default;
}

function writeJson($file, $data) {
  return file_put_contents($file, json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
}

function messagesFile($roomId) {
  $safe = preg_replace('/[^a-zA-Z0-9_\-]/', '_', $roomId);
  return __DIR__ . '/../../data/chat_php/messages_' . $safe . '.json';
}

$action = $_GET['action'] ?? $_POST['action'] ?? '';

switch ($action) {
  case 'send_message':
    // SIN VERIFICACIÓN DE TOKEN PARA PROBAR
    $roomId = $_POST['room_id'] ?? '';
    $text = trim($_POST['text'] ?? '');
    $author = trim($_POST['author'] ?? 'usuario');
    
    if ($roomId === '' || $text === '') {
      http_response_code(400);
      echo json_encode([ 'ok' => false, 'error' => 'room_id y text requeridos' ]);
      break;
    }
    
    $file = messagesFile($roomId);
    $messages = readJson($file, []);
    $msg = [
      'id' => uniqid('m_', true),
      'room_id' => $roomId,
      'author' => $author,
      'type' => 'text',
      'text' => $text,
      'enc' => false,
      'ipfs_uri' => null,
      'filename' => null,
      'mime' => null,
      'ts' => time()
    ];
    $messages[] = $msg;
    $result = writeJson($file, $messages);
    
    if ($result === false) {
      http_response_code(500);
      echo json_encode([ 'ok' => false, 'error' => 'write_failed' ]);
      break;
    }
    
    echo json_encode([ 'ok' => true, 'message' => $msg ]);
    break;
    
  case 'messages':
    $roomId = $_GET['room_id'] ?? '';
    if ($roomId === '') {
      http_response_code(400);
      echo json_encode([ 'ok' => false, 'error' => 'room_id requerido' ]);
      break;
    }
    $file = messagesFile($roomId);
    $messages = readJson($file, []);
    echo json_encode([ 'ok' => true, 'messages' => $messages ]);
    break;
    
  default:
    http_response_code(404);
    echo json_encode([ 'ok' => false, 'error' => 'acción no encontrada' ]);
    break;
}
?>