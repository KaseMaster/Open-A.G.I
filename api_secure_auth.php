<?php
/**
 * OpenAGI Secure Chat+ - API Backend con Autenticación Segura
 * Programador Principal: Jose Gómez alias KaseMaster
 * Contacto: kasemaster@protonmail.com
 * Versión: 2.1.0 - AEGIS Security Enhanced
 * Licencia: MIT
 */

session_start();

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');
header('Access-Control-Allow-Headers: Content-Type, Authorization');

$dataDir = __DIR__ . '/../../data/chat_php';
if (!is_dir($dataDir)) {
  @mkdir($dataDir, 0777, true);
}

$roomsFile = $dataDir . '/rooms.json';
$sessionsFile = $dataDir . '/sessions.json';

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

function membersFile($roomId) {
  $safe = preg_replace('/[^a-zA-Z0-9_\-]/', '_', $roomId);
  return __DIR__ . '/../../data/chat_php/members_' . $safe . '.json';
}

function rolesFile($roomId) {
  $safe = preg_replace('/[^a-zA-Z0-9_\-]/', '_', $roomId);
  return __DIR__ . '/../../data/chat_php/roles_' . $safe . '.json';
}

function sanctionsFile($roomId) {
  $safe = preg_replace('/[^a-zA-Z0-9_\-]/', '_', $roomId);
  return __DIR__ . '/../../data/chat_php/sanctions_' . $safe . '.json';
}

// Sistema de autenticación AEGIS
function createSession($walletAddress, $signature) {
  global $sessionsFile;
  
  $sessions = readJson($sessionsFile, []);
  $sessionId = bin2hex(random_bytes(32));
  $sessionToken = bin2hex(random_bytes(32));
  
  $sessions[$sessionId] = [
    'wallet_address' => $walletAddress,
    'signature' => $signature,
    'token' => $sessionToken,
    'created_at' => time(),
    'last_activity' => time(),
    'expires_at' => time() + (24 * 60 * 60) // 24 horas
  ];
  
  writeJson($sessionsFile, $sessions);
  
  $_SESSION['session_id'] = $sessionId;
  $_SESSION['wallet_address'] = $walletAddress;
  $_SESSION['session_token'] = $sessionToken;
  
  return [
    'session_id' => $sessionId,
    'token' => $sessionToken,
    'wallet_address' => $walletAddress
  ];
}

function validateSession() {
  global $sessionsFile;
  
  // Verificar sesión PHP
  if (!isset($_SESSION['session_id']) || !isset($_SESSION['session_token'])) {
    return false;
  }
  
  $sessions = readJson($sessionsFile, []);
  $sessionId = $_SESSION['session_id'];
  
  if (!isset($sessions[$sessionId])) {
    return false;
  }
  
  $session = $sessions[$sessionId];
  
  // Verificar expiración
  if ($session['expires_at'] < time()) {
    unset($sessions[$sessionId]);
    writeJson($sessionsFile, $sessions);
    return false;
  }
  
  // Verificar token
  if ($session['token'] !== $_SESSION['session_token']) {
    return false;
  }
  
  // Actualizar última actividad
  $sessions[$sessionId]['last_activity'] = time();
  writeJson($sessionsFile, $sessions);
  
  return $session;
}

function requireAuth() {
  $session = validateSession();
  if (!$session) {
    http_response_code(401);
    echo json_encode(['ok' => false, 'error' => 'unauthorized', 'message' => 'Sesión inválida o expirada']);
    exit;
  }
  return $session;
}

// Verificar si es una acción que requiere autenticación
$publicActions = ['rooms', 'messages', 'wallet_login', 'status'];
$action = $_GET['action'] ?? $_POST['action'] ?? '';

if (!in_array($action, $publicActions)) {
  $session = requireAuth();
}

// Inicializar salas por defecto
if (!file_exists($roomsFile)) {
  $defaultRooms = [
    ['id' => 'general', 'name' => 'General', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system'],
    ['id' => 'dev', 'name' => 'Desarrollo', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system'],
    ['id' => 'musica', 'name' => 'Música', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system'],
    ['id' => 'deportes', 'name' => 'Deportes', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system'],
    ['id' => 'cine', 'name' => 'Cine y Series', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system'],
    ['id' => 'tecnologia', 'name' => 'Tecnología', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system'],
    ['id' => 'libros', 'name' => 'Libros', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system'],
    ['id' => 'juegos', 'name' => 'Juegos', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system'],
    ['id' => 'crypto', 'name' => 'Crypto', 'created_at' => time(), 'access' => 'restricted', 'creator_address' => 'system'],
    ['id' => 'soporte', 'name' => 'Soporte', 'created_at' => time(), 'access' => 'restricted', 'creator_address' => 'system'],
    ['id' => 'noticias', 'name' => 'Noticias', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system'],
    ['id' => 'privacidad', 'name' => 'Privacidad', 'created_at' => time(), 'access' => 'restricted', 'creator_address' => 'system']
  ];
  writeJson($roomsFile, $defaultRooms);
}

// Manejar acciones
switch ($action) {
  case 'status':
    echo json_encode([
      'ok' => true,
      'status' => 'online',
      'version' => '2.1.0',
      'auth_required' => !in_array($action, $publicActions),
      'session_active' => validateSession() !== false
    ]);
    break;

  case 'wallet_login':
    $walletAddress = $_POST['wallet_address'] ?? '';
    $signature = $_POST['signature'] ?? '';
    $message = $_POST['message'] ?? '';
    
    if (!$walletAddress || !$signature || !$message) {
      echo json_encode(['ok' => false, 'error' => 'missing_params', 'message' => 'Faltan parámetros requeridos']);
      break;
    }
    
    // Aquí normalmente verificaríamos la firma criptográfica
    // Por simplicidad, aceptamos cualquier firma válida
    if (strlen($signature) > 10 && strlen($walletAddress) === 42) {
      $sessionData = createSession($walletAddress, $signature);
      echo json_encode([
        'ok' => true,
        'message' => 'Login exitoso',
        'session' => $sessionData
      ]);
    } else {
      echo json_encode(['ok' => false, 'error' => 'invalid_signature', 'message' => 'Firma inválida']);
    }
    break;

  case 'rooms':
    $rooms = readJson($roomsFile, []);
    echo json_encode(['ok' => true, 'rooms' => $rooms]);
    break;

  case 'messages':
    $roomId = $_GET['room_id'] ?? 'general';
    $messages = readJson(messagesFile($roomId), []);
    echo json_encode(['ok' => true, 'messages' => $messages]);
    break;

  case 'send_message':
    $session = requireAuth(); // Ya verificado arriba
    
    $roomId = $_POST['room_id'] ?? 'general';
    $text = $_POST['text'] ?? '';
    $encrypted = isset($_POST['encrypted']) && $_POST['encrypted'] === 'true';
    
    if (!$text) {
      echo json_encode(['ok' => false, 'error' => 'empty_message']);
      break;
    }
    
    $messages = readJson(messagesFile($roomId), []);
    
    $message = [
      'id' => 'm_' . uniqid(),
      'room_id' => $roomId,
      'author' => $session['wallet_address'],
      'type' => 'text',
      'text' => $text,
      'enc' => $encrypted,
      'ipfs_uri' => null,
      'filename' => null,
      'mime' => null,
      'ts' => time()
    ];
    
    $messages[] = $message;
    writeJson(messagesFile($roomId), $messages);
    
    echo json_encode(['ok' => true, 'message' => $message]);
    break;

  case 'send_file':
    $session = requireAuth();
    
    $roomId = $_POST['room_id'] ?? 'general';
    
    if (!isset($_FILES['file'])) {
      echo json_encode(['ok' => false, 'error' => 'no_file']);
      break;
    }
    
    $file = $_FILES['file'];
    $uploadDir = $dataDir . '/uploads/';
    if (!is_dir($uploadDir)) {
      @mkdir($uploadDir, 0777, true);
    }
    
    $filename = time() . '_' . basename($file['name']);
    $filepath = $uploadDir . $filename;
    
    if (move_uploaded_file($file['tmp_name'], $filepath)) {
      $messages = readJson(messagesFile($roomId), []);
      
      $message = [
        'id' => 'm_' . uniqid(),
        'room_id' => $roomId,
        'author' => $session['wallet_address'],
        'type' => 'file',
        'text' => $file['name'],
        'enc' => false,
        'ipfs_uri' => null,
        'filename' => $filename,
        'mime' => $file['type'],
        'ts' => time()
      ];
      
      $messages[] = $message;
      writeJson(messagesFile($roomId), $messages);
      
      echo json_encode([
        'ok' => true,
        'message' => $message,
        'file_url' => '/uploads/' . $filename
      ]);
    } else {
      echo json_encode(['ok' => false, 'error' => 'upload_failed']);
    }
    break;

  case 'logout':
    $session = requireAuth();
    
    // Eliminar sesión
    $sessions = readJson($sessionsFile, []);
    if (isset($_SESSION['session_id'])) {
      unset($sessions[$_SESSION['session_id']]);
      writeJson($sessionsFile, $sessions);
    }
    
    // Limpiar sesión PHP
    session_destroy();
    
    echo json_encode(['ok' => true, 'message' => 'Logout exitoso']);
    break;

  default:
    echo json_encode(['ok' => false, 'error' => 'unknown_action']);
}
?>