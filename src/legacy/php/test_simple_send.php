<?php
// Test simple send message
$token = 'openagi_ca578ac9d6450822f1e78f2c1cf1f41b0f8f5afbbfea0d1672ad26c7eccd4d4d';

function verifySessionToken($token) {
  if (!$token) return false;
  $url = 'http://127.0.0.1:8087/auth_service.php?action=verify&token=' . urlencode($token);
  try {
    $resp = @file_get_contents($url);
    if ($resp === false) return false;
    $j = json_decode($resp, true);
    return isset($j['ok']) && $j['ok'] === true;
  } catch (Exception $e) {
    return false;
  }
}

echo "=== TEST SIMPLE SEND MESSAGE ===\n";

// Simular POST data
$_POST['action'] = 'send_message';
$_POST['room_id'] = 'general';
$_POST['text'] = 'Test simple message';
$_POST['author'] = 'TestUser';
$_SERVER['HTTP_X_SESSION_TOKEN'] = $token;

echo "1. Verificando token...\n";
if (!verifySessionToken($token)) {
    echo "ERROR: Token inválido\n";
    exit(1);
}
echo "Token válido\n";

echo "2. Preparando mensaje...\n";
$message = [
    'id' => 'm_' . uniqid(),
    'room_id' => 'general',
    'author' => 'TestUser',
    'type' => 'text',
    'text' => 'Test simple message',
    'enc' => false,
    'ipfs_uri' => null,
    'filename' => null,
    'mime' => null,
    'ts' => time()
];

echo "3. Guardando mensaje...\n";
$file = '/opt/openagi/web/data/chat_php/messages_general.json';
$messages = [];
if (file_exists($file)) {
    $content = file_get_contents($file);
    if ($content) {
        $messages = json_decode($content, true) ?: [];
    }
}

$messages[] = $message;
$result = file_put_contents($file, json_encode($messages, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));

if ($result === false) {
    echo "ERROR: No se pudo guardar el mensaje\n";
    exit(1);
}

echo "Mensaje guardado exitosamente\n";
echo "Resultado: " . json_encode(['ok' => true, 'message_id' => $message['id']]) . "\n";
?>