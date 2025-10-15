<?php
/**
 * OpenAGI Secure Chat+ - API Backend
 * Programador Principal: Jose Gómez alias KaseMaster
 * Contacto: kasemaster@aegis-framework.com
 * Versión: 2.0.0
 * Licencia: MIT
 */

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');

$dataDir = __DIR__ . '/../../data/chat_php';
if (!is_dir($dataDir)) {
  @mkdir($dataDir, 0777, true);
}

$roomsFile = $dataDir . '/rooms.json';

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

// Verificar token de sesión emitido por FastAPI
function verifySessionToken($token) {
  if (!$token) return false;
  $url = 'http://localhost:8182/auth/session/verify?token=' . urlencode($token);
  try {
    $resp = @file_get_contents($url);
    if ($resp === false) return false;
    $j = json_decode($resp, true);
    return isset($j['ok']) && $j['ok'] === true;
  } catch (Exception $e) {
    return false;
  }
}

function sessionAddress($token) {
  if (!$token) return null;
  $url = 'http://localhost:8182/auth/session/verify?token=' . urlencode($token);
  try {
    $resp = @file_get_contents($url);
    if ($resp === false) return null;
    $j = json_decode($resp, true);
    if (isset($j['ok']) && $j['ok'] === true) {
      return isset($j['address']) ? strtolower($j['address']) : null;
    }
    return null;
  } catch (Exception $e) {
    return null;
  }
}

// Publicar evento a FastAPI para WebSocket
function publishToFastApi($roomId, $message) {
  $url = 'http://localhost:8182/events/message';
  $payload = json_encode([ 'room_id' => $roomId, 'message' => $message ]);
  $secret = getenv('OPENAGI_EVENT_SECRET');
  if (!$secret || $secret === '') { $secret = 'openagi-dev-secret'; }
  $opts = [
    'http' => [
      'method' => 'POST',
      'header' => "Content-Type: application/json\r\nX-Server-Secret: " . $secret . "\r\n",
      'content' => $payload,
      'timeout' => 2
    ]
  ];
  try {
    @file_get_contents($url, false, stream_context_create($opts));
  } catch (Exception $e) {
    // Silenciar errores de publicación para no romper la API
  }
}

$action = $_GET['action'] ?? $_POST['action'] ?? '';

switch ($action) {
  case 'rooms':
    $rooms = readJson($roomsFile, [
      [ 'id' => 'general', 'name' => 'General', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system' ],
      [ 'id' => 'dev', 'name' => 'Desarrollo', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system' ],
      [ 'id' => 'musica', 'name' => 'Música', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system' ],
      [ 'id' => 'deportes', 'name' => 'Deportes', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system' ],
      [ 'id' => 'cine', 'name' => 'Cine y Series', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system' ],
      [ 'id' => 'tecnologia', 'name' => 'Tecnología', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system' ],
      [ 'id' => 'libros', 'name' => 'Libros', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system' ],
      [ 'id' => 'juegos', 'name' => 'Juegos', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system' ],
      [ 'id' => 'crypto', 'name' => 'Crypto', 'created_at' => time(), 'access' => 'restricted', 'creator_address' => 'system' ],
      [ 'id' => 'soporte', 'name' => 'Soporte', 'created_at' => time(), 'access' => 'restricted', 'creator_address' => 'system' ],
      [ 'id' => 'noticias', 'name' => 'Noticias', 'created_at' => time(), 'access' => 'open', 'creator_address' => 'system' ],
      [ 'id' => 'privacidad', 'name' => 'Privacidad', 'created_at' => time(), 'access' => 'restricted', 'creator_address' => 'system' ],
    ]);
    // rellenar campos por compatibilidad
    foreach ($rooms as &$rr) {
      if (!isset($rr['access'])) $rr['access'] = 'open';
      if (!isset($rr['creator_address'])) $rr['creator_address'] = 'system';
    }
    // Inicializar roles/membresía por defecto para salas restringidas
  foreach ($rooms as $rrx) {
    if (($rrx['access'] ?? 'open') === 'restricted') {
      $rid = $rrx['id'];
      $creator = strtolower($rrx['creator_address'] ?? 'system');
      $mf = membersFile($rid);
      $members = readJson($mf, []);
      if (!in_array($creator, array_map('strtolower', $members))) {
        $members[] = $creator;
        writeJson($mf, $members);
      }
      $rf = rolesFile($rid);
      $roles = readJson($rf, []);
      if (!is_array($roles)) $roles = [];
      if (!isset($roles[$creator])) { $roles[$creator] = 'admin'; writeJson($rf, $roles); }
      // inicializar archivo de sanciones si no existe
      $sf = sanctionsFile($rid);
      $sanctions = readJson($sf, []);
      if (!is_array($sanctions)) $sanctions = [];
      if (!isset($sanctions['muted'])) $sanctions['muted'] = [];
      if (!isset($sanctions['banned'])) $sanctions['banned'] = [];
      if (!isset($sanctions['banned_meta'])) $sanctions['banned_meta'] = [];
      writeJson($sf, $sanctions);
    }
  }
  echo json_encode([ 'ok' => true, 'rooms' => $rooms ]);
  break;

  case 'create_room':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) {
      http_response_code(401);
      echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]);
      break;
    }
    $creator = sessionAddress($token);
    $name = trim($_POST['name'] ?? '');
    $access = strtolower(trim($_POST['access'] ?? 'open'));
    if ($access !== 'open' && $access !== 'restricted') $access = 'open';
    if ($name === '') {
      http_response_code(400);
      echo json_encode([ 'ok' => false, 'error' => 'Nombre requerido' ]);
      break;
    }
    $rooms = readJson($roomsFile, []);
    $id = strtolower(preg_replace('/\s+/', '-', preg_replace('/[^a-zA-Z0-9\s]/', '', $name)));
    if (!$id) $id = 'room_' . substr(md5($name . microtime()), 0, 6);
    // evitar duplicados
    foreach ($rooms as $r) {
      if ($r['id'] === $id) {
        echo json_encode([ 'ok' => true, 'room' => $r ]);
        exit;
      }
    }
    $room = [ 'id' => $id, 'name' => $name, 'created_at' => time(), 'access' => $access, 'creator_address' => $creator ?? 'system' ];
    $rooms[] = $room;
    writeJson($roomsFile, $rooms);
    // inicializar membresía si restringida (creador siempre miembro)
    if ($access === 'restricted' && $creator) {
      $mf = membersFile($id);
      writeJson($mf, [ $creator ]);
      // roles por defecto: creador como admin (owner se infiere por creator_address)
      $rf = rolesFile($id);
      $roles = readJson($rf, []);
      if (!is_array($roles)) $roles = [];
      $roles[strtolower($creator)] = 'admin';
      writeJson($rf, $roles);
    }
    echo json_encode([ 'ok' => true, 'room' => $room ]);
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

  case 'send_message':
    // (bloque duplicado eliminado; ver lógica extendida más abajo)
  case 'send_message':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) {
      http_response_code(401);
      echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]);
      break;
    }
    $address = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    $text = trim($_POST['text'] ?? '');
    $author = trim($_POST['author'] ?? 'usuario');
    $enc = isset($_POST['enc']) ? (($_POST['enc'] === '1' || $_POST['enc'] === 'true') ? true : false) : false;
    $type = $_POST['type'] ?? 'text';
    $ipfsUri = $_POST['ipfs_uri'] ?? null;
    $filename = $_POST['filename'] ?? null;
    $mime = $_POST['mime'] ?? null;
    // Validación según tipo
    if ($roomId === '') {
      http_response_code(400);
      echo json_encode([ 'ok' => false, 'error' => 'room_id requerido' ]);
      break;
    }
    // si la sala es restringida, validar membresía
    $rooms = readJson($roomsFile, []);
    $roomObj = null;
    foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    $accessMode = isset($roomObj['access']) ? $roomObj['access'] : 'open';
  if ($accessMode === 'restricted') {
    $mf = membersFile($roomId);
    $members = readJson($mf, []);
    $isMember = $address ? in_array(strtolower($address), array_map('strtolower', $members)) : false;
    if (!$isMember) {
      http_response_code(403);
      echo json_encode([ 'ok' => false, 'error' => 'no_member' ]);
      break;
    }
    // Rol: si es viewer, no puede enviar
    $rf = rolesFile($roomId);
    $roles = readJson($rf, []);
    $myRole = null;
    if ($address) {
      $addrL = strtolower($address);
      if (is_array($roles)) {
        // roles almacenado como mapa address=>role
        if (isset($roles[$addrL])) $myRole = strtolower($roles[$addrL]);
      }
    }
    if ($myRole === 'viewer') {
      http_response_code(403);
      echo json_encode([ 'ok' => false, 'error' => 'role_denied' ]);
      break;
    }
    // Sanciones: si está muted, no puede enviar
    $sf = sanctionsFile($roomId);
    $sanctions = readJson($sf, []);
    $muted = is_array($sanctions) && isset($sanctions['muted']) && is_array($sanctions['muted']) ? array_map('strtolower', $sanctions['muted']) : [];
    if ($address && in_array(strtolower($address), $muted)) {
      http_response_code(403);
      echo json_encode([ 'ok' => false, 'error' => 'muted' ]);
      break;
    }
  }
    if ($type === 'attachment') {
      if (!$ipfsUri) {
        http_response_code(400);
        echo json_encode([ 'ok' => false, 'error' => 'ipfs_uri requerido para adjunto' ]);
        break;
      }
    } else if ($text === '') {
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
      'type' => $type,
      'text' => $text,
      'enc' => $enc,
      'ipfs_uri' => $ipfsUri,
      'filename' => $filename,
      'mime' => $mime,
      'ts' => time()
    ];
    $messages[] = $msg;
    writeJson($file, $messages);
    // Emitir a FastAPI para retransmisión por WebSocket
    publishToFastApi($roomId, $msg);
    echo json_encode([ 'ok' => true, 'message' => $msg ]);
    break;

  case 'join_room':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) {
      http_response_code(401);
      echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]);
      break;
    }
    $address = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    if ($roomId === '') {
      http_response_code(400);
      echo json_encode([ 'ok' => false, 'error' => 'room_id requerido' ]);
      break;
    }
  $rooms = readJson($roomsFile, []);
  $roomObj = null;
  foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
  if (!$roomObj) {
    http_response_code(404);
    echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]);
    break;
  }
  // bloquear ingreso si sancionado (banned)
  $sf = sanctionsFile($roomId);
  $sanctions = readJson($sf, []);
  $bannedRaw = is_array($sanctions) && isset($sanctions['banned']) && is_array($sanctions['banned']) ? $sanctions['banned'] : [];
  $bannedMeta = is_array($sanctions) && isset($sanctions['banned_meta']) && is_array($sanctions['banned_meta']) ? $sanctions['banned_meta'] : [];
  $now = time();
  $bannedActive = [];
  $changed = false;
  foreach ($bannedRaw as $baddr) {
    $k = strtolower($baddr);
    $meta = isset($bannedMeta[$k]) && is_array($bannedMeta[$k]) ? $bannedMeta[$k] : null;
    $exp = ($meta && isset($meta['expires_at'])) ? intval($meta['expires_at']) : 0;
    if ($exp && $exp <= $now) {
      // ban expirado: limpiar
      $changed = true;
      if (isset($bannedMeta[$k])) { unset($bannedMeta[$k]); }
    } else {
      $bannedActive[] = $k;
    }
  }
  if ($changed) {
    $sanctions['banned'] = $bannedActive;
    $sanctions['banned_meta'] = $bannedMeta;
    writeJson($sf, $sanctions);
  }
  if ($address && in_array(strtolower($address), $bannedActive)) {
    http_response_code(403);
    echo json_encode([ 'ok' => false, 'error' => 'banned' ]);
    break;
  }
  $mf = membersFile($roomId);
  $members = readJson($mf, []);
  if ($address && !in_array(strtolower($address), array_map('strtolower', $members))) {
    $members[] = strtolower($address);
    writeJson($mf, $members);
      // asignar rol por defecto 'member'
      $rf = rolesFile($roomId);
      $roles = readJson($rf, []);
      if (!is_array($roles)) $roles = [];
      $roles[strtolower($address)] = 'member';
      writeJson($rf, $roles);
    }
    echo json_encode([ 'ok' => true, 'members' => $members ]);
    break;

  case 'room_info':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) {
      http_response_code(401);
      echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]);
      break;
    }
    $roomId = $_GET['room_id'] ?? $_POST['room_id'] ?? '';
    if ($roomId === '') {
      http_response_code(400);
      echo json_encode([ 'ok' => false, 'error' => 'room_id requerido' ]);
      break;
    }
    $rooms = readJson($roomsFile, []);
    $roomObj = null;
    foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    if (!$roomObj) {
      http_response_code(404);
      echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]);
      break;
    }
    if (!isset($roomObj['access'])) $roomObj['access'] = 'open';
    if (!isset($roomObj['creator_address'])) $roomObj['creator_address'] = 'system';
    $mf = membersFile($roomId);
    $members = readJson($mf, []);
    // roles como lista de objetos {address, role}
  $rf = rolesFile($roomId);
  $rolesMap = readJson($rf, []);
  $rolesOut = [];
  if (is_array($rolesMap)) {
    foreach ($rolesMap as $addr => $role) {
      $rolesOut[] = [ 'address' => strtolower($addr), 'role' => strtolower($role) ];
    }
  }
  // sanciones
  $sf = sanctionsFile($roomId);
  $sanctions = readJson($sf, []);
  if (!is_array($sanctions)) $sanctions = [];
  if (!isset($sanctions['muted'])) $sanctions['muted'] = [];
  if (!isset($sanctions['banned'])) $sanctions['banned'] = [];
  if (!isset($sanctions['banned_meta'])) $sanctions['banned_meta'] = [];
  // limpiar bans expirados y normalizar
  $now = time();
  $bannedRaw = is_array($sanctions['banned']) ? $sanctions['banned'] : [];
  $bannedMeta = is_array($sanctions['banned_meta']) ? $sanctions['banned_meta'] : [];
  $bannedActive = [];
  $changed = false;
  foreach ($bannedRaw as $baddr) {
    $k = strtolower($baddr);
    $meta = isset($bannedMeta[$k]) && is_array($bannedMeta[$k]) ? $bannedMeta[$k] : null;
    $exp = ($meta && isset($meta['expires_at'])) ? intval($meta['expires_at']) : 0;
    if ($exp && $exp <= $now) {
      $changed = true;
      if (isset($bannedMeta[$k])) { unset($bannedMeta[$k]); }
    } else {
      $bannedActive[] = $k;
    }
  }
  if ($changed) {
    $sanctions['banned'] = $bannedActive;
    $sanctions['banned_meta'] = $bannedMeta;
    writeJson($sf, $sanctions);
  }
  $sanctions['muted'] = array_values(array_map('strtolower', $sanctions['muted']));
  $sanctions['banned'] = array_values(array_map('strtolower', $sanctions['banned']));
  echo json_encode([ 'ok' => true, 'room' => $roomObj, 'members' => $members, 'roles' => $rolesOut, 'sanctions' => $sanctions ]);
  break;

  case 'leave_room':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) {
      http_response_code(401);
      echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]);
      break;
    }
    $address = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    if ($roomId === '') {
      http_response_code(400);
      echo json_encode([ 'ok' => false, 'error' => 'room_id requerido' ]);
      break;
    }
    $rooms = readJson($roomsFile, []);
    $roomObj = null;
    foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    if (!$roomObj) {
      http_response_code(404);
      echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]);
      break;
    }
    $mf = membersFile($roomId);
    $members = readJson($mf, []);
    if ($address) {
      $addrL = strtolower($address);
      $members = array_values(array_filter($members, function($m) use ($addrL) { return strtolower($m) !== $addrL; }));
      writeJson($mf, $members);
      // eliminar rol asignado
      $rf = rolesFile($roomId);
      $roles = readJson($rf, []);
      if (is_array($roles) && isset($roles[$addrL])) {
        unset($roles[$addrL]);
        writeJson($rf, $roles);
      }
    }
    echo json_encode([ 'ok' => true, 'members' => $members ]);
    break;

  case 'add_member':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) {
      http_response_code(401);
      echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]);
      break;
    }
    $requester = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    $addr = strtolower(trim($_POST['address'] ?? ''));
    $role = strtolower(trim($_POST['role'] ?? 'member'));
    if ($roomId === '' || $addr === '') {
      http_response_code(400);
      echo json_encode([ 'ok' => false, 'error' => 'room_id y address requeridos' ]);
      break;
    }
    if (!in_array($role, ['member','viewer','admin'])) $role = 'member';
    $rooms = readJson($roomsFile, []);
    $roomObj = null;
    foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    if (!$roomObj) { http_response_code(404); echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]); break; }
    $access = isset($roomObj['access']) ? $roomObj['access'] : 'open';
    if ($access !== 'restricted') { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'solo salas restringidas' ]); break; }
    $creator = isset($roomObj['creator_address']) ? strtolower($roomObj['creator_address']) : 'system';
    $rf = rolesFile($roomId);
    $roles = readJson($rf, []);
    if (!is_array($roles)) $roles = [];
    $myRole = ($requester && isset($roles[strtolower($requester)])) ? strtolower($roles[strtolower($requester)]) : null;
    // permisos: creador o admin puede añadir
    if (!($requester && ((strtolower($requester) === $creator) || ($myRole === 'admin')))) {
      http_response_code(403);
      echo json_encode([ 'ok' => false, 'error' => 'permiso_denegado' ]);
      break;
    }
    $mf = membersFile($roomId);
    $members = readJson($mf, []);
    if (!in_array($addr, array_map('strtolower', $members))) {
      $members[] = $addr;
      writeJson($mf, $members);
    }
    $roles[$addr] = $role;
    writeJson($rf, $roles);
    echo json_encode([ 'ok' => true, 'members' => $members, 'roles' => array_map(function($a) use ($roles) { return [ 'address' => $a, 'role' => $roles[$a] ]; }, array_keys($roles)) ]);
    break;

  case 'remove_member':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) { http_response_code(401); echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]); break; }
    $requester = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    $addr = strtolower(trim($_POST['address'] ?? ''));
    if ($roomId === '' || $addr === '') { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'room_id y address requeridos' ]); break; }
    $rooms = readJson($roomsFile, []);
    $roomObj = null; foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    if (!$roomObj) { http_response_code(404); echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]); break; }
    $creator = isset($roomObj['creator_address']) ? strtolower($roomObj['creator_address']) : 'system';
    $rf = rolesFile($roomId);
    $roles = readJson($rf, []); if (!is_array($roles)) $roles = [];
    $myRole = ($requester && isset($roles[strtolower($requester)])) ? strtolower($roles[strtolower($requester)]) : null;
    if (!($requester && ((strtolower($requester) === $creator) || ($myRole === 'admin')))) { http_response_code(403); echo json_encode([ 'ok' => false, 'error' => 'permiso_denegado' ]); break; }
    if ($addr === $creator) { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'no_puedes_eliminar_creador' ]); break; }
    $mf = membersFile($roomId);
    $members = readJson($mf, []);
    $members = array_values(array_filter($members, function($m) use ($addr) { return strtolower($m) !== $addr; }));
    writeJson($mf, $members);
    if (isset($roles[$addr])) { unset($roles[$addr]); writeJson($rf, $roles); }
    echo json_encode([ 'ok' => true, 'members' => $members ]);
    break;

  case 'set_role':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) { http_response_code(401); echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]); break; }
    $requester = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    $addr = strtolower(trim($_POST['address'] ?? ''));
    $role = strtolower(trim($_POST['role'] ?? 'member'));
    if ($roomId === '' || $addr === '') { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'room_id y address requeridos' ]); break; }
    if (!in_array($role, ['member','viewer','admin'])) { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'rol inválido' ]); break; }
    $rooms = readJson($roomsFile, []);
    $roomObj = null; foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    if (!$roomObj) { http_response_code(404); echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]); break; }
    $creator = isset($roomObj['creator_address']) ? strtolower($roomObj['creator_address']) : 'system';
    if (!( $requester && (strtolower($requester) === $creator) )) { http_response_code(403); echo json_encode([ 'ok' => false, 'error' => 'solo_creador' ]); break; }
    if ($addr === $creator) { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'creador_no_cambia_rol' ]); break; }
    $mf = membersFile($roomId);
    $members = readJson($mf, []);
    $exists = in_array($addr, array_map('strtolower', $members));
    if (!$exists) { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'miembro_no_encontrado' ]); break; }
    $rf = rolesFile($roomId);
    $roles = readJson($rf, []); if (!is_array($roles)) $roles = [];
    $roles[$addr] = $role;
    writeJson($rf, $roles);
    echo json_encode([ 'ok' => true, 'address' => $addr, 'role' => $role ]);
    break;

  case 'mute_member':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) { http_response_code(401); echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]); break; }
    $requester = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    $addr = strtolower(trim($_POST['address'] ?? ''));
    if ($roomId === '' || $addr === '') { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'room_id y address requeridos' ]); break; }
    $rooms = readJson($roomsFile, []);
    $roomObj = null; foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    if (!$roomObj) { http_response_code(404); echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]); break; }
    $access = isset($roomObj['access']) ? $roomObj['access'] : 'open';
    if ($access !== 'restricted') { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'solo salas restringidas' ]); break; }
    $creator = isset($roomObj['creator_address']) ? strtolower($roomObj['creator_address']) : 'system';
    $rf = rolesFile($roomId); $roles = readJson($rf, []); if (!is_array($roles)) $roles = [];
    $myRole = ($requester && isset($roles[strtolower($requester)])) ? strtolower($roles[strtolower($requester)]) : null;
    if (!($requester && ((strtolower($requester) === $creator) || ($myRole === 'admin')))) { http_response_code(403); echo json_encode([ 'ok' => false, 'error' => 'permiso_denegado' ]); break; }
    $sf = sanctionsFile($roomId); $sanctions = readJson($sf, []); if (!is_array($sanctions)) $sanctions = [];
    if (!isset($sanctions['muted'])) $sanctions['muted'] = [];
    if (!in_array($addr, array_map('strtolower', $sanctions['muted']))) { $sanctions['muted'][] = $addr; }
    writeJson($sf, $sanctions);
    echo json_encode([ 'ok' => true, 'sanctions' => $sanctions ]);
    break;

  case 'unmute_member':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) { http_response_code(401); echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]); break; }
    $requester = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    $addr = strtolower(trim($_POST['address'] ?? ''));
    if ($roomId === '' || $addr === '') { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'room_id y address requeridos' ]); break; }
    $rooms = readJson($roomsFile, []);
    $roomObj = null; foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    if (!$roomObj) { http_response_code(404); echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]); break; }
    $creator = isset($roomObj['creator_address']) ? strtolower($roomObj['creator_address']) : 'system';
    $rf = rolesFile($roomId); $roles = readJson($rf, []); if (!is_array($roles)) $roles = [];
    $myRole = ($requester && isset($roles[strtolower($requester)])) ? strtolower($roles[strtolower($requester)]) : null;
    if (!($requester && ((strtolower($requester) === $creator) || ($myRole === 'admin')))) { http_response_code(403); echo json_encode([ 'ok' => false, 'error' => 'permiso_denegado' ]); break; }
    $sf = sanctionsFile($roomId); $sanctions = readJson($sf, []); if (!is_array($sanctions)) $sanctions = [];
    if (!isset($sanctions['muted'])) $sanctions['muted'] = [];
    $sanctions['muted'] = array_values(array_filter($sanctions['muted'], function($m) use ($addr) { return strtolower($m) !== $addr; }));
    writeJson($sf, $sanctions);
    echo json_encode([ 'ok' => true, 'sanctions' => $sanctions ]);
    break;

  case 'ban_member':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) { http_response_code(401); echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]); break; }
    $requester = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    $addr = strtolower(trim($_POST['address'] ?? ''));
    if ($roomId === '' || $addr === '') { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'room_id y address requeridos' ]); break; }
    $rooms = readJson($roomsFile, []);
    $roomObj = null; foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    if (!$roomObj) { http_response_code(404); echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]); break; }
    $access = isset($roomObj['access']) ? $roomObj['access'] : 'open';
    if ($access !== 'restricted') { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'solo salas restringidas' ]); break; }
    $creator = isset($roomObj['creator_address']) ? strtolower($roomObj['creator_address']) : 'system';
    $rf = rolesFile($roomId); $roles = readJson($rf, []); if (!is_array($roles)) $roles = [];
    $myRole = ($requester && isset($roles[strtolower($requester)])) ? strtolower($roles[strtolower($requester)]) : null;
    if (!($requester && ((strtolower($requester) === $creator) || ($myRole === 'admin')))) { http_response_code(403); echo json_encode([ 'ok' => false, 'error' => 'permiso_denegado' ]); break; }
  // añadir a lista banned
  $sf = sanctionsFile($roomId); $sanctions = readJson($sf, []); if (!is_array($sanctions)) $sanctions = [];
  if (!isset($sanctions['banned'])) $sanctions['banned'] = [];
  if (!isset($sanctions['banned_meta'])) $sanctions['banned_meta'] = [];
  if (!in_array($addr, array_map('strtolower', $sanctions['banned']))) { $sanctions['banned'][] = $addr; }
  // razón y expiración opcionales
  $reason = trim($_POST['reason'] ?? '');
  $expiresRaw = trim($_POST['expires_at'] ?? '');
  $expTs = 0;
  if ($expiresRaw !== '') {
    if (is_numeric($expiresRaw)) { $expTs = intval($expiresRaw); }
    else {
      $ts = strtotime($expiresRaw);
      if ($ts !== false) { $expTs = intval($ts); }
    }
  }
  $sanctions['banned_meta'][$addr] = [ 'reason' => ($reason !== '' ? $reason : null), 'expires_at' => ($expTs > 0 ? $expTs : null) ];
  writeJson($sf, $sanctions);
  // remover de miembros y roles
  $mf = membersFile($roomId); $members = readJson($mf, []);
  $members = array_values(array_filter($members, function($m) use ($addr) { return strtolower($m) !== $addr; }));
  writeJson($mf, $members);
  $rolesMap = readJson($rf, []); if (!is_array($rolesMap)) $rolesMap = [];
  if (isset($rolesMap[$addr])) { unset($rolesMap[$addr]); writeJson($rf, $rolesMap); }
  echo json_encode([ 'ok' => true, 'sanctions' => $sanctions, 'members' => $members ]);
  break;

  case 'unban_member':
    $token = $_SERVER['HTTP_X_SESSION_TOKEN'] ?? '';
    if (!verifySessionToken($token)) { http_response_code(401); echo json_encode([ 'ok' => false, 'error' => 'unauthorized' ]); break; }
    $requester = sessionAddress($token);
    $roomId = $_POST['room_id'] ?? '';
    $addr = strtolower(trim($_POST['address'] ?? ''));
    if ($roomId === '' || $addr === '') { http_response_code(400); echo json_encode([ 'ok' => false, 'error' => 'room_id y address requeridos' ]); break; }
    $rooms = readJson($roomsFile, []);
    $roomObj = null; foreach ($rooms as $r) { if ($r['id'] === $roomId) { $roomObj = $r; break; } }
    if (!$roomObj) { http_response_code(404); echo json_encode([ 'ok' => false, 'error' => 'sala no encontrada' ]); break; }
    $creator = isset($roomObj['creator_address']) ? strtolower($roomObj['creator_address']) : 'system';
    $rf = rolesFile($roomId); $roles = readJson($rf, []); if (!is_array($roles)) $roles = [];
    $myRole = ($requester && isset($roles[strtolower($requester)])) ? strtolower($roles[strtolower($requester)]) : null;
    if (!($requester && ((strtolower($requester) === $creator) || ($myRole === 'admin')))) { http_response_code(403); echo json_encode([ 'ok' => false, 'error' => 'permiso_denegado' ]); break; }
  $sf = sanctionsFile($roomId); $sanctions = readJson($sf, []); if (!is_array($sanctions)) $sanctions = [];
  if (!isset($sanctions['banned'])) $sanctions['banned'] = [];
  if (!isset($sanctions['banned_meta'])) $sanctions['banned_meta'] = [];
  $sanctions['banned'] = array_values(array_filter($sanctions['banned'], function($m) use ($addr) { return strtolower($m) !== $addr; }));
  if (isset($sanctions['banned_meta'][$addr])) { unset($sanctions['banned_meta'][$addr]); }
  writeJson($sf, $sanctions);
  echo json_encode([ 'ok' => true, 'sanctions' => $sanctions ]);
  break;

  default:
    http_response_code(404);
    echo json_encode([ 'ok' => false, 'error' => 'acción no encontrada' ]);
}