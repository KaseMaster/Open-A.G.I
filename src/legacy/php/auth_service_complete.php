<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, X-Session-Token');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    exit(0);
}

$action = $_GET['action'] ?? '';

switch ($action) {
    case 'verify':
        $token = $_GET['token'] ?? '';
        if (strlen($token) > 10) {
            $address = '0x' . substr(hash('sha256', $token), 0, 40);
            echo json_encode([
                'ok' => true,
                'valid' => true,
                'address' => $address,
                'expires' => time() + 3600
            ]);
        } else {
            echo json_encode(['ok' => false, 'valid' => false]);
        }
        break;

    case 'create':
        $address = $_GET['address'] ?? '';
        if ($address) {
            $token = 'openagi_' . hash('sha256', $address . time() . rand());
            echo json_encode([
                'ok' => true,
                'token' => $token,
                'address' => $address,
                'expires' => time() + 3600
            ]);
        } else {
            echo json_encode(['ok' => false, 'error' => 'Address required']);
        }
        break;

    case 'challenge':
        $address = $_GET['address'] ?? '';
        if ($address) {
            $message = 'OpenAGI Auth Challenge: ' . time() . ' - ' . rand();
            echo json_encode([
                'ok' => true,
                'message' => $message,
                'address' => $address
            ]);
        } else {
            echo json_encode(['ok' => false, 'error' => 'Address required']);
        }
        break;

    case 'health':
        echo json_encode(['status' => 'ok', 'service' => 'openagi-auth-php']);
        break;

    default:
        echo json_encode(['ok' => false, 'error' => 'Unknown action']);
        break;
}
?>