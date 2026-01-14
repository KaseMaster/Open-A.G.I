<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

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
                'address' => $address,
                'timestamp' => time()
            ]);
        } else {
            echo json_encode(['ok' => false, 'error' => 'Invalid token']);
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