<?php
header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');

$action = $_GET['action'] ?? 'status';

switch ($action) {
  case 'status':
    $summaryFile = __DIR__ . '/../../config/archon_project_summary.json';
    $status = [
      'name' => 'OpenAGI',
      'state' => 'desconocido',
      'timestamp' => time()
    ];
    if (file_exists($summaryFile)) {
      $raw = file_get_contents($summaryFile);
      $json = json_decode($raw, true);
      if (is_array($json)) {
        $status['state'] = $json['status'] ?? ($json['state'] ?? 'operacional');
        $status['version'] = $json['version'] ?? null;
        $status['last_update'] = $json['last_update'] ?? null;
      }
    }
    echo json_encode([ 'ok' => true, 'status' => $status ]);
    break;

  default:
    http_response_code(404);
    echo json_encode([ 'ok' => false, 'error' => 'acción no encontrada' ]);
}