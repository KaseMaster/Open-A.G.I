<?php
include 'api.php';
$token = 'test123456789';
echo 'Token: ' . $token . PHP_EOL;
$result = verifySessionToken($token);
echo 'Resultado verifySessionToken: ' . ($result ? 'true' : 'false') . PHP_EOL;
$address = sessionAddress($token);
echo 'Address: ' . $address . PHP_EOL;
?>