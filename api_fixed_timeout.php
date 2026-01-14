<?php
// Parche temporal para api.php - comentar publishToFastApi para evitar timeout

// Leer el archivo api.php original
$apiContent = file_get_contents('api.php');

// Comentar la línea que causa el timeout
$fixedContent = str_replace(
    'publishToFastApi($roomId, $msg);',
    '// publishToFastApi($roomId, $msg); // COMENTADO: FastAPI no está corriendo',
    $apiContent
);

// Guardar el archivo modificado
file_put_contents('api_fixed.php', $fixedContent);

echo "Archivo api_fixed.php creado con publishToFastApi comentado\n";
echo "Para aplicar el fix:\n";
echo "1. cp api.php api_backup.php\n";
echo "2. cp api_fixed.php api.php\n";
echo "3. Reiniciar servidor PHP si es necesario\n";
?>