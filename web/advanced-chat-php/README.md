# OpenAGI Secure Chat+ (PHP)

Chat avanzado en HTML/PHP inspirado en la dApp de Secure Chat, con integración básica de estado de OpenAGI.

## Requisitos
- PHP 8.x con servidor embebido (`php -S`).

## Ejecutar en modo desarrollo

```bash
php -S localhost:8086 -t web/advanced-chat-php/public
```

Abrir: http://localhost:8086/

## Funcionalidades
- Gestión de salas (crear/listar).
- Mensajería básica por sala (persistencia en archivos JSON locales).
- UI homogenizada con clases tipo `App.css`.
- Integración de estado OpenAGI leyendo `config/archon_project_summary.json` si está disponible.

## Estructura
- `public/index.php`: UI principal.
- `public/api.php`: API para salas y mensajes.
- `public/openagi.php`: Estado e info de OpenAGI.
- `public/assets/css/app.css`: Estilos.
- `public/assets/js/app.js`: Lógica en el cliente.