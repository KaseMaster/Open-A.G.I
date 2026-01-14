<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenAGI Secure Chat+</title>
  <link rel="stylesheet" href="assets/css/app.css">
  <script src="ethers.umd.min.js"></script>
</head>
<body>
  <div class="app">
    <!-- Header -->
    <header class="header">
      <div class="brand">
        <div class="brand-title">🔒 OpenAGI Secure Chat+</div>
        <div class="badge" id="openagiStatus">Cargando estado...</div>
        <button class="button" id="btnLogin">Conectar cartera</button>
        <button class="button" id="btnLogout" style="display:none;">Cerrar sesión</button>
        <div class="brand-sub" id="userBadge" style="font-size:12px;">Invitado</div>
      </div>
      
      <!-- NUEVO: Panel de información del usuario -->
      <div id="userInfo" class="user-info" style="display:none;">
        <div class="user-avatar">👤</div>
        <div class="user-details">
          <div class="user-name"></div>
          <div class="user-address"></div>
        </div>
      </div>
    </header>

    <!-- NUEVO: Formulario de login -->
    <div id="loginForm" class="login-form" style="display:none;">
      <div class="login-container">
        <h2>Conectar Wallet</h2>
        <p>Conecta tu wallet MetaMask para acceder al chat seguro</p>
        <button id="connectMetaMask" class="button button-primary">🦊 Conectar MetaMask</button>
      </div>
    </div>

    <!-- NUEVO: Contenedor principal del chat -->
    <div id="chatContainer" class="chat-container">
      <div class="container">
        <!-- Sidebar -->
        <aside class="sidebar">
          <div class="sidebar-header">
            <h2>Salas</h2>
            <button class="button button-primary" id="btnOpenCreate">Crear sala</button>
          </div>
          <div class="rooms" id="rooms"></div>
        </aside>

        <!-- Main Chat -->
        <main class="main">
          <div class="chat-header">
            <div class="chat-title" id="chatTitle">Selecciona una sala</div>
            <div class="chat-sub" id="chatSub">Elige una sala para comenzar a conversar</div>
          </div>
          <div id="roomActions" style="display:flex; align-items:center; gap:8px;"></div>

          <div class="messages" id="messages"></div>

          <div class="input-bar">
            <textarea class="input" id="messageInput" placeholder="Escribe tu mensaje (Shift+Enter nueva línea, Enter para enviar)"></textarea>
            <input type="file" id="fileInput" style="margin-left:8px;" />
            
            <!-- NUEVO: Botón para enviar archivos -->
            <button class="button" id="fileBtn" title="Enviar archivo">📎</button>
            
            <label style="margin-left:8px; display:flex; align-items:center; gap:6px;">
              <input type="checkbox" id="encryptToggle" />
              <span>Enviar cifrado</span>
            </label>
            <button class="button button-primary" id="sendBtn">Enviar</button>
          </div>
        </main>
      </div>
    </div>

    <!-- Modales -->
    <div id="createModal" class="modal-overlay" style="display:none;">
      <div class="modal">
        <div class="modal-header">
          <h3>Crear nueva sala</h3>
        </div>
        <div class="modal-body">
          <input class="input" id="newRoomName" placeholder="Nombre de la sala..." />
          <br><br>
          <label style="display:flex; align-items:center; gap:8px;">
            <input type="checkbox" id="newRoomAccessFree" />
            <span>Acceso libre (sin restricciones)</span>
          </label>
        </div>
        <div class="modal-footer">
          <button class="button button-primary" id="createRoomBtn">Crear</button>
          <button class="button" id="cancelCreateBtn">Cancelar</button>
        </div>
      </div>
    </div>

    <div id="membersModal" class="modal-overlay" style="display:none;">
      <div class="modal">
        <div class="modal-header">
          <h3>Miembros de la sala</h3>
        </div>
        <div class="modal-body" id="membersContent">
          <!-- Contenido dinámico -->
        </div>
        <div class="modal-footer">
          <button class="button" id="closeMembersBtn">Cerrar</button>
        </div>
      </div>
    </div>
  </div>

  <script src="assets/js/app.js"></script>

  <style>
    /* Estilos adicionales para elementos nuevos */
    .user-info {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px;
      background: rgba(255,255,255,0.1);
      border-radius: 8px;
      margin-left: auto;
    }

    .user-avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: #4CAF50;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
    }

    .user-details {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .user-name {
      font-weight: bold;
      font-size: 14px;
    }

    .user-address {
      font-size: 12px;
      opacity: 0.8;
      font-family: monospace;
    }

    .login-form {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }

    .login-container {
      background: white;
      padding: 30px;
      border-radius: 12px;
      text-align: center;
      max-width: 400px;
      width: 90%;
    }

    .login-container h2 {
      margin-bottom: 15px;
      color: #333;
    }

    .login-container p {
      margin-bottom: 25px;
      color: #666;
    }

    .chat-container {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    #fileBtn {
      background: #2196F3;
      color: white;
      border: none;
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 16px;
    }

    #fileBtn:hover {
      background: #1976D2;
    }
  </style>
</body>
</html>