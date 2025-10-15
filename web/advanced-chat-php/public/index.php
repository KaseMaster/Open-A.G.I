<?php
/**
 * OpenAGI Secure Chat+ - Aplicación de Chat Seguro
 * Programador Principal: Jose Gómez alias KaseMaster
 * Contacto: kasemaster@aegis-framework.com
 * Versión: 2.0.0
 * Licencia: MIT
 */

// Simple router guard: serve only this file as app entry
?>
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OpenAGI Secure Chat+</title>
    <link rel="stylesheet" href="/assets/css/app.css" />
  </head>
  <body>
    <header class="app-header">
      <div class="brand">
        <div class="brand-title">OpenAGI Secure Chat+</div>
        <div class="brand-sub">Versión avanzada en PHP</div>
      </div>
      <div style="display:flex; align-items:center; gap:12px;">
        <div class="badge" id="openagiStatus">Cargando estado...</div>
        <button class="button" id="btnLogin">Conectar cartera</button>
        <button class="button" id="btnLogout" style="display:none;">Cerrar sesión</button>
        <div class="brand-sub" id="userBadge" style="font-size:12px;">Invitado</div>
      </div>
    </header>

    <main class="app-main">
      <aside class="sidebar">
        <div class="sidebar-header">
          <div>
            <div class="brand-title" style="font-size: 16px;">Salas</div>
            <div class="brand-sub" style="font-size: 12px;">Selecciona o crea una sala</div>
          </div>
          <button class="button button-primary" id="btnOpenCreate">Crear sala</button>
        </div>
        <div class="rooms" id="rooms"></div>
      </aside>

      <section class="chat">
        <div class="chat-header">
          <div>
            <div class="chat-title" id="chatTitle">Selecciona una sala</div>
            <div class="chat-sub" id="chatSub">Elige una sala para comenzar a conversar</div>
          </div>
          <div id="roomActions" style="display:flex; align-items:center; gap:8px;"></div>
        </div>

        <div class="messages" id="messages"></div>

        <div class="input-bar">
          <textarea class="input" id="messageInput" placeholder="Escribe tu mensaje (Shift+Enter nueva línea, Enter para enviar)"></textarea>
          <input type="file" id="fileInput" style="margin-left:8px;" />
          <label style="margin-left:8px; display:flex; align-items:center; gap:6px;">
            <input type="checkbox" id="encryptToggle" />
            <span>Enviar cifrado</span>
          </label>
          <button class="button button-primary" id="sendBtn">Enviar</button>
        </div>
      </section>
    </main>

    <!-- Modal Crear Sala -->
    <div id="createModal" class="modal-overlay" style="display:none;">
      <div class="modal">
        <div class="chat-title">Crear Nueva Sala</div>
        <div style="margin-top:12px;">
          <input class="input" id="newRoomName" placeholder="Nombre de la sala..." />
        </div>
        <div style="margin-top:12px; display:flex; align-items:center; gap:8px;">
          <label style="display:flex; align-items:center; gap:6px;">
            <input type="checkbox" id="newRoomAccessFree" />
            <span>Acceso libre</span>
          </label>
        </div>
        <div class="modal-actions">
          <button class="button button-primary" id="createRoomBtn">Crear</button>
          <button class="button" id="cancelCreateBtn">Cancelar</button>
        </div>
      </div>
    </div>

    <!-- Modal Miembros de Sala -->
    <div id="membersModal" class="modal-overlay" style="display:none;">
      <div class="modal">
        <div class="chat-title">Miembros de la sala</div>
        <div id="membersList" style="margin-top:12px; display:flex; flex-direction:column; gap:6px;"></div>
        <div class="modal-actions">
          <button class="button" id="closeMembersBtn">Cerrar</button>
        </div>
      </div>
    </div>

    <!-- Contenedor de toasts -->
    <div id="toastContainer" aria-live="polite" aria-atomic="true"></div>

    <script src="https://cdn.jsdelivr.net/npm/ethers@6.10.0/dist/ethers.umd.min.js"></script>
    <script src="/assets/js/app.js"></script>
  </body>
  </html>