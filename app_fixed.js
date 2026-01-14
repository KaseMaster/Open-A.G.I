/**
 * OpenAGI Secure Chat+ - Frontend JavaScript Corregido
 * Programador Principal: Jose Gómez alias KaseMaster
 * Contacto: kasemaster@protonmail.com
 * Versión: 2.1.0 - AEGIS Security Enhanced
 * Licencia: MIT
 */

// Variables globales
let currentRoom = 'general';
let currentUser = null;
let eventSource = null;
let isConnected = false;

// Elementos DOM con validación
const elements = {};

// Función para obtener elementos DOM de forma segura
function getElement(id) {
  if (!elements[id]) {
    elements[id] = document.getElementById(id);
    if (!elements[id]) {
      console.warn(`Elemento DOM no encontrado: ${id}`);
      return null;
    }
  }
  return elements[id];
}

// Función para mostrar notificaciones
function showNotification(message, type = 'info') {
  console.log(`[${type.toUpperCase()}] ${message}`);
  
  // Crear notificación visual si existe el contenedor
  const container = getElement('notifications') || document.body;
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.textContent = message;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 10px 20px;
    background: ${type === 'error' ? '#f44336' : type === 'success' ? '#4caf50' : '#2196f3'};
    color: white;
    border-radius: 4px;
    z-index: 1000;
    animation: slideIn 0.3s ease;
  `;
  
  container.appendChild(notification);
  
  setTimeout(() => {
    if (notification.parentNode) {
      notification.parentNode.removeChild(notification);
    }
  }, 5000);
}

// Función para conectar wallet MetaMask
async function connectWalletLogin() {
  try {
    if (!window.ethereum) {
      showNotification('MetaMask no está instalado', 'error');
      return false;
    }

    showNotification('Conectando con MetaMask...', 'info');
    
    // Solicitar acceso a las cuentas
    const accounts = await window.ethereum.request({
      method: 'eth_requestAccounts'
    });

    if (!accounts || accounts.length === 0) {
      showNotification('No se seleccionó ninguna cuenta', 'error');
      return false;
    }

    const walletAddress = accounts[0];
    const message = `OpenAGI Secure Chat+ Login\nAddress: ${walletAddress}\nTimestamp: ${Date.now()}`;

    // Firmar mensaje
    const signature = await window.ethereum.request({
      method: 'personal_sign',
      params: [message, walletAddress]
    });

    // Enviar al servidor
    const response = await fetch('/api_secure.php', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        action: 'wallet_login',
        wallet_address: walletAddress,
        signature: signature,
        message: message
      })
    });

    const result = await response.json();

    if (result.ok) {
      currentUser = {
        address: walletAddress,
        session: result.session
      };
      
      showNotification('Login exitoso', 'success');
      updateUI();
      loadRooms();
      connectWebSocket();
      return true;
    } else {
      showNotification(`Error de login: ${result.message}`, 'error');
      return false;
    }

  } catch (error) {
    console.error('Error en connectWalletLogin:', error);
    showNotification(`Error de conexión: ${error.message}`, 'error');
    return false;
  }
}

// Función para desconectar
async function logout() {
  try {
    if (currentUser) {
      await fetch('/api_secure.php', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          action: 'logout'
        })
      });
    }

    currentUser = null;
    disconnectWebSocket();
    updateUI();
    showNotification('Desconectado exitosamente', 'success');

  } catch (error) {
    console.error('Error en logout:', error);
    showNotification('Error al desconectar', 'error');
  }
}

// Función para actualizar la interfaz
function updateUI() {
  const loginForm = getElement('loginForm');
  const chatContainer = getElement('chatContainer');
  const userInfo = getElement('userInfo');
  const btnLogin = getElement('btnLogin');
  const btnLogout = getElement('btnLogout');

  if (currentUser) {
    // Usuario logueado
    if (loginForm) loginForm.style.display = 'none';
    if (chatContainer) chatContainer.style.display = 'block';
    if (userInfo) {
      userInfo.style.display = 'block';
      userInfo.textContent = `Conectado: ${currentUser.address.substring(0, 8)}...`;
    }
    if (btnLogin) btnLogin.style.display = 'none';
    if (btnLogout) btnLogout.style.display = 'inline-block';
  } else {
    // Usuario no logueado
    if (loginForm) loginForm.style.display = 'block';
    if (chatContainer) chatContainer.style.display = 'none';
    if (userInfo) userInfo.style.display = 'none';
    if (btnLogin) btnLogin.style.display = 'inline-block';
    if (btnLogout) btnLogout.style.display = 'none';
  }
}

// Función para cargar salas
async function loadRooms() {
  try {
    const response = await fetch('/api_secure.php?action=rooms');
    const result = await response.json();

    if (result.ok) {
      const roomsContainer = getElement('rooms');
      if (!roomsContainer) return;

      roomsContainer.innerHTML = '';
      
      result.rooms.forEach(room => {
        const roomEl = document.createElement('div');
        roomEl.className = 'room-item';
        roomEl.textContent = room.name;
        roomEl.onclick = () => selectRoom(room.id);
        
        if (room.id === currentRoom) {
          roomEl.classList.add('active');
        }
        
        roomsContainer.appendChild(roomEl);
      });
    }
  } catch (error) {
    console.error('Error cargando salas:', error);
    showNotification('Error cargando salas', 'error');
  }
}

// Función para seleccionar sala
async function selectRoom(roomId) {
  currentRoom = roomId;
  
  // Actualizar UI
  const chatTitle = getElement('chatTitle');
  if (chatTitle) {
    chatTitle.textContent = `Sala: ${roomId}`;
  }
  
  // Actualizar salas activas
  const roomItems = document.querySelectorAll('.room-item');
  roomItems.forEach(item => {
    item.classList.remove('active');
    if (item.textContent.toLowerCase().includes(roomId)) {
      item.classList.add('active');
    }
  });
  
  // Cargar mensajes
  await loadMessages();
  
  // Reconectar WebSocket para la nueva sala
  if (isConnected) {
    disconnectWebSocket();
    connectWebSocket();
  }
}

// Función para cargar mensajes
async function loadMessages() {
  try {
    const response = await fetch(`/api_secure.php?action=messages&room_id=${currentRoom}`);
    const result = await response.json();

    if (result.ok) {
      const messagesContainer = getElement('messages');
      if (!messagesContainer) return;

      messagesContainer.innerHTML = '';
      
      result.messages.forEach(message => {
        displayMessage(message);
      });
      
      // Scroll al final
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  } catch (error) {
    console.error('Error cargando mensajes:', error);
    showNotification('Error cargando mensajes', 'error');
  }
}

// Función para mostrar mensaje
function displayMessage(message) {
  const messagesContainer = getElement('messages');
  if (!messagesContainer) return;

  const messageEl = document.createElement('div');
  messageEl.className = 'message';
  
  const time = new Date(message.ts * 1000).toLocaleTimeString();
  const author = message.author.substring(0, 8) + '...';
  
  if (message.type === 'file') {
    messageEl.innerHTML = `
      <div class="message-header">
        <span class="author">${author}</span>
        <span class="time">${time}</span>
      </div>
      <div class="message-content">
        📎 <a href="/uploads/${message.filename}" target="_blank">${message.text}</a>
      </div>
    `;
  } else {
    messageEl.innerHTML = `
      <div class="message-header">
        <span class="author">${author}</span>
        <span class="time">${time}</span>
        ${message.enc ? '<span class="encrypted">🔒</span>' : ''}
      </div>
      <div class="message-content">${escapeHtml(message.text)}</div>
    `;
  }
  
  messagesContainer.appendChild(messageEl);
}

// Función para escapar HTML
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Función para enviar mensaje
async function sendMessage() {
  const messageInput = getElement('messageInput');
  const encryptToggle = getElement('encryptToggle');
  
  if (!messageInput) return;
  
  const text = messageInput.value.trim();
  if (!text) return;
  
  const encrypted = encryptToggle ? encryptToggle.checked : false;
  
  try {
    const response = await fetch('/api_secure.php', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        action: 'send_message',
        room_id: currentRoom,
        text: text,
        encrypted: encrypted
      })
    });

    const result = await response.json();

    if (result.ok) {
      messageInput.value = '';
      // El mensaje aparecerá via WebSocket
    } else {
      showNotification(`Error enviando mensaje: ${result.error}`, 'error');
    }

  } catch (error) {
    console.error('Error enviando mensaje:', error);
    showNotification('Error enviando mensaje', 'error');
  }
}

// Función para enviar archivo
async function sendFile() {
  const fileInput = getElement('fileInput');
  if (!fileInput || !fileInput.files[0]) {
    showNotification('Selecciona un archivo', 'error');
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('action', 'send_file');
  formData.append('room_id', currentRoom);
  formData.append('file', file);

  try {
    const response = await fetch('/api_secure.php', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();

    if (result.ok) {
      fileInput.value = '';
      showNotification('Archivo enviado', 'success');
    } else {
      showNotification(`Error enviando archivo: ${result.error}`, 'error');
    }

  } catch (error) {
    console.error('Error enviando archivo:', error);
    showNotification('Error enviando archivo', 'error');
  }
}

// Función para conectar WebSocket
function connectWebSocket() {
  if (!currentUser) return;
  
  disconnectWebSocket();
  
  try {
    eventSource = new EventSource(`/websocket_real.php?room=${currentRoom}`);
    
    eventSource.onopen = function() {
      isConnected = true;
      showNotification('WebSocket conectado', 'success');
    };
    
    eventSource.onmessage = function(event) {
      console.log('WebSocket mensaje:', event.data);
    };
    
    eventSource.addEventListener('connected', function(event) {
      const data = JSON.parse(event.data);
      console.log('WebSocket conectado a sala:', data.room);
    });
    
    eventSource.addEventListener('new_message', function(event) {
      const data = JSON.parse(event.data);
      if (data.room_id === currentRoom) {
        displayMessage(data.message);
        
        // Scroll al final
        const messagesContainer = getElement('messages');
        if (messagesContainer) {
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
      }
    });
    
    eventSource.addEventListener('heartbeat', function(event) {
      const data = JSON.parse(event.data);
      console.log('WebSocket heartbeat:', data.server_time);
    });
    
    eventSource.addEventListener('active_users', function(event) {
      const data = JSON.parse(event.data);
      console.log('Usuarios activos:', data.count);
    });
    
    eventSource.onerror = function(event) {
      console.error('WebSocket error:', event);
      isConnected = false;
      showNotification('Error de conexión WebSocket', 'error');
    };
    
  } catch (error) {
    console.error('Error conectando WebSocket:', error);
    showNotification('Error conectando WebSocket', 'error');
  }
}

// Función para desconectar WebSocket
function disconnectWebSocket() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
    isConnected = false;
  }
}

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
  console.log('OpenAGI Secure Chat+ iniciando...');
  
  // Configurar event listeners con validación
  const btnLogin = getElement('btnLogin');
  if (btnLogin) {
    btnLogin.addEventListener('click', connectWalletLogin);
  }
  
  const btnLogout = getElement('btnLogout');
  if (btnLogout) {
    btnLogout.addEventListener('click', logout);
  }
  
  const sendBtn = getElement('sendBtn');
  if (sendBtn) {
    sendBtn.addEventListener('click', sendMessage);
  }
  
  const fileBtn = getElement('fileBtn');
  if (fileBtn) {
    fileBtn.addEventListener('click', sendFile);
  }
  
  const messageInput = getElement('messageInput');
  if (messageInput) {
    messageInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }
  
  // Inicializar UI
  updateUI();
  
  // Cargar salas públicas
  loadRooms();
  
  console.log('OpenAGI Secure Chat+ iniciado correctamente');
});

// Limpiar al cerrar la página
window.addEventListener('beforeunload', function() {
  disconnectWebSocket();
});