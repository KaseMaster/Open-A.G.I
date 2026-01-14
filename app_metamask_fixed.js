// OpenAGI Secure Chat+ - JavaScript Corregido con MetaMask Mejorado
// Basado en diagnóstico exitoso de MetaMask

let currentUser = null;
let currentRoom = 'general';
let eventSource = null;
let isConnected = false;

// Verificación de elementos DOM al cargar
document.addEventListener('DOMContentLoaded', function() {
  console.log('🚀 DOM loaded, initializing MetaMask functionality...');

  // Verificar elementos críticos
  const requiredElements = ['btnLogin', 'userInfo', 'loginForm', 'chatContainer', 'messageInput', 'sendBtn'];
  const missingElements = [];
  
  requiredElements.forEach(id => {
    if (!document.getElementById(id)) {
      missingElements.push(id);
    }
  });

  if (missingElements.length > 0) {
    console.error('❌ Elementos DOM faltantes:', missingElements);
    return;
  }

  // Verificar MetaMask con diagnóstico mejorado
  checkMetaMaskAvailability();
  
  // Inicializar event listeners
  initializeEventListeners();
  
  console.log('✅ MetaMask initialization complete');
});

// Función mejorada de verificación de MetaMask
function checkMetaMaskAvailability() {
  if (typeof window.ethereum !== 'undefined') {
    console.log('✅ MetaMask detectado!');
    
    // Verificar si es MetaMask oficial
    if (window.ethereum.isMetaMask) {
      console.log('✅ Confirmado: Es MetaMask oficial');
    } else {
      console.warn('⚠️ Wallet detectado pero no es MetaMask oficial');
    }
    
    // Verificar red actual
    window.ethereum.request({ method: 'eth_chainId' })
      .then(chainId => {
        console.log(`🌐 Red actual: ${chainId}`);
      })
      .catch(error => {
        console.error('❌ Error obteniendo red:', error);
      });
      
  } else {
    console.log('❌ MetaMask no detectado');
  }
}

// Función de conexión de cartera mejorada
async function connectWalletLogin() {
  console.log('🔄 Iniciando conexión de cartera...');
  
  try {
    // Paso 1: Verificar MetaMask
    if (typeof window.ethereum === 'undefined') {
      alert('MetaMask no está instalado. Por favor, instala MetaMask para continuar.');
      return;
    }
    console.log('✅ Paso 1: MetaMask detectado');

    // Paso 2: Solicitar acceso a cuentas
    console.log('🔄 Solicitando acceso a cuentas...');
    const accounts = await window.ethereum.request({ 
      method: 'eth_requestAccounts' 
    });

    if (!accounts || accounts.length === 0) {
      alert('No se encontraron cuentas. Por favor, desbloquea MetaMask.');
      return;
    }
    
    const walletAddress = accounts[0];
    console.log(`✅ Paso 2: Cuenta conectada: ${walletAddress}`);

    // Paso 3: Crear mensaje para firmar
    const message = `OpenAGI Secure Chat+ Login\nAddress: ${walletAddress}\nTimestamp: ${Date.now()}`;
    console.log('🔄 Solicitando firma de mensaje...');

    const signature = await window.ethereum.request({
      method: 'personal_sign',
      params: [message, walletAddress]
    });
    console.log('✅ Paso 3: Mensaje firmado exitosamente');

    // Paso 4: Enviar al servidor
    console.log('🔄 Paso 4: Enviando al servidor...');
    
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
      console.log('✅ Paso 4: Login exitoso en servidor');
      console.log(`🎉 Sesión creada: ${result.session.session_id.substring(0, 10)}...`);
      
      // Actualizar UI
      currentUser = {
        address: walletAddress,
        session: result.session
      };
      
      updateUIAfterLogin(walletAddress);
      loadRooms();
      connectWebSocket();
      
      alert('¡Conexión exitosa con MetaMask!');
      
    } else {
      console.error(`❌ Error del servidor: ${result.message}`);
      alert(`Error de autenticación: ${result.message}`);
    }

  } catch (error) {
    console.error('❌ Error en conexión de cartera:', error);
    
    // Manejo específico de errores de MetaMask
    if (error.code === 4001) {
      alert('Conexión cancelada por el usuario.');
    } else if (error.code === -32002) {
      alert('Ya hay una solicitud de conexión pendiente. Revisa MetaMask.');
    } else {
      alert(`Error de conexión: ${error.message}`);
    }
  }
}

// Función para actualizar UI después del login
function updateUIAfterLogin(address) {
  const loginForm = document.getElementById('loginForm');
  const chatContainer = document.getElementById('chatContainer');
  const userInfo = document.getElementById('userInfo');
  
  if (loginForm) loginForm.style.display = 'none';
  if (chatContainer) chatContainer.style.display = 'block';
  if (userInfo) {
    userInfo.innerHTML = `
      <div class="user-info">
        <span class="wallet-address">${address.substring(0, 6)}...${address.substring(38)}</span>
        <button onclick="disconnectWallet()" class="disconnect-btn">Desconectar</button>
      </div>
    `;
    userInfo.style.display = 'block';
  }
}

// Función para desconectar cartera
async function disconnectWallet() {
  try {
    if (currentUser && currentUser.session) {
      // Notificar al servidor
      await fetch('/api_secure.php', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          action: 'logout',
          session_id: currentUser.session.session_id
        })
      });
    }
    
    // Limpiar estado local
    currentUser = null;
    
    // Desconectar WebSocket
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    
    // Actualizar UI
    const loginForm = document.getElementById('loginForm');
    const chatContainer = document.getElementById('chatContainer');
    const userInfo = document.getElementById('userInfo');
    
    if (loginForm) loginForm.style.display = 'block';
    if (chatContainer) chatContainer.style.display = 'none';
    if (userInfo) userInfo.style.display = 'none';
    
    console.log('✅ Desconectado exitosamente');
    
  } catch (error) {
    console.error('❌ Error al desconectar:', error);
  }
}

// Función para cargar salas
async function loadRooms() {
  if (!currentUser) return;
  
  try {
    const response = await fetch('/api_secure.php', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        action: 'rooms',
        session_id: currentUser.session.session_id
      })
    });
    
    const result = await response.json();
    
    if (result.ok && result.rooms) {
      updateRoomsList(result.rooms);
    }
    
  } catch (error) {
    console.error('❌ Error cargando salas:', error);
  }
}

// Función para actualizar lista de salas
function updateRoomsList(rooms) {
  const roomsList = document.getElementById('roomsList');
  if (!roomsList) return;
  
  roomsList.innerHTML = '';
  
  rooms.forEach(room => {
    const roomElement = document.createElement('div');
    roomElement.className = `room-item ${room.name === currentRoom ? 'active' : ''}`;
    roomElement.innerHTML = `
      <span class="room-name">${room.name}</span>
      <span class="room-count">${room.member_count || 0}</span>
    `;
    roomElement.onclick = () => selectRoom(room.name);
    roomsList.appendChild(roomElement);
  });
}

// Función para seleccionar sala
async function selectRoom(roomName) {
  currentRoom = roomName;
  
  // Actualizar UI
  document.querySelectorAll('.room-item').forEach(item => {
    item.classList.remove('active');
  });
  
  event.target.closest('.room-item').classList.add('active');
  
  // Cargar mensajes de la sala
  await loadMessages(roomName);
}

// Función para cargar mensajes
async function loadMessages(roomName) {
  if (!currentUser) return;
  
  try {
    const response = await fetch('/api_secure.php', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        action: 'messages',
        room_id: roomName,
        session_id: currentUser.session.session_id
      })
    });
    
    const result = await response.json();
    
    if (result.ok && result.messages) {
      displayMessages(result.messages);
    }
    
  } catch (error) {
    console.error('❌ Error cargando mensajes:', error);
  }
}

// Función para mostrar mensajes
function displayMessages(messages) {
  const messagesContainer = document.getElementById('messagesContainer');
  if (!messagesContainer) return;
  
  messagesContainer.innerHTML = '';
  
  messages.forEach(message => {
    const messageElement = document.createElement('div');
    messageElement.className = 'message';
    messageElement.innerHTML = `
      <div class="message-header">
        <span class="message-author">${message.author}</span>
        <span class="message-time">${new Date(message.timestamp * 1000).toLocaleTimeString()}</span>
      </div>
      <div class="message-content">${message.text}</div>
    `;
    messagesContainer.appendChild(messageElement);
  });
  
  // Scroll al final
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Función para enviar mensaje
async function sendMessage() {
  if (!currentUser) return;
  
  const messageInput = document.getElementById('messageInput');
  if (!messageInput) return;
  
  const text = messageInput.value.trim();
  if (!text) return;
  
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
        session_id: currentUser.session.session_id
      })
    });
    
    const result = await response.json();
    
    if (result.ok) {
      messageInput.value = '';
      console.log('✅ Mensaje enviado');
    } else {
      console.error('❌ Error enviando mensaje:', result.message);
    }
    
  } catch (error) {
    console.error('❌ Error enviando mensaje:', error);
  }
}

// Función para conectar WebSocket
function connectWebSocket() {
  if (!currentUser || eventSource) return;
  
  try {
    eventSource = new EventSource('/websocket_real.php');
    
    eventSource.onopen = function() {
      console.log('✅ WebSocket conectado');
      isConnected = true;
    };
    
    eventSource.onmessage = function(event) {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (error) {
        console.error('❌ Error procesando mensaje WebSocket:', error);
      }
    };
    
    eventSource.onerror = function(error) {
      console.error('❌ Error WebSocket:', error);
      isConnected = false;
    };
    
  } catch (error) {
    console.error('❌ Error conectando WebSocket:', error);
  }
}

// Función para manejar mensajes WebSocket
function handleWebSocketMessage(data) {
  switch (data.type) {
    case 'new_message':
      if (data.room_id === currentRoom) {
        addNewMessage(data.message);
      }
      break;
      
    case 'room_update':
      loadRooms();
      break;
      
    case 'user_joined':
    case 'user_left':
      console.log(`Usuario ${data.type}: ${data.user}`);
      break;
      
    default:
      console.log('Mensaje WebSocket:', data);
  }
}

// Función para agregar nuevo mensaje
function addNewMessage(message) {
  const messagesContainer = document.getElementById('messagesContainer');
  if (!messagesContainer) return;
  
  const messageElement = document.createElement('div');
  messageElement.className = 'message';
  messageElement.innerHTML = `
    <div class="message-header">
      <span class="message-author">${message.author}</span>
      <span class="message-time">${new Date(message.timestamp * 1000).toLocaleTimeString()}</span>
    </div>
    <div class="message-content">${message.text}</div>
  `;
  
  messagesContainer.appendChild(messageElement);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Función para inicializar event listeners
function initializeEventListeners() {
  // Botón de login
  const btnLogin = document.getElementById('btnLogin');
  if (btnLogin) {
    btnLogin.addEventListener('click', connectWalletLogin);
  }
  
  // Botón de envío
  const sendBtn = document.getElementById('sendBtn');
  if (sendBtn) {
    sendBtn.addEventListener('click', sendMessage);
  }
  
  // Enter en input de mensaje
  const messageInput = document.getElementById('messageInput');
  if (messageInput) {
    messageInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }
  
  // Botón de archivo
  const fileBtn = document.getElementById('fileBtn');
  const fileInput = document.getElementById('fileInput');
  if (fileBtn && fileInput) {
    fileBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);
  }
}

// Función para manejar carga de archivos
async function handleFileUpload(event) {
  if (!currentUser) return;
  
  const file = event.target.files[0];
  if (!file) return;
  
  const formData = new FormData();
  formData.append('action', 'send_file');
  formData.append('room_id', currentRoom);
  formData.append('file', file);
  formData.append('session_id', currentUser.session.session_id);
  
  try {
    const response = await fetch('/api_secure.php', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    if (result.ok) {
      console.log('✅ Archivo enviado');
      event.target.value = ''; // Limpiar input
    } else {
      console.error('❌ Error enviando archivo:', result.message);
    }
    
  } catch (error) {
    console.error('❌ Error enviando archivo:', error);
  }
}

// Función de utilidad para logging
function log(message, type = 'info') {
  const timestamp = new Date().toLocaleTimeString();
  const prefix = type === 'error' ? '❌' : type === 'success' ? '✅' : '🔄';
  console.log(`${prefix} [${timestamp}] ${message}`);
}

console.log('🚀 OpenAGI Secure Chat+ JavaScript cargado con MetaMask mejorado');