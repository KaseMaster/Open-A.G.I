// OpenAGI Secure Chat+ - JavaScript Simplificado basado en diagnóstico exitoso
// Replica exactamente la lógica que funciona en metamask_debug.html

let currentUser = null;
let currentRoom = 'general';
let eventSource = null;
let isConnected = false;

// Log function para debugging
function log(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${type.toUpperCase()}] [${timestamp}] ${message}`);
}

// Verificación de elementos DOM al cargar
document.addEventListener('DOMContentLoaded', function() {
    log('🚀 DOM loaded, initializing MetaMask functionality...', 'info');
    
    // Verificar elementos críticos
    const requiredElements = ['btnLogin', 'userInfo', 'loginForm', 'chatContainer'];
    const missingElements = [];
    
    requiredElements.forEach(id => {
        if (!document.getElementById(id)) {
            missingElements.push(id);
        }
    });

    if (missingElements.length > 0) {
        log(`❌ Elementos DOM faltantes: ${missingElements.join(', ')}`, 'error');
        return;
    }

    // Inicializar event listeners
    initializeEventListeners();
    
    log('✅ MetaMask initialization complete', 'success');
});

// Inicializar event listeners
function initializeEventListeners() {
    const btnLogin = document.getElementById('btnLogin');
    if (btnLogin) {
        btnLogin.addEventListener('click', connectWalletLogin);
        log('✅ Event listener agregado al botón de login', 'success');
    }

    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
}

// Función de conexión simplificada - EXACTAMENTE como en diagnóstico
async function connectWalletLogin() {
    log('🔄 Iniciando conexión de cartera...', 'info');
    
    try {
        // Paso 1: Verificar MetaMask - EXACTO como diagnóstico
        if (typeof window.ethereum === 'undefined') {
            log('❌ MetaMask NO detectado', 'error');
            alert('MetaMask no está instalado. Por favor, instala MetaMask para continuar.');
            return;
        }
        log('✅ Paso 1: MetaMask detectado', 'success');

        // Paso 2: Conectar cuentas - EXACTO como diagnóstico
        log('🔄 Solicitando acceso a cuentas...', 'info');
        const accounts = await window.ethereum.request({
            method: 'eth_requestAccounts'
        });

        if (!accounts || accounts.length === 0) {
            log('❌ No se obtuvieron cuentas', 'error');
            alert('No se encontraron cuentas. Por favor, desbloquea MetaMask.');
            return;
        }
        
        const walletAddress = accounts[0];
        log(`✅ Paso 2: Cuenta conectada: ${walletAddress}`, 'success');

        // Paso 3: Firmar mensaje - EXACTO como diagnóstico
        const message = `OpenAGI Secure Chat+ Login\nAddress: ${walletAddress}\nTimestamp: ${Date.now()}`;
        log('🔄 Solicitando firma de mensaje...', 'info');

        const signature = await window.ethereum.request({
            method: 'personal_sign',
            params: [message, walletAddress]
        });
        log('✅ Paso 3: Mensaje firmado exitosamente', 'success');

        // Paso 4: Enviar al servidor - EXACTO como diagnóstico
        log('🔄 Paso 4: Enviando al servidor...', 'info');
        
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
            log('✅ Paso 4: Login exitoso en servidor', 'success');
            log(`🎉 Sesión creada: ${result.session.session_id.substring(0, 10)}...`, 'success');
            
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
            log(`❌ Error del servidor: ${result.message}`, 'error');
            alert(`Error de autenticación: ${result.message}`);
        }

    } catch (error) {
        log(`❌ Error en conexión de cartera: ${error.message}`, 'error');
        
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

// Actualizar UI después del login
function updateUIAfterLogin(walletAddress) {
    log(`🔄 Actualizando UI para usuario: ${walletAddress}`, 'info');
    
    const loginForm = document.getElementById('loginForm');
    const chatContainer = document.getElementById('chatContainer');
    const userInfo = document.getElementById('userInfo');

    if (loginForm) {
        loginForm.style.display = 'none';
    }

    if (chatContainer) {
        chatContainer.style.display = 'block';
    }

    if (userInfo) {
        userInfo.innerHTML = `
            <div class="user-wallet">
                <span>🔗 ${walletAddress.substring(0, 6)}...${walletAddress.substring(-4)}</span>
                <button onclick="disconnectWallet()" class="disconnect-btn">Desconectar</button>
            </div>
        `;
        userInfo.style.display = 'block';
    }
    
    log('✅ UI actualizada exitosamente', 'success');
}

// Desconectar cartera
function disconnectWallet() {
    log('🔄 Desconectando cartera...', 'info');
    
    currentUser = null;
    
    const loginForm = document.getElementById('loginForm');
    const chatContainer = document.getElementById('chatContainer');
    const userInfo = document.getElementById('userInfo');

    if (loginForm) {
        loginForm.style.display = 'block';
    }

    if (chatContainer) {
        chatContainer.style.display = 'none';
    }

    if (userInfo) {
        userInfo.style.display = 'none';
    }

    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    
    log('✅ Cartera desconectada', 'success');
}

// Cargar salas
async function loadRooms() {
    log('🔄 Cargando salas...', 'info');
    
    try {
        const response = await fetch('/api.php?action=rooms');
        const rooms = await response.json();
        
        const roomsList = document.getElementById('roomsList');
        if (roomsList && rooms) {
            roomsList.innerHTML = '';
            rooms.forEach(room => {
                const roomElement = document.createElement('div');
                roomElement.className = 'room-item';
                roomElement.textContent = room.name;
                roomElement.onclick = () => switchRoom(room.name);
                roomsList.appendChild(roomElement);
            });
        }
        
        log(`✅ ${rooms.length} salas cargadas`, 'success');
    } catch (error) {
        log(`❌ Error cargando salas: ${error.message}`, 'error');
    }
}

// Cambiar sala
function switchRoom(roomName) {
    currentRoom = roomName;
    log(`🔄 Cambiando a sala: ${roomName}`, 'info');
    loadMessages();
}

// Cargar mensajes
async function loadMessages() {
    log(`🔄 Cargando mensajes de sala: ${currentRoom}`, 'info');
    
    try {
        const response = await fetch(`/api.php?action=messages&room=${currentRoom}`);
        const messages = await response.json();
        
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer && messages) {
            messagesContainer.innerHTML = '';
            messages.forEach(message => {
                displayMessage(message);
            });
        }
        
        log(`✅ ${messages.length} mensajes cargados`, 'success');
    } catch (error) {
        log(`❌ Error cargando mensajes: ${error.message}`, 'error');
    }
}

// Mostrar mensaje
function displayMessage(message) {
    const messagesContainer = document.getElementById('messagesContainer');
    if (!messagesContainer) return;

    const messageElement = document.createElement('div');
    messageElement.className = 'message';
    messageElement.innerHTML = `
        <div class="message-header">
            <span class="author">${message.author}</span>
            <span class="timestamp">${new Date(message.timestamp).toLocaleTimeString()}</span>
        </div>
        <div class="message-content">${message.text}</div>
    `;
    
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Enviar mensaje
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    if (!messageInput || !currentUser) return;

    const text = messageInput.value.trim();
    if (!text) return;

    log(`🔄 Enviando mensaje: ${text.substring(0, 20)}...`, 'info');

    try {
        const response = await fetch('/api.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                action: 'send_message',
                room_id: currentRoom,
                text: text,
                author: currentUser.address
            })
        });

        const result = await response.json();
        
        if (result.ok) {
            messageInput.value = '';
            log('✅ Mensaje enviado exitosamente', 'success');
        } else {
            log(`❌ Error enviando mensaje: ${result.message}`, 'error');
        }
    } catch (error) {
        log(`❌ Error enviando mensaje: ${error.message}`, 'error');
    }
}

// Conectar WebSocket
function connectWebSocket() {
    log('🔄 Conectando WebSocket...', 'info');
    
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource(`/websocket.php?room=${currentRoom}`);
    
    eventSource.onopen = function() {
        isConnected = true;
        log('✅ WebSocket conectado', 'success');
    };

    eventSource.onmessage = function(event) {
        try {
            const message = JSON.parse(event.data);
            displayMessage(message);
        } catch (error) {
            log(`❌ Error procesando mensaje WebSocket: ${error.message}`, 'error');
        }
    };

    eventSource.onerror = function() {
        isConnected = false;
        log('❌ Error en WebSocket', 'error');
    };
}