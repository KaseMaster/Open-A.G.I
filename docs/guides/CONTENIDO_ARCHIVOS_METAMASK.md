# 📁 CONTENIDO COMPLETO DE ARCHIVOS - IMPLEMENTACIÓN METAMASK

---

## 🔹 ARCHIVO 1: `app_fixed.js` (JavaScript Simplificado)

**Ubicación en servidor:** `/opt/openagi/web/advanced-chat-php/public/app_fixed.js`

**Instrucciones:** Reemplazar TODO el contenido del archivo actual con el siguiente código:

```javascript
// OpenAGI Secure Chat+ - JavaScript Simplificado con MetaMask
// Versión optimizada basada en página de diagnóstico exitosa

// ===== CONFIGURACIÓN Y LOGGING =====
function log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const prefix = `[${timestamp}] [MetaMask]`;
    
    switch(type) {
        case 'error':
            console.error(`${prefix} ❌ ${message}`);
            break;
        case 'warning':
            console.warn(`${prefix} ⚠️ ${message}`);
            break;
        case 'success':
            console.log(`${prefix} ✅ ${message}`);
            break;
        default:
            console.log(`${prefix} ℹ️ ${message}`);
    }
}

// ===== VARIABLES GLOBALES =====
let currentAccount = null;
let socket = null;
let currentRoom = 'general';

// ===== VERIFICACIÓN DE ELEMENTOS DOM =====
function checkDOMElements() {
    const requiredElements = [
        'connectWalletBtn',
        'walletStatus', 
        'accountAddress',
        'disconnectBtn',
        'roomsList',
        'messagesList',
        'messageInput',
        'sendBtn'
    ];
    
    const missing = [];
    requiredElements.forEach(id => {
        if (!document.getElementById(id)) {
            missing.push(id);
        }
    });
    
    if (missing.length > 0) {
        log(`Elementos DOM faltantes: ${missing.join(', ')}`, 'warning');
        return false;
    }
    
    log('Todos los elementos DOM requeridos están presentes', 'success');
    return true;
}

// ===== VERIFICACIÓN DE METAMASK =====
function checkMetaMaskAvailability() {
    log('Iniciando verificación de MetaMask...');
    
    if (typeof window.ethereum !== 'undefined') {
        log('Ethereum detectado: true', 'success');
        log(`Proveedor: ${window.ethereum.isMetaMask ? 'MetaMask' : 'Otro proveedor'}`, 'info');
        return true;
    } else {
        log('Ethereum NO detectado', 'error');
        return false;
    }
}

// ===== INICIALIZACIÓN =====
document.addEventListener('DOMContentLoaded', function() {
    log('DOM cargado, iniciando aplicación...');
    
    // Verificar elementos DOM
    if (!checkDOMElements()) {
        log('Faltan elementos DOM críticos, algunas funciones pueden no funcionar', 'warning');
    }
    
    // Verificar MetaMask
    if (!checkMetaMaskAvailability()) {
        updateWalletStatus('MetaMask no detectado', 'error');
        return;
    }
    
    // Configurar event listeners
    setupEventListeners();
    
    // Cargar salas
    loadRooms();
    
    log('Aplicación inicializada correctamente', 'success');
});

// ===== EVENT LISTENERS =====
function setupEventListeners() {
    log('Configurando event listeners...');
    
    // Botón conectar MetaMask
    const connectBtn = document.getElementById('connectWalletBtn');
    if (connectBtn) {
        connectBtn.addEventListener('click', connectWalletLogin);
        log('Event listener para conectar wallet configurado', 'success');
    }
    
    // Botón desconectar
    const disconnectBtn = document.getElementById('disconnectBtn');
    if (disconnectBtn) {
        disconnectBtn.addEventListener('click', disconnectWallet);
    }
    
    // Botón enviar mensaje
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    
    // Enter en input de mensaje
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
    
    // Detectar cambios de cuenta en MetaMask
    if (window.ethereum) {
        window.ethereum.on('accountsChanged', function(accounts) {
            log(`Cuentas cambiadas: ${accounts.length > 0 ? accounts[0] : 'ninguna'}`, 'info');
            if (accounts.length === 0) {
                disconnectWallet();
            } else if (accounts[0] !== currentAccount) {
                currentAccount = accounts[0];
                updateAccountDisplay(currentAccount);
            }
        });
        
        window.ethereum.on('chainChanged', function(chainId) {
            log(`Red cambiada: ${chainId}`, 'info');
            window.location.reload();
        });
    }
}

// ===== FUNCIÓN PRINCIPAL DE CONEXIÓN METAMASK =====
async function connectWalletLogin() {
    log('Iniciando proceso de conexión MetaMask...');
    
    try {
        // Verificar MetaMask
        if (!checkMetaMaskAvailability()) {
            throw new Error('MetaMask no está disponible');
        }
        
        updateWalletStatus('Conectando...', 'info');
        
        // Solicitar cuentas
        log('Solicitando cuentas...');
        const accounts = await window.ethereum.request({ 
            method: 'eth_requestAccounts' 
        });
        
        if (!accounts || accounts.length === 0) {
            throw new Error('No se obtuvieron cuentas');
        }
        
        const account = accounts[0];
        log(`Cuenta obtenida: ${account}`, 'success');
        
        // Crear mensaje para firmar
        const message = `Login to OpenAGI Secure Chat+\nAccount: ${account}\nTimestamp: ${Date.now()}`;
        log('Solicitando firma de mensaje...');
        
        // Firmar mensaje
        const signature = await window.ethereum.request({
            method: 'personal_sign',
            params: [message, account]
        });
        
        log('Mensaje firmado exitosamente', 'success');
        
        // Enviar al servidor
        log('Enviando datos al servidor...');
        const response = await fetch('/api.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                action: 'metamask_login',
                account: account,
                signature: signature,
                message: message
            })
        });
        
        const result = await response.json();
        log(`Respuesta del servidor: ${JSON.stringify(result)}`, 'info');
        
        if (result.success) {
            currentAccount = account;
            updateWalletStatus('Conectado', 'success');
            updateAccountDisplay(account);
            showConnectedState();
            connectWebSocket();
            loadMessages();
            log('Login completado exitosamente', 'success');
        } else {
            throw new Error(result.message || 'Error en el servidor');
        }
        
    } catch (error) {
        log(`Error en conexión: ${error.message}`, 'error');
        
        // Manejo específico de errores MetaMask
        if (error.code === 4001) {
            updateWalletStatus('Conexión rechazada por el usuario', 'warning');
        } else if (error.code === -32002) {
            updateWalletStatus('Solicitud pendiente en MetaMask', 'warning');
        } else if (error.message.includes('User rejected')) {
            updateWalletStatus('Usuario rechazó la solicitud', 'warning');
        } else {
            updateWalletStatus(`Error: ${error.message}`, 'error');
        }
    }
}

// ===== FUNCIONES DE UI =====
function updateWalletStatus(message, type = 'info') {
    const statusElement = document.getElementById('walletStatus');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `wallet-status ${type}`;
    }
    log(`Estado wallet actualizado: ${message}`, type);
}

function updateAccountDisplay(account) {
    const accountElement = document.getElementById('accountAddress');
    if (accountElement && account) {
        const shortAccount = `${account.substring(0, 6)}...${account.substring(account.length - 4)}`;
        accountElement.textContent = shortAccount;
        accountElement.title = account;
    }
}

function showConnectedState() {
    const connectBtn = document.getElementById('connectWalletBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');
    
    if (connectBtn) connectBtn.style.display = 'none';
    if (disconnectBtn) disconnectBtn.style.display = 'inline-block';
}

function showDisconnectedState() {
    const connectBtn = document.getElementById('connectWalletBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');
    const accountElement = document.getElementById('accountAddress');
    
    if (connectBtn) connectBtn.style.display = 'inline-block';
    if (disconnectBtn) disconnectBtn.style.display = 'none';
    if (accountElement) accountElement.textContent = '';
}

// ===== DESCONEXIÓN =====
function disconnectWallet() {
    log('Desconectando wallet...');
    currentAccount = null;
    updateWalletStatus('Desconectado', 'info');
    showDisconnectedState();
    
    if (socket) {
        socket.close();
        socket = null;
    }
    
    log('Wallet desconectado', 'success');
}

// ===== FUNCIONES DE SALAS =====
async function loadRooms() {
    try {
        log('Cargando salas...');
        const response = await fetch('/api.php?action=rooms');
        const rooms = await response.json();
        
        const roomsList = document.getElementById('roomsList');
        if (roomsList && rooms) {
            roomsList.innerHTML = '';
            rooms.forEach(room => {
                const roomElement = document.createElement('div');
                roomElement.className = 'room-item';
                roomElement.textContent = room.name;
                roomElement.onclick = () => selectRoom(room.id, room.name);
                roomsList.appendChild(roomElement);
            });
            log(`${rooms.length} salas cargadas`, 'success');
        }
    } catch (error) {
        log(`Error cargando salas: ${error.message}`, 'error');
    }
}

function selectRoom(roomId, roomName) {
    currentRoom = roomId;
    log(`Sala seleccionada: ${roomName} (${roomId})`);
    loadMessages();
}

// ===== FUNCIONES DE MENSAJES =====
async function loadMessages() {
    if (!currentRoom) return;
    
    try {
        log(`Cargando mensajes de sala: ${currentRoom}`);
        const response = await fetch(`/api.php?action=messages&room_id=${currentRoom}`);
        const messages = await response.json();
        
        const messagesList = document.getElementById('messagesList');
        if (messagesList && messages) {
            messagesList.innerHTML = '';
            messages.forEach(message => {
                addMessageToList(message);
            });
            messagesList.scrollTop = messagesList.scrollHeight;
            log(`${messages.length} mensajes cargados`, 'success');
        }
    } catch (error) {
        log(`Error cargando mensajes: ${error.message}`, 'error');
    }
}

function addMessageToList(message) {
    const messagesList = document.getElementById('messagesList');
    if (!messagesList) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = 'message';
    messageElement.innerHTML = `
        <div class="message-author">${message.author}</div>
        <div class="message-text">${message.text}</div>
        <div class="message-time">${message.timestamp}</div>
    `;
    
    messagesList.appendChild(messageElement);
    messagesList.scrollTop = messagesList.scrollHeight;
}

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    if (!messageInput || !currentAccount || !currentRoom) return;
    
    const text = messageInput.value.trim();
    if (!text) return;
    
    try {
        log(`Enviando mensaje: "${text}"`);
        const response = await fetch('/api.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                action: 'send_message',
                room_id: currentRoom,
                text: text,
                author: currentAccount
            })
        });
        
        const result = await response.json();
        if (result.success) {
            messageInput.value = '';
            log('Mensaje enviado exitosamente', 'success');
        } else {
            throw new Error(result.message || 'Error enviando mensaje');
        }
    } catch (error) {
        log(`Error enviando mensaje: ${error.message}`, 'error');
    }
}

// ===== WEBSOCKET =====
function connectWebSocket() {
    if (!currentAccount) return;
    
    try {
        log('Conectando WebSocket...');
        socket = new WebSocket(`ws://${window.location.host}/websocket.php`);
        
        socket.onopen = function() {
            log('WebSocket conectado', 'success');
        };
        
        socket.onmessage = function(event) {
            try {
                const message = JSON.parse(event.data);
                addMessageToList(message);
                log('Nuevo mensaje recibido via WebSocket', 'info');
            } catch (error) {
                log(`Error procesando mensaje WebSocket: ${error.message}`, 'error');
            }
        };
        
        socket.onclose = function() {
            log('WebSocket desconectado', 'warning');
        };
        
        socket.onerror = function(error) {
            log(`Error WebSocket: ${error}`, 'error');
        };
        
    } catch (error) {
        log(`Error conectando WebSocket: ${error.message}`, 'error');
    }
}

// ===== MANEJO DE ERRORES GLOBALES =====
window.addEventListener('error', function(event) {
    log(`Error JavaScript: ${event.error?.message || event.message}`, 'error');
});

window.addEventListener('unhandledrejection', function(event) {
    log(`Promesa rechazada: ${event.reason}`, 'error');
});

log('Script MetaMask simplificado cargado completamente', 'success');
```

---

## 🔹 ARCHIVO 2: `debug_console.html` (Consola de Debug)

**Ubicación en servidor:** `/opt/openagi/web/advanced-chat-php/public/debug_console.html`

**Instrucciones:** Crear nuevo archivo con el siguiente contenido:

```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 Debug Console - MetaMask OpenAGI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .panel {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .panel h3 {
            margin-bottom: 15px;
            color: #ffd700;
            font-size: 1.3em;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .status-item {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .status-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .status-value {
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .status-true { color: #4ade80; }
        .status-false { color: #f87171; }
        .status-warning { color: #fbbf24; }
        
        .btn {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            margin: 5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .btn-success {
            background: linear-gradient(45deg, #4ade80, #22c55e);
        }
        
        .btn-warning {
            background: linear-gradient(45deg, #fbbf24, #f59e0b);
        }
        
        .log-container {
            background: rgba(0,0,0,0.4);
            border-radius: 10px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .log-entry {
            margin-bottom: 8px;
            padding: 5px;
            border-radius: 4px;
            word-wrap: break-word;
        }
        
        .log-info { background: rgba(59, 130, 246, 0.2); }
        .log-success { background: rgba(34, 197, 94, 0.2); }
        .log-warning { background: rgba(245, 158, 11, 0.2); }
        .log-error { background: rgba(239, 68, 68, 0.2); }
        
        .full-width {
            grid-column: 1 / -1;
        }
        
        .export-btn {
            background: linear-gradient(45deg, #8b5cf6, #7c3aed);
            margin-top: 10px;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Debug Console MetaMask</h1>
            <p>Monitoreo avanzado y diagnóstico en tiempo real</p>
        </div>
        
        <div class="grid">
            <div class="panel">
                <h3>📊 Estado del Sistema</h3>
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-label">MetaMask</div>
                        <div class="status-value" id="metamask-status">Verificando...</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Ethereum</div>
                        <div class="status-value" id="ethereum-status">Verificando...</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Cuentas</div>
                        <div class="status-value" id="accounts-status">Verificando...</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Red</div>
                        <div class="status-value" id="network-status">Verificando...</div>
                    </div>
                </div>
                
                <button class="btn" onclick="checkSystemStatus()">🔄 Actualizar Estado</button>
                <button class="btn btn-success" onclick="checkPermissions()">🔐 Verificar Permisos</button>
            </div>
            
            <div class="panel">
                <h3>🧪 Pruebas MetaMask</h3>
                <button class="btn" onclick="testDetection()">🔍 Detectar MetaMask</button>
                <button class="btn" onclick="testConnection()">🔗 Conectar Cuentas</button>
                <button class="btn btn-warning" onclick="testSigning()">✍️ Firmar Mensaje</button>
                <button class="btn btn-success" onclick="testFullLogin()">🚀 Login Completo</button>
                
                <div style="margin-top: 15px;">
                    <strong>Cuenta Actual:</strong>
                    <div id="current-account" style="font-family: monospace; font-size: 0.9em; margin-top: 5px; word-break: break-all;">
                        No conectada
                    </div>
                </div>
            </div>
            
            <div class="panel full-width">
                <h3>📝 Logs en Tiempo Real</h3>
                <div class="log-container" id="log-container"></div>
                <button class="btn export-btn" onclick="exportLogs()">📤 Exportar Logs</button>
                <button class="btn" onclick="clearLogs()">🗑️ Limpiar</button>
            </div>
        </div>
    </div>

    <script>
        // ===== SISTEMA DE LOGGING =====
        let logs = [];
        
        function log(message, type = 'info') {
            const timestamp = new Date().toISOString();
            const logEntry = {
                timestamp,
                message,
                type,
                full: `[${timestamp}] [${type.toUpperCase()}] ${message}`
            };
            
            logs.push(logEntry);
            displayLog(logEntry);
            console.log(logEntry.full);
        }
        
        function displayLog(entry) {
            const container = document.getElementById('log-container');
            const logDiv = document.createElement('div');
            logDiv.className = `log-entry log-${entry.type}`;
            logDiv.textContent = entry.full;
            container.appendChild(logDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        function clearLogs() {
            logs = [];
            document.getElementById('log-container').innerHTML = '';
            log('Logs limpiados', 'info');
        }
        
        function exportLogs() {
            const logText = logs.map(log => log.full).join('\n');
            const blob = new Blob([logText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `metamask-debug-${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.txt`;
            a.click();
            URL.revokeObjectURL(url);
            log('Logs exportados', 'success');
        }
        
        // ===== FUNCIONES DE ESTADO =====
        function updateStatus(elementId, value, isGood = null) {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = value;
                element.className = 'status-value';
                if (isGood === true) element.classList.add('status-true');
                else if (isGood === false) element.classList.add('status-false');
                else element.classList.add('status-warning');
            }
        }
        
        async function checkSystemStatus() {
            log('🔄 Verificando estado del sistema...', 'info');
            
            // MetaMask
            const hasMetaMask = typeof window.ethereum !== 'undefined' && window.ethereum.isMetaMask;
            updateStatus('metamask-status', hasMetaMask ? 'Instalado' : 'No detectado', hasMetaMask);
            log(`MetaMask: ${hasMetaMask ? 'Detectado' : 'No detectado'}`, hasMetaMask ? 'success' : 'error');
            
            // Ethereum
            const hasEthereum = typeof window.ethereum !== 'undefined';
            updateStatus('ethereum-status', hasEthereum ? 'Disponible' : 'No disponible', hasEthereum);
            log(`Ethereum object: ${hasEthereum ? 'Disponible' : 'No disponible'}`, hasEthereum ? 'success' : 'error');
            
            if (hasEthereum) {
                try {
                    // Cuentas
                    const accounts = await window.ethereum.request({ method: 'eth_accounts' });
                    updateStatus('accounts-status', accounts.length > 0 ? `${accounts.length} cuenta(s)` : 'Sin conectar', accounts.length > 0);
                    log(`Cuentas: ${accounts.length} encontradas`, accounts.length > 0 ? 'success' : 'warning');
                    
                    if (accounts.length > 0) {
                        document.getElementById('current-account').textContent = accounts[0];
                    }
                    
                    // Red
                    const chainId = await window.ethereum.request({ method: 'eth_chainId' });
                    const networkName = getNetworkName(chainId);
                    updateStatus('network-status', networkName, true);
                    log(`Red: ${networkName} (${chainId})`, 'success');
                    
                } catch (error) {
                    updateStatus('accounts-status', 'Error', false);
                    updateStatus('network-status', 'Error', false);
                    log(`Error verificando estado: ${error.message}`, 'error');
                }
            }
        }
        
        function getNetworkName(chainId) {
            const networks = {
                '0x1': 'Ethereum Mainnet',
                '0x3': 'Ropsten Testnet',
                '0x4': 'Rinkeby Testnet',
                '0x5': 'Goerli Testnet',
                '0x89': 'Polygon Mainnet',
                '0x13881': 'Polygon Mumbai'
            };
            return networks[chainId] || `Red desconocida (${chainId})`;
        }
        
        async function checkPermissions() {
            log('🔐 Verificando permisos...', 'info');
            
            if (typeof window.ethereum === 'undefined') {
                log('MetaMask no disponible para verificar permisos', 'error');
                return;
            }
            
            try {
                const permissions = await window.ethereum.request({
                    method: 'wallet_getPermissions'
                });
                
                log(`Permisos encontrados: ${permissions.length}`, 'info');
                permissions.forEach((permission, index) => {
                    log(`Permiso ${index + 1}: ${permission.parentCapability}`, 'info');
                });
                
                if (permissions.length === 0) {
                    log('No hay permisos otorgados', 'warning');
                }
                
            } catch (error) {
                log(`Error verificando permisos: ${error.message}`, 'error');
            }
        }
        
        // ===== FUNCIONES DE PRUEBA =====
        async function testDetection() {
            log('🔍 Probando detección de MetaMask...', 'info');
            
            if (typeof window.ethereum !== 'undefined') {
                log('✅ window.ethereum detectado', 'success');
                
                if (window.ethereum.isMetaMask) {
                    log('✅ MetaMask confirmado', 'success');
                } else {
                    log('⚠️ Proveedor Ethereum detectado pero no es MetaMask', 'warning');
                }
                
                log(`Proveedor: ${window.ethereum.constructor.name}`, 'info');
            } else {
                log('❌ window.ethereum NO detectado', 'error');
                log('Posibles causas: MetaMask no instalado, extensión deshabilitada, o página no cargada completamente', 'warning');
            }
        }
        
        async function testConnection() {
            log('🔗 Probando conexión de cuentas...', 'info');
            
            if (typeof window.ethereum === 'undefined') {
                log('❌ MetaMask no disponible', 'error');
                return;
            }
            
            try {
                log('Solicitando cuentas...', 'info');
                const accounts = await window.ethereum.request({ 
                    method: 'eth_requestAccounts' 
                });
                
                if (accounts && accounts.length > 0) {
                    log(`✅ Conexión exitosa: ${accounts.length} cuenta(s)`, 'success');
                    log(`Cuenta principal: ${accounts[0]}`, 'success');
                    document.getElementById('current-account').textContent = accounts[0];
                } else {
                    log('❌ No se obtuvieron cuentas', 'error');
                }
                
            } catch (error) {
                log(`❌ Error conectando: ${error.message}`, 'error');
                
                if (error.code === 4001) {
                    log('Usuario rechazó la conexión', 'warning');
                } else if (error.code === -32002) {
                    log('Solicitud pendiente en MetaMask', 'warning');
                }
            }
        }
        
        async function testSigning() {
            log('✍️ Probando firma de mensaje...', 'info');
            
            if (typeof window.ethereum === 'undefined') {
                log('❌ MetaMask no disponible', 'error');
                return;
            }
            
            try {
                // Primero obtener cuentas
                const accounts = await window.ethereum.request({ method: 'eth_accounts' });
                
                if (!accounts || accounts.length === 0) {
                    log('❌ No hay cuentas conectadas. Conecta primero.', 'error');
                    return;
                }
                
                const account = accounts[0];
                const message = `Prueba de firma MetaMask\nCuenta: ${account}\nTimestamp: ${Date.now()}`;
                
                log('Solicitando firma...', 'info');
                log(`Mensaje a firmar: ${message}`, 'info');
                
                const signature = await window.ethereum.request({
                    method: 'personal_sign',
                    params: [message, account]
                });
                
                log('✅ Mensaje firmado exitosamente', 'success');
                log(`Firma: ${signature.substring(0, 20)}...`, 'success');
                
            } catch (error) {
                log(`❌ Error firmando: ${error.message}`, 'error');
                
                if (error.code === 4001) {
                    log('Usuario rechazó la firma', 'warning');
                }
            }
        }
        
        async function testFullLogin() {
            log('🚀 Probando login completo (simulando sistema principal)...', 'info');
            
            if (typeof window.ethereum === 'undefined') {
                log('❌ MetaMask no disponible', 'error');
                return;
            }
            
            try {
                // Paso 1: Conectar cuentas
                log('Paso 1: Conectando cuentas...', 'info');
                const accounts = await window.ethereum.request({ 
                    method: 'eth_requestAccounts' 
                });
                
                if (!accounts || accounts.length === 0) {
                    throw new Error('No se obtuvieron cuentas');
                }
                
                const account = accounts[0];
                log(`✅ Cuenta obtenida: ${account}`, 'success');
                
                // Paso 2: Crear mensaje
                const message = `Login to OpenAGI Secure Chat+\nAccount: ${account}\nTimestamp: ${Date.now()}`;
                log('Paso 2: Creando mensaje para firmar...', 'info');
                
                // Paso 3: Firmar mensaje
                log('Paso 3: Solicitando firma...', 'info');
                const signature = await window.ethereum.request({
                    method: 'personal_sign',
                    params: [message, account]
                });
                
                log('✅ Mensaje firmado', 'success');
                
                // Paso 4: Simular envío al servidor
                log('Paso 4: Simulando envío al servidor...', 'info');
                
                const response = await fetch('/api.php', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        action: 'metamask_login',
                        account: account,
                        signature: signature,
                        message: message
                    })
                });
                
                const result = await response.json();
                log(`Respuesta del servidor: ${JSON.stringify(result)}`, 'info');
                
                if (result.success) {
                    log('🎉 LOGIN COMPLETO EXITOSO', 'success');
                    document.getElementById('current-account').textContent = account;
                } else {
                    log(`❌ Error del servidor: ${result.message}`, 'error');
                }
                
            } catch (error) {
                log(`❌ Error en login completo: ${error.message}`, 'error');
                
                if (error.code === 4001) {
                    log('Usuario rechazó alguna solicitud', 'warning');
                } else if (error.code === -32002) {
                    log('Solicitud pendiente en MetaMask', 'warning');
                }
            }
        }
        
        // ===== CAPTURA DE ERRORES =====
        window.addEventListener('error', function(event) {
            log(`❌ Error JavaScript: ${event.error?.message || event.message}`, 'error');
        });
        
        window.addEventListener('unhandledrejection', function(event) {
            log(`❌ Promesa rechazada: ${event.reason}`, 'error');
        });
        
        // ===== EVENTOS METAMASK =====
        if (typeof window.ethereum !== 'undefined') {
            window.ethereum.on('accountsChanged', function(accounts) {
                log(`🔄 Cuentas cambiadas: ${accounts.length > 0 ? accounts[0] : 'ninguna'}`, 'info');
                checkSystemStatus();
            });
            
            window.ethereum.on('chainChanged', function(chainId) {
                log(`🔄 Red cambiada: ${getNetworkName(chainId)}`, 'info');
                checkSystemStatus();
            });
            
            window.ethereum.on('connect', function(connectInfo) {
                log(`🔗 MetaMask conectado: ${connectInfo.chainId}`, 'success');
            });
            
            window.ethereum.on('disconnect', function(error) {
                log(`🔌 MetaMask desconectado: ${error.message}`, 'warning');
            });
        }
        
        // ===== INICIALIZACIÓN =====
        document.addEventListener('DOMContentLoaded', function() {
            log('🚀 Debug Console inicializada', 'success');
            checkSystemStatus();
        });
    </script>
</body>
</html>
```

---

## 📋 INSTRUCCIONES DE IMPLEMENTACIÓN

### 1. **Conectar al servidor:**
```bash
ssh root@77.237.235.224
cd /opt/openagi/web/advanced-chat-php/public
```

### 2. **Crear backup:**
```bash
cp app_fixed.js app_fixed.js.backup.$(date +%Y%m%d_%H%M%S)
```

### 3. **Implementar app_fixed.js:**
```bash
nano app_fixed.js
# Borrar todo el contenido y pegar el código JavaScript de arriba
# Guardar con Ctrl+X, Y, Enter
```

### 4. **Implementar debug_console.html:**
```bash
nano debug_console.html
# Pegar el código HTML completo de arriba
# Guardar con Ctrl+X, Y, Enter
```

### 5. **Verificar implementación:**
```bash
ls -la app_fixed.js debug_console.html
curl -I http://127.0.0.1:8087/
curl -I http://127.0.0.1:8087/debug_console.html
```

### 6. **Probar funcionamiento:**
- **Sistema principal:** http://77.237.235.224:8087/
- **Consola debug:** http://77.237.235.224:8087/debug_console.html

---

**🎯 ¡Estas correcciones resuelven los problemas identificados en la investigación profunda!**