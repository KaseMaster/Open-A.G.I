/**
 * OpenAGI Secure Chat+ - JavaScript Frontend
 * Programador Principal: Jose Gómez alias KaseMaster
 * Contacto: kasemaster@protonmail.com
 * Versión: 2.0.0
 * Licencia: MIT
 */

document.addEventListener("DOMContentLoaded", function() {
  const roomsEl = document.getElementById('rooms');
  const messagesEl = document.getElementById('messages');
  const chatTitleEl = document.getElementById('chatTitle');
  const chatSubEl = document.getElementById('chatSub');
  const messageInputEl = document.getElementById('messageInput');
  const fileInputEl = document.getElementById('fileInput');
  const sendBtnEl = document.getElementById('sendBtn');
  const fileBtnEl = document.getElementById('fileBtn');
  const btnLoginEl = document.getElementById('btnLogin');
  const btnLogoutEl = document.getElementById('btnLogout');
  const userInfoEl = document.getElementById('userInfo');
  const loginFormEl = document.getElementById('loginForm');
  const chatContainerEl = document.getElementById('chatContainer');

  let currentRoom = 'general';
  let currentUser = null;
  let ws = null;
  let isConnected = false;

  // MetaMask wallet connection
  async function connectWalletLogin() {
    if (typeof window.ethereum !== 'undefined') {
      try {
        // Request account access
        const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
        
        if (accounts.length > 0) {
          const account = accounts[0];
          
          // Create a message to sign
          const message = `Login to OpenAGI Chat at ${new Date().toISOString()}`;
          
          try {
            // Request signature
            const signature = await window.ethereum.request({
              method: 'personal_sign',
              params: [message, account]
            });
            
            // Verify signature on server
            const response = await fetch('/api.php', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
              },
              body: `action=wallet_login&address=${account}&message=${encodeURIComponent(message)}&signature=${signature}`
            });
            
            const result = await response.json();
            
            if (result.success) {
              currentUser = {
                address: account,
                name: `User_${account.substring(0, 8)}...`
              };
              
              showChat();
              connectWebSocket();
            } else {
              alert('Error de autenticación: ' + result.error);
            }
          } catch (signError) {
            console.error('Error signing message:', signError);
            alert('Error al firmar el mensaje');
          }
        }
      } catch (error) {
        console.error('Error connecting to MetaMask:', error);
        alert('Error conectando con MetaMask');
      }
    } else {
      alert('MetaMask no está instalado. Por favor instala MetaMask para continuar.');
    }
  }

  // Traditional login (fallback)
  async function traditionalLogin() {
    const username = prompt('Ingresa tu nombre de usuario:');
    if (username && username.trim()) {
      currentUser = {
        name: username.trim(),
        address: null
      };
      showChat();
      connectWebSocket();
    }
  }

  function showLogin() {
    if (loginFormEl) loginFormEl.style.display = 'block';
    if (chatContainerEl) chatContainerEl.style.display = 'none';
    if (btnLoginEl) btnLoginEl.style.display = 'none';
    if (btnLogoutEl) btnLogoutEl.style.display = 'none';
  }

  function showChat() {
    if (loginFormEl) loginFormEl.style.display = 'none';
    if (chatContainerEl) chatContainerEl.style.display = 'block';
    if (btnLoginEl) btnLoginEl.style.display = 'none';
    if (btnLogoutEl) btnLogoutEl.style.display = 'inline-block';
    
    if (userInfoEl && currentUser) {
      userInfoEl.textContent = currentUser.name;
    }
    
    loadRooms();
    loadMessages();
  }

  function logout() {
    currentUser = null;
    if (ws) {
      ws.close();
      ws = null;
    }
    isConnected = false;
    showLogin();
  }

  // WebSocket connection
  function connectWebSocket() {
    if (ws) {
      ws.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/websocket.php`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
      console.log('WebSocket connected');
      isConnected = true;
      
      // Join current room
      ws.send(JSON.stringify({
        type: 'join_room',
        room: currentRoom,
        user: currentUser
      }));
    };
    
    ws.onmessage = function(event) {
      const data = JSON.parse(event.data);
      
      if (data.type === 'new_message') {
        displayMessage(data.message);
      } else if (data.type === 'room_update') {
        loadRooms();
      }
    };
    
    ws.onclose = function() {
      console.log('WebSocket disconnected');
      isConnected = false;
      
      // Attempt to reconnect after 3 seconds
      setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = function(error) {
      console.error('WebSocket error:', error);
    };
  }

  // Load rooms
  async function loadRooms() {
    try {
      const response = await fetch('/api.php?action=rooms');
      const rooms = await response.json();
      
      if (roomsEl) {
        roomsEl.innerHTML = '';
        rooms.forEach(room => {
          const roomEl = document.createElement('div');
          roomEl.className = 'room-item';
          roomEl.textContent = room.name;
          roomEl.onclick = () => selectRoom(room.id, room.name);
          
          if (room.id === currentRoom) {
            roomEl.classList.add('active');
          }
          
          roomsEl.appendChild(roomEl);
        });
      }
    } catch (error) {
      console.error('Error loading rooms:', error);
    }
  }

  // Select room
  function selectRoom(roomId, roomName) {
    currentRoom = roomId;
    
    if (chatTitleEl) chatTitleEl.textContent = roomName;
    if (chatSubEl) chatSubEl.textContent = `Sala: ${roomName}`;
    
    // Update active room in UI
    const roomItems = document.querySelectorAll('.room-item');
    roomItems.forEach(item => {
      item.classList.remove('active');
      if (item.textContent === roomName) {
        item.classList.add('active');
      }
    });
    
    loadMessages();
    
    // Join room via WebSocket
    if (ws && isConnected) {
      ws.send(JSON.stringify({
        type: 'join_room',
        room: currentRoom,
        user: currentUser
      }));
    }
  }

  // Load messages
  async function loadMessages() {
    try {
      const response = await fetch(`/api.php?action=messages&room_id=${currentRoom}`);
      const messages = await response.json();
      
      if (messagesEl) {
        messagesEl.innerHTML = '';
        messages.forEach(message => {
          displayMessage(message);
        });
        
        // Scroll to bottom
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    } catch (error) {
      console.error('Error loading messages:', error);
    }
  }

  // Display message
  function displayMessage(message) {
    if (!messagesEl) return;
    
    const messageEl = document.createElement('div');
    messageEl.className = 'message';
    
    const timeStr = new Date(message.timestamp).toLocaleTimeString();
    
    if (message.file_url) {
      // File message
      messageEl.innerHTML = `
        <div class="message-header">
          <span class="author">${message.author}</span>
          <span class="time">${timeStr}</span>
        </div>
        <div class="message-content">
          <a href="${message.file_url}" target="_blank" class="file-link">
            📎 ${message.text || 'Archivo adjunto'}
          </a>
        </div>
      `;
    } else {
      // Text message
      messageEl.innerHTML = `
        <div class="message-header">
          <span class="author">${message.author}</span>
          <span class="time">${timeStr}</span>
        </div>
        <div class="message-content">${message.text}</div>
      `;
    }
    
    messagesEl.appendChild(messageEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // Send message
  async function sendMessage() {
    if (!messageInputEl || !currentUser) return;
    
    const text = messageInputEl.value.trim();
    if (!text) return;
    
    try {
      const response = await fetch('/api.php', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `action=send_message&room_id=${currentRoom}&text=${encodeURIComponent(text)}&author=${encodeURIComponent(currentUser.name)}`
      });
      
      const result = await response.json();
      
      if (result.success) {
        messageInputEl.value = '';
        
        // Send via WebSocket for real-time update
        if (ws && isConnected) {
          ws.send(JSON.stringify({
            type: 'new_message',
            room: currentRoom,
            message: {
              text: text,
              author: currentUser.name,
              timestamp: new Date().toISOString()
            }
          }));
        }
      } else {
        alert('Error enviando mensaje: ' + result.error);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      alert('Error enviando mensaje');
    }
  }

  // Send file
  async function sendFile() {
    if (!fileInputEl || !currentUser) return;
    
    const file = fileInputEl.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('action', 'send_file');
    formData.append('room_id', currentRoom);
    formData.append('file', file);
    formData.append('author', currentUser.name);
    
    try {
      const response = await fetch('/api.php', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (result.success) {
        fileInputEl.value = '';
        
        // Send via WebSocket for real-time update
        if (ws && isConnected) {
          ws.send(JSON.stringify({
            type: 'new_message',
            room: currentRoom,
            message: {
              text: file.name,
              author: currentUser.name,
              timestamp: new Date().toISOString(),
              file_url: result.file_url
            }
          }));
        }
      } else {
        alert('Error enviando archivo: ' + result.error);
      }
    } catch (error) {
      console.error('Error sending file:', error);
      alert('Error enviando archivo');
    }
  }

  // Event listeners
  if (btnLoginEl) {
    btnLoginEl.onclick = connectWalletLogin;
  }
  
  if (btnLogoutEl) {
    btnLogoutEl.onclick = logout;
  }
  
  if (sendBtnEl) {
    sendBtnEl.onclick = sendMessage;
  }
  
  if (fileBtnEl) {
    fileBtnEl.onclick = sendFile;
  }
  
  if (messageInputEl) {
    messageInputEl.addEventListener('keypress', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  // Initialize app
  showLogin();
  
  // Update UI based on login state
  if (currentUser) {
    if (btnLoginEl) btnLoginEl.style.display = 'none';
    if (btnLogoutEl) btnLogoutEl.style.display = '';
  } else {
    if (btnLogoutEl) btnLogoutEl.style.display = 'none';
  }
});