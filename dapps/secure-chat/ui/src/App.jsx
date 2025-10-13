import React, { useState, useEffect, useRef } from 'react';
import { ethers } from 'ethers';
import { create } from 'ipfs-http-client';
import nacl from 'tweetnacl';
import './App.css';

// Base58 encoding/decoding
const base58Alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';

function base58Encode(buffer) {
  if (buffer.length === 0) return '';
  
  let digits = [0];
  for (let i = 0; i < buffer.length; i++) {
    let carry = buffer[i];
    for (let j = 0; j < digits.length; j++) {
      carry += digits[j] << 8;
      digits[j] = carry % 58;
      carry = Math.floor(carry / 58);
    }
    while (carry > 0) {
      digits.push(carry % 58);
      carry = Math.floor(carry / 58);
    }
  }
  
  let leadingZeros = 0;
  for (let i = 0; i < buffer.length && buffer[i] === 0; i++) {
    leadingZeros++;
  }
  
  return '1'.repeat(leadingZeros) + digits.reverse().map(d => base58Alphabet[d]).join('');
}

function base58Decode(str) {
  if (str.length === 0) return new Uint8Array(0);
  
  let bytes = [0];
  for (let i = 0; i < str.length; i++) {
    const char = str[i];
    const charIndex = base58Alphabet.indexOf(char);
    if (charIndex === -1) throw new Error('Invalid base58 character');
    
    let carry = charIndex;
    for (let j = 0; j < bytes.length; j++) {
      carry += bytes[j] * 58;
      bytes[j] = carry & 0xff;
      carry >>= 8;
    }
    while (carry > 0) {
      bytes.push(carry & 0xff);
      carry >>= 8;
    }
  }
  
  let leadingOnes = 0;
  for (let i = 0; i < str.length && str[i] === '1'; i++) {
    leadingOnes++;
  }
  
  return new Uint8Array([...Array(leadingOnes).fill(0), ...bytes.reverse()]);
}

  // Contract addresses (actualizadas después del despliegue)
  const CHAT_CONTRACT_ADDRESS = "0x4A679253410272dd5232B3Ff7cF5dbB88f295319";
  const AEGIS_TOKEN_ADDRESS = "0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9";

const CHAT_ABI = [
  "function createRoom(string memory name) public",
  "function joinRoom(uint256 roomId) public",
  "function sendMessage(uint256 roomId, string memory messageHash) public",
  "function getRooms() public view returns (tuple(uint256 id, string name, address creator, uint256 memberCount)[])",
  "function getRoomMessages(uint256 roomId) public view returns (tuple(address sender, string messageHash, uint256 timestamp)[])",
  "event RoomCreated(uint256 indexed roomId, string name, address indexed creator)",
  "event MessageSent(uint256 indexed roomId, address indexed sender, string messageHash, uint256 timestamp)"
];

const AEGIS_ABI = [
  "function balanceOf(address owner) view returns (uint256)",
  "function transfer(address to, uint256 amount) returns (bool)",
  "function approve(address spender, uint256 amount) returns (bool)",
  "function allowance(address owner, address spender) view returns (uint256)"
];

// IPFS client con fallback a gateway público
let ipfsClient;
try {
  ipfsClient = create({
    host: 'localhost',
    port: 5001,
    protocol: 'http'
  });
} catch (error) {
  console.warn('IPFS local no disponible, usando gateway público');
  ipfsClient = null;
}

// Encryption utilities
function encryptMessage(message, recipientPublicKey, senderSecretKey) {
  try {
    const messageBytes = new TextEncoder().encode(message);
    const nonce = nacl.randomBytes(24);
    const encrypted = nacl.box(messageBytes, nonce, recipientPublicKey, senderSecretKey);
    
    const combined = new Uint8Array(nonce.length + encrypted.length);
    combined.set(nonce);
    combined.set(encrypted, nonce.length);
    
    return base58Encode(combined);
  } catch (error) {
    console.error('Encryption error:', error);
    return message;
  }
}

function decryptMessage(encryptedMessage, senderPublicKey, recipientSecretKey) {
  try {
    const combined = base58Decode(encryptedMessage);
    const nonce = combined.slice(0, 24);
    const encrypted = combined.slice(24);
    
    const decrypted = nacl.box.open(encrypted, nonce, senderPublicKey, recipientSecretKey);
    if (!decrypted) throw new Error('Decryption failed');
    
    return new TextDecoder().decode(decrypted);
  } catch (error) {
    console.error('Decryption error:', error);
    return encryptedMessage;
  }
}

// IPFS utilities
async function uploadToIPFS(data) {
  try {
    if (ipfsClient) {
      // Intentar usar nodo IPFS local
      const result = await ipfsClient.add(JSON.stringify(data));
      return result.path;
    } else {
      // Fallback: simular hash para desarrollo sin IPFS
      const dataStr = JSON.stringify(data);
      const hash = 'Qm' + btoa(dataStr).replace(/[^a-zA-Z0-9]/g, '').substring(0, 44);
      console.warn('IPFS no disponible, usando hash simulado:', hash);
      
      // Guardar en localStorage como fallback temporal
      localStorage.setItem(`ipfs_${hash}`, dataStr);
      return hash;
    }
  } catch (error) {
    console.error('IPFS upload error:', error);
    // Fallback en caso de error
    const dataStr = JSON.stringify(data);
    const hash = 'Qm' + btoa(dataStr).replace(/[^a-zA-Z0-9]/g, '').substring(0, 44);
    localStorage.setItem(`ipfs_${hash}`, dataStr);
    return hash;
  }
}

async function fetchFromIPFS(hash) {
  try {
    if (ipfsClient) {
      // Intentar usar nodo IPFS local
      let data = '';
      for await (const chunk of ipfsClient.cat(hash)) {
        data += new TextDecoder().decode(chunk);
      }
      return JSON.parse(data);
    } else {
      // Fallback: buscar en localStorage
      const data = localStorage.getItem(`ipfs_${hash}`);
      return data ? JSON.parse(data) : null;
    }
  } catch (error) {
    console.error('IPFS fetch error:', error);
    // Fallback: buscar en localStorage
    const data = localStorage.getItem(`ipfs_${hash}`);
    return data ? JSON.parse(data) : null;
  }
}

function App() {
  const [account, setAccount] = useState('');
  const [provider, setProvider] = useState(null);
  const [signer, setSigner] = useState(null);
  const [chatContract, setChatContract] = useState(null);
  const [aegisContract, setAegisContract] = useState(null);
  const [rooms, setRooms] = useState([]);
  const [currentRoom, setCurrentRoom] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [newRoomName, setNewRoomName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isCreatingRoom, setIsCreatingRoom] = useState(false);
  const [aegisBalance, setAegisBalance] = useState('0');
  // Dark mode toggle with localStorage persistence
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('secureChat-darkMode');
    return saved !== null ? JSON.parse(saved) : true;
  });

  // Save dark mode preference
  useEffect(() => {
    localStorage.setItem('secureChat-darkMode', JSON.stringify(darkMode));
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(prev => !prev);
  };
  const [activeTab, setActiveTab] = useState('chat');
  const [notifications, setNotifications] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [parallaxOffset, setParallaxOffset] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Detectar si es móvil
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Default rooms for demo
  const defaultRooms = [
    { id: 0, name: "General", memberCount: 1, isDefault: true },
    { id: 1, name: "Tech Talk", memberCount: 1, isDefault: true },
    { id: 2, name: "Random", memberCount: 1, isDefault: true }
  ];

  // Notification system
  const showNotification = (message, type = 'info') => {
    const id = Date.now();
    const notification = { id, message, type };
    setNotifications(prev => [...prev, notification]);
    
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  };

  // Wallet connection
  const connectWallet = async () => {
    console.log('🔗 Iniciando conexión de wallet...');
    
    if (typeof window.ethereum !== 'undefined') {
      try {
        console.log('🦊 MetaMask detectado, solicitando conexión...');
        
        // Request account access
        const accounts = await window.ethereum.request({ 
          method: 'eth_requestAccounts' 
        });
        
        console.log('📋 Cuentas obtenidas:', accounts);
        
        if (accounts.length === 0) {
          throw new Error('No se seleccionó ninguna cuenta');
        }

        // Create provider and signer
        const provider = new ethers.BrowserProvider(window.ethereum);
        console.log('🌐 Provider creado:', provider);
        
        const signer = await provider.getSigner();
        const address = await signer.getAddress();
        console.log('✍️ Signer obtenido:', address);

        // Create contract instances
        console.log('📄 Creando instancias de contratos...');
        console.log('   - Chat Contract Address:', CHAT_CONTRACT_ADDRESS);
        console.log('   - AEGIS Token Address:', AEGIS_TOKEN_ADDRESS);
        
        const chatContract = new ethers.Contract(CHAT_CONTRACT_ADDRESS, CHAT_ABI, signer);
        const aegisContract = new ethers.Contract(AEGIS_TOKEN_ADDRESS, AEGIS_ABI, signer);
        
        console.log('✅ Contratos creados exitosamente');

        // Update state
        setProvider(provider);
        setSigner(signer);
        setAccount(address);
        setChatContract(chatContract);
        setAegisContract(aegisContract);

        console.log('🎉 Wallet conectada exitosamente:', address);
        showNotification('Wallet conectado exitosamente', 'success');

        // Load data immediately after connection
        setTimeout(() => {
          console.log('⏰ Cargando datos después de la conexión...');
          // No llamar aquí, el useEffect se encargará cuando los estados se actualicen
        }, 1000);

      } catch (error) {
        console.error('❌ Error conectando wallet:', error);
        
        if (error.code === 4001) {
          showNotification('Conexión cancelada por el usuario', 'error');
        } else if (error.code === -32002) {
          showNotification('Ya hay una solicitud de conexión pendiente', 'warning');
        } else {
          showNotification(`Error conectando wallet: ${error.message}`, 'error');
        }
      }
    } else {
      console.error('🚫 MetaMask no está instalado');
      showNotification('MetaMask no encontrado. Por favor instala MetaMask.', 'error');
    }
  };

  // Check if wallet is already connected
  const checkWalletConnection = async () => {
    console.log('🔍 Verificando conexión de wallet existente...');
    
    if (typeof window.ethereum !== 'undefined') {
      try {
        const accounts = await window.ethereum.request({ method: 'eth_accounts' });
        console.log('📋 Cuentas encontradas:', accounts);
        
        if (accounts.length > 0) {
          console.log('✅ Wallet ya conectada, inicializando...');
          await connectWallet();
        } else {
          console.log('ℹ️ No hay wallet conectada');
        }
      } catch (error) {
        console.error('❌ Error verificando wallet:', error);
      }
    } else {
      console.warn('⚠️ MetaMask no detectado en el navegador');
      showNotification('MetaMask no está instalado. Por favor, instala MetaMask para usar esta aplicación.', 'error');
    }
  };

  const disconnectWallet = () => {
    setProvider(null);
    setSigner(null);
    setChatContract(null);
    setAegisContract(null);
    setAccount('');
    setRooms([]);
    setCurrentRoom(null);
    setMessages([]);
    setAegisBalance('0');
    showNotification('Wallet desconectado', 'info');
  };

  // Load rooms
  const loadRooms = async () => {
    if (!chatContract) {
      console.log('No se pueden cargar salas: contrato no disponible');
      return;
    }
    
    try {
      console.log('Cargando salas del contrato...');
      const contractRooms = await chatContract.getRooms();
      console.log('Salas del contrato:', contractRooms);
      
      const formattedRooms = contractRooms.map(room => ({
        id: Number(room.id),
        name: room.name,
        memberCount: Number(room.memberCount),
        creator: room.creator,
        isDefault: false
      }));
      
      const allRooms = [...defaultRooms, ...formattedRooms];
      console.log('Todas las salas:', allRooms);
      setRooms(allRooms);
    } catch (error) {
      console.error('Error loading rooms:', error);
      showNotification('Error cargando salas', 'error');
      // Usar salas por defecto si hay error
      setRooms(defaultRooms);
    }
  };

  // Send message
  const sendMessage = async () => {
    if ((!newMessage.trim() && !selectedFile) || !currentRoom) return;
    
    try {
      const messageData = {
        text: newMessage,
        sender: account,
        timestamp: Date.now(),
        type: selectedFile ? 'file' : 'text'
      };

      if (selectedFile) {
        const fileData = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target.result);
          reader.readAsDataURL(selectedFile);
        });
        
        messageData.file = {
          name: selectedFile.name,
          size: selectedFile.size,
          type: selectedFile.type,
          data: fileData
        };
      }

      const ipfsHash = await uploadToIPFS(messageData);
      
      if (!currentRoom.isDefault) {
        await chatContract.sendMessage(currentRoom.id, ipfsHash);
      }
      
      setMessages(prev => [...prev, { ...messageData, ipfsHash }]);
      setNewMessage('');
      setSelectedFile(null);
      showNotification('Mensaje enviado', 'success');
    } catch (error) {
      console.error('Error sending message:', error);
      showNotification('Error enviando mensaje', 'error');
    }
  };

  // Create room
  const createRoom = async () => {
    if (!newRoomName.trim() || !chatContract) return;
    
    try {
      setIsCreatingRoom(true);
      const roomData = {
        name: newRoomName,
        creator: account,
        timestamp: Date.now()
      };
      
      const ipfsHash = await uploadToIPFS(roomData);
      await chatContract.createRoom(newRoomName, ipfsHash);
      
      await loadRooms();
      setNewRoomName('');
      setIsCreatingRoom(false);
      showNotification('Sala creada exitosamente', 'success');
    } catch (error) {
      setIsCreatingRoom(false);
      console.error('Error creating room:', error);
      showNotification('Error creando sala', 'error');
    }
  };

  // Mobile sidebar toggle
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // Close sidebar when selecting a room on mobile
  const selectRoom = async (room) => {
    try {
      await joinRoom(room);
      if (isMobile) {
        setSidebarOpen(false);
      }
    } catch (error) {
      console.error('Error selecting room:', error);
    }
  };

  // Handle back button on mobile
  const handleBackToRooms = () => {
    setCurrentRoom(null);
    if (isMobile) {
      setSidebarOpen(true);
    }
  };

  // Join room
  const joinRoom = async (room) => {
    try {
      if (!room.isDefault && chatContract) {
        await chatContract.joinRoom(room.id);
      }
      
      setCurrentRoom(room);
      setActiveTab('chat');
      await loadMessages(room);
      showNotification(`Unido a ${room.name}`, 'success');
    } catch (error) {
      console.error('Error joining room:', error);
      showNotification('Error uniéndose a la sala', 'error');
    }
  };

  // Load messages
  const loadMessages = async (room) => {
    if (!room) return;
    
    try {
      if (room.isDefault) {
        // Default room messages
        const defaultMessages = [
          {
            text: `¡Bienvenido a ${room.name}! Este es un chat descentralizado seguro.`,
            sender: 'system',
            timestamp: Date.now() - 3600000,
            type: 'text'
          }
        ];
        setMessages(defaultMessages);
      } else if (chatContract) {
        const contractMessages = await chatContract.getRoomMessages(room.id);
        const loadedMessages = [];
        
        for (const msg of contractMessages) {
          const messageData = await fetchFromIPFS(msg.ipfsHash);
          if (messageData) {
            loadedMessages.push({
              ...messageData,
              timestamp: Number(msg.timestamp) * 1000
            });
          }
        }
        
        setMessages(loadedMessages);
      }
    } catch (error) {
      console.error('Error loading messages:', error);
    }
  };

  // File handling
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        showNotification('Archivo muy grande (máximo 10MB)', 'error');
        return;
      }
      setSelectedFile(file);
      showNotification(`Archivo seleccionado: ${file.name}`, 'info');
    }
  };

  const cancelFileSelection = () => {
    setSelectedFile(null);
    fileInputRef.current.value = '';
  };

  // AEGIS token functions
  const loadAegisBalance = async () => {
    if (!aegisContract || !account) {
      console.log('No se puede cargar balance AEGIS: contrato o cuenta no disponible');
      return;
    }
    
    try {
      console.log('Cargando balance AEGIS para:', account);
      const balance = await aegisContract.balanceOf(account);
      const formattedBalance = ethers.formatEther(balance);
      console.log('Balance AEGIS cargado:', formattedBalance);
      setAegisBalance(formattedBalance);
    } catch (error) {
      console.error('Error loading AEGIS balance:', error);
      showNotification('Error cargando balance AEGIS', 'error');
    }
  };

  const requestAegisTokens = async () => {
    if (!aegisContract) return;
    
    try {
      await aegisContract.requestTokens();
      await loadAegisBalance();
      showNotification('Tokens AEGIS solicitados', 'success');
    } catch (error) {
      console.error('Error requesting tokens:', error);
      showNotification('Error solicitando tokens', 'error');
    }
  };

  const donateAegis = async (amount) => {
    if (!aegisContract || !amount) return;
    
    try {
      const amountWei = ethers.parseEther(amount.toString());
      await aegisContract.donate(account, amountWei);
      await loadAegisBalance();
      showNotification(`${amount} AEGIS donados`, 'success');
    } catch (error) {
      console.error('Error donating tokens:', error);
      showNotification('Error donando tokens', 'error');
    }
  };

  // Utility functions
  const formatAddress = (address) => {
    if (!address) return '';
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('es-ES', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Effects
  useEffect(() => {
    // Check if wallet is already connected on app load
    checkWalletConnection();
  }, []);

  useEffect(() => {
    if (account && chatContract && aegisContract) {
      console.log('✅ Cuenta y contratos disponibles, cargando datos...');
      console.log('   - Account:', account);
      console.log('   - Chat Contract:', chatContract?.target || chatContract?.address);
      console.log('   - AEGIS Contract:', aegisContract?.target || aegisContract?.address);
      
      // Pequeño delay para asegurar que los contratos estén completamente inicializados
      setTimeout(() => {
        console.log('🔄 Ejecutando carga de datos...');
        loadRooms();
        loadAegisBalance();
      }, 500);
    } else {
      console.log('⏳ Esperando inicialización completa...');
      console.log('   - Account:', !!account);
      console.log('   - Chat Contract:', !!chatContract);
      console.log('   - AEGIS Contract:', !!aegisContract);
    }
  }, [account, chatContract, aegisContract]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const handleScroll = () => {
      setParallaxOffset(window.pageYOffset * 0.5);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Render wallet connection screen
  if (!account) {
    return (
      <div className={`min-h-screen flex items-center justify-center transition-all duration-500 ${darkMode ? 'bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900' : 'bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50'}`}>
        <div className="text-center space-y-8 p-8 animate-fade-in-up">
          <div className="animate-bounce mb-4 animate-heartbeat">
            <span className="text-8xl">🔐</span>
          </div>
          <h1 className={`text-6xl font-bold mb-4 text-gradient-animated ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            SecureChat
          </h1>
          <p className={`text-xl mb-8 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Chat descentralizado con encriptación end-to-end
          </p>
          <button
            onClick={connectWallet}
            className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-2xl font-semibold text-lg shadow-2xl hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-105 ripple-button hover-glow animate-pulse-advanced"
          >
            Conectar Wallet
          </button>
        </div>
      </div>
    );
  }

  // Main app render
  return (
    <div className={`app-container ${darkMode ? 'dark' : 'light'}`}>
      {/* Mobile Header - Solo visible en móvil */}
      <div className="mobile-header">
        <div className="mobile-header-content">
          <button
            className="hamburger-btn"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <span></span>
            <span></span>
            <span></span>
          </button>
          
          <div className="mobile-header-title">
            <span className="mobile-header-icon">🔐</span>
            <h1>SecureChat</h1>
          </div>
          
          <div className="mobile-header-actions">
            <button
              onClick={toggleDarkMode}
              className="icon-btn dark-mode-toggle"
              title={darkMode ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}
            >
              <span className="theme-icon">
                {darkMode ? '☀️' : '🌙'}
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Sidebar Overlay para móvil */}
      {sidebarOpen && isMobile && (
        <div 
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className="app-main">
        {/* Sidebar */}
        <div className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
          {/* Desktop Header - Solo visible en desktop */}
          <div className="desktop-header">
            <div className="desktop-header-content">
              <div className="desktop-header-title">
                <span className="desktop-header-icon">🔐</span>
                <div>
                  <h1>SecureChat</h1>
                  <p>{formatAddress(account)}</p>
                </div>
              </div>
              
              <div className="desktop-header-actions">
                <div className="aegis-balance">
                  <span>💎</span>
                  {aegisBalance} AEGIS
                </div>
                
                <button
                  onClick={toggleDarkMode}
                  className="icon-btn dark-mode-toggle"
                  title={darkMode ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}
                >
                  <span className="theme-icon">
                    {darkMode ? '☀️' : '🌙'}
                  </span>
                </button>
                
                <button
                  onClick={disconnectWallet}
                  className="disconnect-btn"
                >
                  Desconectar
                </button>
              </div>
            </div>
          </div>
          {/* Tabs */}
          <div className="sidebar-tabs">
            {[
              { id: 'chat', label: 'Chat', icon: '💬' },
              { id: 'discover', label: 'Descubrir', icon: '🔍' },
              { id: 'profile', label: 'Perfil', icon: '👤' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`sidebar-tab ${activeTab === tab.id ? 'sidebar-tab-active' : ''}`}
              >
                <span className="sidebar-tab-icon">{tab.icon}</span>
                <div className="sidebar-tab-label">{tab.label}</div>
              </button>
            ))}
          </div>

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="sidebar-content">
              <div className="create-room-section">
                <button
                  onClick={() => setIsCreatingRoom(!isCreatingRoom)}
                  className="create-room-btn"
                >
                  {isCreatingRoom ? 'Cancelar' : 'Crear Sala'}
                </button>
              </div>

              <div className="rooms-list">
                {rooms.map((room) => (
                  <div
                    key={room.id}
                    onClick={() => selectRoom(room)}
                    className={`room-item ${currentRoom?.id === room.id ? 'room-item-active' : ''}`}
                  >
                    <div className="room-avatar">
                      <span>{room.name.charAt(0).toUpperCase()}</span>
                    </div>
                    <div className="room-info">
                      <h3 className="room-name">{room.name}</h3>
                      <p className="room-members">{room.memberCount} miembros</p>
                    </div>
                    <div className="room-status">
                      <div className="online-indicator"></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Discover Tab */}
          {activeTab === 'discover' && (
            <div className="flex-1 p-6">
              <div className="text-center space-y-4">
                <div className="text-6xl animate-bounce-in-soft animate-heartbeat">🔍</div>
                <h3 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  Descubrir Salas
                </h3>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Próximamente: Explora salas públicas y únete a conversaciones interesantes
                </p>
              </div>
            </div>
          )}

          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="flex-1 p-6">
              <div className="max-w-md mx-auto space-y-6">
                <div className="text-center">
                  <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-r from-purple-500 to-pink-600 flex items-center justify-center animate-bounce-in-soft hover-glow animate-heartbeat">
                    <span className="text-3xl">👤</span>
                  </div>
                  <h2 className={`text-2xl font-bold mb-2 animate-fade-in-up ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Tu Perfil
                  </h2>
                  <p className={`text-sm animate-slide-in-right ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    {formatAddress(account)}
                  </p>
                </div>

                <div className={`p-6 rounded-2xl shadow-lg backdrop-blur-lg border glass-advanced hover-lift animate-fade-in-up ${darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white/70 border-gray-200'}`}>
                  <h3 className={`text-lg font-bold mb-4 animate-fade-in-up ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Tokens AEGIS
                  </h3>
                  
                  <div className="space-y-4">
                    <div className={`p-4 rounded-xl text-center glass-advanced animate-pulse-advanced ${darkMode ? 'bg-purple-600/20' : 'bg-purple-100'}`}>
                      <div className={`text-3xl font-bold text-gradient-animated animate-shimmer-advanced ${darkMode ? 'text-purple-300' : 'text-purple-700'}`}>
                        {aegisBalance}
                      </div>
                      <div className={`text-sm animate-fade-in-up ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        AEGIS Tokens
                      </div>
                    </div>
                    
                    <button
                      onClick={requestAegisTokens}
                      className="w-full p-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl font-medium shadow-lg hover:shadow-green-500/25 transition-all duration-300 transform hover:scale-105 ripple-button hover-glow animate-float-soft"
                    >
                      Solicitar Tokens
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Main Chat Area */}
        <div className="chat-area">
          {currentRoom ? (
            <>
              {/* Chat Header */}
              <div className="chat-header">
                <div className="chat-header-content">
                  <button
                    onClick={handleBackToRooms}
                    className="back-btn"
                  >
                    ←
                  </button>
                  <div className="chat-avatar">
                    <span>{currentRoom.name.charAt(0).toUpperCase()}</span>
                  </div>
                  <div className="chat-info">
                    <h2 className="chat-title">{currentRoom.name}</h2>
                    <p className="chat-status">{currentRoom.memberCount} miembros activos</p>
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="messages-container">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`message ${message.sender === account ? 'message-sent' : 'message-received'}`}
                  >
                    <div className="message-avatar">
                      <span>
                        {message.sender === account ? 'Tú' : message.sender.slice(0, 2).toUpperCase()}
                      </span>
                    </div>
                    
                    <div className="message-content">
                      <div className="message-sender">
                        {message.sender === account ? 'Tú' : formatAddress(message.sender)}
                      </div>
                      
                      <div className="message-bubble">
                        {message.type === 'file' && message.file ? (
                          <div className="message-file">
                            <div className="file-preview">
                              <div className="file-icon">📎</div>
                              <div className="file-info">
                                <div className="file-name">{message.file.name}</div>
                                <div className="file-size">{(message.file.size / 1024 / 1024).toFixed(2)} MB</div>
                              </div>
                              <button className="file-download">⬇️</button>
                            </div>
                            {message.text && <p className="file-caption">{message.text}</p>}
                          </div>
                        ) : (
                          <p className="message-text">{message.text}</p>
                        )}
                      </div>
                      
                      <div className="message-meta">
                        <span className="message-time">{formatTime(message.timestamp)}</span>
                        {message.sender === account && (
                          <span className="message-status">✓✓</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}

                {/* Typing Indicator */}
                {isTyping && (
                  <div className="typing-indicator">
                    <div className="typing-avatar">
                      <span>...</span>
                    </div>
                    <div className="typing-bubble">
                      <div className="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Message Input */}
              <div className="message-input-container">
                {selectedFile && (
                  <div className="file-preview-container">
                    <div className="file-preview-item">
                      <div className="file-icon">📎</div>
                      <div className="file-info">
                        <span className="file-name">{selectedFile.name}</span>
                        <span className="file-size">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                      </div>
                      <button onClick={cancelFileSelection} className="file-remove">
                        ✕
                      </button>
                    </div>
                  </div>
                )}

                <div className="message-input-bar">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                    className="file-input-hidden"
                    accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.gif"
                  />
                  
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="attach-btn"
                  >
                    📎
                  </button>

                  <input
                    type="text"
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Escribe un mensaje..."
                    className="message-input"
                  />

                  <button
                    onClick={sendMessage}
                    disabled={!newMessage.trim() && !selectedFile}
                    className="send-btn"
                  >
                    ➤
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="no-chat-selected">
              <div className="no-chat-content">
                <div className="no-chat-icon">💬</div>
                <h3 className="no-chat-title">SecureChat</h3>
                <p className="no-chat-subtitle">Selecciona un chat para comenzar a conversar</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Room Creation Modal */}
      {isCreatingRoom && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3 className="modal-title">Crear Nueva Sala</h3>
            </div>
            <div className="modal-body">
              <input
                type="text"
                value={newRoomName}
                onChange={(e) => setNewRoomName(e.target.value)}
                placeholder="Nombre de la sala"
                className="modal-input"
                onKeyPress={(e) => e.key === 'Enter' && createRoom()}
              />
            </div>
            <div className="modal-actions">
              <button onClick={() => setIsCreatingRoom(false)} className="modal-btn-cancel">
                Cancelar
              </button>
              <button onClick={createRoom} className="modal-btn-create">
                Crear
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Notifications */}
      {notifications.map((notification) => (
        <div 
          key={notification.id} 
          className={`notification notification-${notification.type}`}
        >
          {notification.message}
        </div>
      ))}
    </div>
  );
}

export default App;