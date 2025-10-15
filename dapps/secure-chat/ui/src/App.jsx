import React, { useState, useEffect, useRef } from 'react';
import { ethers } from 'ethers';
import { create } from 'ipfs-http-client';
import nacl from 'tweetnacl';
import './App.css';
import StatusBar from './components/StatusBar';

// Base58 implementation
const BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';

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
  
  return '1'.repeat(leadingZeros) + digits.reverse().map(d => BASE58_ALPHABET[d]).join('');
}

function base58Decode(str) {
  if (str.length === 0) return new Uint8Array();
  
  let bytes = [0];
  for (let i = 0; i < str.length; i++) {
    const char = str[i];
    const charIndex = BASE58_ALPHABET.indexOf(char);
    if (charIndex === -1) throw new Error('Invalid Base58 character');
    
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
  
  let leadingZeros = 0;
  for (let i = 0; i < str.length && str[i] === '1'; i++) {
    leadingZeros++;
  }
  
  return new Uint8Array(leadingZeros + bytes.reverse().length).fill(0, 0, leadingZeros).set(bytes, leadingZeros);
}

// Contract addresses and ABIs
const CHAT_CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3";
const AEGIS_TOKEN_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";

const CHAT_ABI = [
  "function createRoom(string memory name, string memory ipfsHash) public",
  "function joinRoom(uint256 roomId) public",
  "function sendMessage(uint256 roomId, string memory ipfsHash) public",
  "function getRooms() public view returns (tuple(uint256 id, string name, string ipfsHash, address creator, uint256 memberCount)[])",
  "function getRoomMessages(uint256 roomId) public view returns (tuple(address sender, string ipfsHash, uint256 timestamp)[])",
  "function getRoomMembers(uint256 roomId) public view returns (address[])",
  "event RoomCreated(uint256 indexed roomId, string name, address creator)",
  "event MessageSent(uint256 indexed roomId, address sender, string ipfsHash)",
  "event MemberJoined(uint256 indexed roomId, address member)"
];

const AEGIS_ABI = [
  "function balanceOf(address owner) view returns (uint256)",
  "function transfer(address to, uint256 amount) returns (bool)",
  "function mint(address to, uint256 amount) public",
  "function requestTokens() public",
  "function donate(address to, uint256 amount) public",
  "event Transfer(address indexed from, address indexed to, uint256 value)"
];

// IPFS client setup (use Vite proxy to avoid CORS in dev)
const ipfsBaseUrl = (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:5173') + '/ipfs-api';
const ipfsClient = create({ url: ipfsBaseUrl });

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
    const result = await ipfsClient.add(JSON.stringify(data));
    return result.path;
  } catch (error) {
    console.error('IPFS upload error:', error);
    throw error;
  }
}

async function fetchFromIPFS(hash) {
  try {
    let data = '';
    for await (const chunk of ipfsClient.cat(hash)) {
      data += new TextDecoder().decode(chunk);
    }
    return JSON.parse(data);
  } catch (error) {
    console.error('IPFS fetch error:', error);
    return null;
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
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState('chat');
  const [notifications, setNotifications] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [parallaxOffset, setParallaxOffset] = useState(0);
  const [ipfsOk, setIpfsOk] = useState(false);
  const [chainOk, setChainOk] = useState(false);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

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
    try {
      if (typeof window.ethereum !== 'undefined') {
        await window.ethereum.request({ method: 'eth_requestAccounts' });
        const provider = new ethers.BrowserProvider(window.ethereum);
        const signer = await provider.getSigner();
        const address = await signer.getAddress();
        
        setProvider(provider);
        setSigner(signer);
        setAccount(address);
        
        const chatContract = new ethers.Contract(CHAT_CONTRACT_ADDRESS, CHAT_ABI, signer);
        const aegisContract = new ethers.Contract(AEGIS_TOKEN_ADDRESS, AEGIS_ABI, signer);
        
        setChatContract(chatContract);
        setAegisContract(aegisContract);
        
        showNotification('Wallet conectado exitosamente', 'success');
      } else {
        showNotification('MetaMask no encontrado', 'error');
      }
    } catch (error) {
      console.error('Error connecting wallet:', error);
      showNotification('Error conectando wallet', 'error');
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
    if (!chatContract) return;
    
    try {
      const contractRooms = await chatContract.getRooms();
      const formattedRooms = contractRooms.map(room => ({
        id: Number(room.id),
        name: room.name,
        memberCount: Number(room.memberCount),
        creator: room.creator,
        isDefault: false
      }));
      
      setRooms([...defaultRooms, ...formattedRooms]);
    } catch (error) {
      console.error('Error loading rooms:', error);
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
    if (!aegisContract || !account) return;
    
    try {
      const balance = await aegisContract.balanceOf(account);
      setAegisBalance(ethers.formatEther(balance));
    } catch (error) {
      console.error('Error loading AEGIS balance:', error);
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
    if (account) {
      loadRooms();
      loadAegisBalance();
    }
  }, [account, chatContract, aegisContract]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check IPFS API availability
  useEffect(() => {
    (async () => {
      try {
        await ipfsClient.version();
        setIpfsOk(true);
      } catch (e) {
        setIpfsOk(false);
      }
    })();
  }, []);

  // Check chain/provider availability
  useEffect(() => {
    (async () => {
      try {
        if (provider) {
          const net = await provider.getNetwork();
          setChainOk(!!net);
        } else {
          setChainOk(false);
        }
      } catch (e) {
        setChainOk(false);
      }
    })();
  }, [provider]);

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
      <>
        <StatusBar ipfsOk={ipfsOk} chainOk={chainOk} account={account} aegisBalance={aegisBalance} />
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
      </>
    );
  }

  // Main app render
  return (
    <>
      <StatusBar ipfsOk={ipfsOk} chainOk={chainOk} account={account} aegisBalance={aegisBalance} />
      <div className={`min-h-screen transition-all duration-500 ${darkMode ? 'bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900' : 'bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50'}`}>
      {/* Header */}
      <div className="app-header">
        <div className="brand">
          <span>🔐</span>
          <div>
            <div className="brand-title">SecureChat</div>
            <div className="brand-sub">{formatAddress(account)}</div>
          </div>
        </div>

        <div className="flex items-center gap-12">
          <div className="badge">
            <span>💎</span> {aegisBalance} AEGIS
          </div>

          <button
            onClick={() => setDarkMode(!darkMode)}
            className="button"
          >
            {darkMode ? '☀️' : '🌙'}
          </button>

          <button
            onClick={disconnectWallet}
            className="button"
          >
            Desconectar
          </button>
        </div>
      </div>

      <div className="app-main">
        {/* Sidebar */}
        <div className="sidebar">
          {/* Tabs */}
          <div className="sidebar-header">
            {[
              { id: 'chat', label: 'Chat', icon: '💬' },
              { id: 'discover', label: 'Descubrir', icon: '🔍' },
              { id: 'profile', label: 'Perfil', icon: '👤' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`button ${
                  activeTab === tab.id
                    ? darkMode
                      ? 'bg-purple-600/20 text-purple-300 border-b-2 border-purple-500'
                      : 'bg-purple-100 text-purple-700 border-b-2 border-purple-500'
                    : darkMode
                    ? 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <span className="text-lg animate-bounce-in-soft">{tab.icon}</span>
                <div className="text-sm mt-1">{tab.label}</div>
              </button>
            ))}
          </div>

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="flex-1 overflow-hidden">
              <div className="sidebar-header">
                <button
                  onClick={() => setIsCreatingRoom(!isCreatingRoom)}
                  className="button button-primary"
                >
                  {isCreatingRoom ? 'Cancelar' : 'Crear Sala'}
                </button>
              </div>

              <div className="rooms">
                {rooms.map((room) => (
                  <div
                    key={room.id}
                    onClick={() => joinRoom(room)}
                    className="room-item"
                  >
                    <div>
                      <div className="room-name">{room.name}</div>
                      <div className="room-meta">{room.memberCount} miembros</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Discover Tab */}
          {activeTab === 'discover' && (
            <div className="flex-1 p-6">
              <div className="text-center">
                <div className="chat-title">Descubrir Salas</div>
                <div className="chat-sub">Próximamente: Explora salas públicas y únete</div>
              </div>
            </div>
          )}

          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="flex-1 p-6">
              <div className="text-center">
                <div className="chat-title">Tu Perfil</div>
                <div className="chat-sub">{formatAddress(account)}</div>
              </div>
              <div className="rooms" style={{ maxWidth: 420, margin: '0 auto' }}>
                <div className="message" style={{ width: '100%' }}>
                  <div className="room-name">Tokens AEGIS</div>
                  <div className="room-meta">Balance: {aegisBalance}</div>
                  <div style={{ marginTop: 12 }}>
                    <button onClick={requestAegisTokens} className="button button-primary" style={{ width: '100%' }}>
                      Solicitar Tokens
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Main Chat Area */}
        <div className="chat">
          {currentRoom ? (
            <>
              {/* Chat Header */}
              <div className="chat-header">
                <button onClick={() => setCurrentRoom(null)} className="button">←</button>
                <span className="chat-title">{currentRoom.name}</span>
                <span className="chat-sub">{currentRoom.memberCount} miembros activos</span>
              </div>

              {/* Messages */}
              <div className="messages" style={{ transform: `translateY(${parallaxOffset}px)` }}>
                {messages.map((message, index) => (
                  <div key={index} className={`message ${message.sender === account ? 'message--self' : ''}`}>
                    <div>{message.sender === account ? 'Tú' : formatAddress(message.sender)}</div>
                    <div>
                      {message.type === 'file' && message.file ? (
                        <div>
                          <div>
                            {message.file.name} ({(message.file.size / 1024 / 1024).toFixed(2)} MB)
                          </div>
                          {message.text && (<p>{message.text}</p>)}
                        </div>
                      ) : (
                        <p>{message.text}</p>
                      )}
                    </div>
                    <div className="message-meta">
                      <span>{formatTime(message.timestamp)}</span>
                      {message.sender === account && (<span>✓✓</span>)}
                    </div>
                  </div>
                ))}

                {/* Typing Indicator */}
                {isTyping && (
                  <div className="flex space-x-3 animate-fade-in animate-slide-in-left">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-gray-400 to-gray-600 flex items-center justify-center shadow-lg animate-bounce-in-soft hover-glow animate-heartbeat">
                      <span className="text-white text-xs">...</span>
                    </div>
                    <div className={`p-3 rounded-2xl shadow-lg backdrop-blur-lg border glass-advanced animate-pulse-advanced ${darkMode ? 'bg-gray-700/50 border-gray-600' : 'bg-white/70 border-gray-200'}`}>
                      <div className="flex space-x-1 animate-bounce">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce animate-heartbeat" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce animate-heartbeat" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce animate-heartbeat" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Message Input */}
              <div className="input-bar">
                {selectedFile && (
                  <div className="message">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center animate-bounce">
                          <span className="text-white text-sm">📎</span>
                        </div>
                        <div>
                          <div className="text-sm">
                            {selectedFile.name}
                          </div>
                          <div className="text-xs">
                            {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                          </div>
                        </div>
                      </div>
                      <div className="flex space-x-2">
                        <button
                          onClick={sendMessage}
                          className="button button-primary"
                        >
                          Compartir
                        </button>
                        <button
                          onClick={cancelFileSelection}
                          className="button"
                        >
                          Cancelar
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-[auto,1fr,auto] gap-2">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                    className="hidden"
                    accept="*/*"
                  />
                  
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="button"
                  >
                    📎
                  </button>
                  
                  <input
                    type="text"
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Escribe tu mensaje..."
                    className="input"
                  />
                  
                  <button
                    onClick={sendMessage}
                    disabled={!newMessage.trim() && !selectedFile}
                    className="button button-primary"
                  >
                    🚀
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className="chat-title">Selecciona una sala</div>
                <div className="chat-sub">Elige una sala para comenzar a conversar</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Room Creation Modal */}
      {isCreatingRoom && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
          <div className={`p-6 rounded-2xl shadow-2xl backdrop-blur-lg border max-w-md w-full mx-4 glass-advanced animate-scale-in ${darkMode ? 'bg-gray-800/90 border-gray-700' : 'bg-white/90 border-gray-200'}`}>
            <div className="chat-title">Crear Nueva Sala</div>
            <div style={{ marginTop: 12 }}>
              <input
                type="text"
                value={newRoomName}
                onChange={(e) => setNewRoomName(e.target.value)}
                placeholder="Nombre de la sala..."
                className="input"
              />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 12 }}>
              <button
                onClick={createRoom}
                disabled={!newRoomName.trim()}
                className="button button-primary"
              >
                Crear
              </button>
              <button
                onClick={() => setIsCreatingRoom(false)}
                className="button"
              >
                Cancelar
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Notifications */}
      <div className="fixed top-4 right-4 space-y-2 z-50">
        {notifications.map((notification) => (
          <div
            key={notification.id}
            className={`p-4 rounded-xl shadow-lg backdrop-blur-lg border animate-slide-in-right glass-advanced ${
              notification.type === 'success'
                ? 'bg-green-500/20 border-green-500/50 text-green-300'
                : notification.type === 'error'
                ? 'bg-red-500/20 border-red-500/50 text-red-300'
                : 'bg-blue-500/20 border-blue-500/50 text-blue-300'
            }`}
          >
            {notification.message}
          </div>
        ))}
      </div>
      </div>
    </>
  );
}

export default App;