/**
 * Configuración de endpoints para OpenAGI
 * Dominio principal: xn--conexinsecreta-qob.site y aegis-openagi.eth
 */

window.ENDPOINTS = {
  // Dominio principal para la aplicación
  PRIMARY_DOMAIN: 'https://xn--conexinsecreta-qob.site',
  
  // Dominio ENS para Web3
  ENS_DOMAIN: 'aegis-openagi.eth',
  
  // API principal de OpenAGI
  OPENAGI_API: 'https://xn--conexinsecreta-qob.site:8182',
  
  // WebSocket para chat en tiempo real
  WEBSOCKET: 'wss://xn--conexinsecreta-qob.site:8183',
  
  // Gateway IPFS
  IPFS_GATEWAY: 'https://xn--conexinsecreta-qob.site:8184',
  
  // Endpoints de DApps
  DAPPS: {
    CHAT: 'https://xn--conexinsecreta-qob.site/chat',
    WALLET: 'https://xn--conexinsecreta-qob.site/wallet',
    DEFI: 'https://xn--conexinsecreta-qob.site/defi',
    NFT: 'https://xn--conexinsecreta-qob.site/nft',
    DAO: 'https://xn--conexinsecreta-qob.site/dao',
    MARKETPLACE: 'https://xn--conexinsecreta-qob.site/marketplace'
  },
  
  // Configuración de red blockchain
  BLOCKCHAIN: {
    CHAIN_ID: 1, // Ethereum Mainnet
    RPC_URL: 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID',
    EXPLORER: 'https://etherscan.io'
  },
  
  // Configuración de desarrollo
  DEV: {
    LOCAL_API: 'http://localhost:8182',
    LOCAL_WS: 'ws://localhost:8183',
    LOCAL_IPFS: 'http://localhost:8184'
  }
};

// Función para detectar si estamos en desarrollo
window.isDevelopment = () => {
  return window.location.hostname === 'localhost' || 
         window.location.hostname === '127.0.0.1' ||
         window.location.hostname.includes('192.168.');
};

// Configurar endpoints según el entorno
if (window.isDevelopment()) {
  window.ENDPOINTS.OPENAGI_API = window.ENDPOINTS.DEV.LOCAL_API;
  window.ENDPOINTS.WEBSOCKET = window.ENDPOINTS.DEV.LOCAL_WS;
  window.ENDPOINTS.IPFS_GATEWAY = window.ENDPOINTS.DEV.LOCAL_IPFS;
} else {
  // Para producción, usar el mismo puerto que el servidor PHP
  window.ENDPOINTS.WEBSOCKET = 'ws://' + window.location.host + '/ws';
}

// Configuración de dominios para diferentes DApps
window.DAPP_DOMAINS = {
  'chat': 'chat.xn--conexinsecreta-qob.site',
  'wallet': 'wallet.xn--conexinsecreta-qob.site', 
  'defi': 'defi.xn--conexinsecreta-qob.site',
  'nft': 'nft.xn--conexinsecreta-qob.site',
  'dao': 'dao.xn--conexinsecreta-qob.site',
  'marketplace': 'market.xn--conexinsecreta-qob.site'
};

// Función para obtener la URL de una DApp específica
window.getDAppURL = (dappName) => {
  const domain = window.DAPP_DOMAINS[dappName];
  return domain ? `https://${domain}` : `${window.ENDPOINTS.PRIMARY_DOMAIN}/${dappName}`;
};

// Configuración de ENS
window.ENS_CONFIG = {
  domain: 'aegis-openagi.eth',
  resolver: '0x4976fb03C32e5B8cfe2b6cCB31c09Ba78EBaBa41', // ENS Public Resolver
  subdomains: {
    'chat': 'chat.aegis-openagi.eth',
    'wallet': 'wallet.aegis-openagi.eth',
    'defi': 'defi.aegis-openagi.eth',
    'nft': 'nft.aegis-openagi.eth',
    'dao': 'dao.aegis-openagi.eth',
    'api': 'api.aegis-openagi.eth'
  }
};

console.log('🌐 Configuración de dominios cargada:', {
  primary: window.ENDPOINTS.PRIMARY_DOMAIN,
  ens: window.ENS_CONFIG.domain,
  development: window.isDevelopment()
});