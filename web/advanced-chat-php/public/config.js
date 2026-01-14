/**
 * Configuración de dominios para OpenAGI Secure Chat+
 * Dominio principal integrado: xn--conexinsecreta-qob.site y aegis-openagi.eth
 */

window.OPENAGI_CONFIG = {
    // Dominios principales
    PRIMARY_DOMAINS: [
        'xn--conexinsecreta-qob.site',
        'www.xn--conexinsecreta-qob.site',
        'aegis-openagi.eth',
        'www.aegis-openagi.eth'
    ],
    
    // URLs de API según el entorno
    API_ENDPOINTS: {
        // Producción - usar dominios principales
        PRODUCTION: {
            OPENAGI_API: window.location.protocol + '//' + window.location.host + '/api',
            CHAT_API: window.location.protocol + '//' + window.location.host + '/api.php',
            WEBSOCKET: (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host + '/ws',
            IPFS_GATEWAY: window.location.protocol + '//' + window.location.host + '/ipfs'
        },
        
        // Desarrollo local
        DEVELOPMENT: {
            OPENAGI_API: 'http://localhost:8182',
            CHAT_API: '/api.php',
            WEBSOCKET: 'ws://localhost:8182/ws',
            IPFS_GATEWAY: 'http://localhost:8182/ipfs'
        }
    },
    
    // Detectar entorno automáticamente
    getEnvironment() {
        const hostname = window.location.hostname;
        
        // Si es uno de los dominios principales, usar producción
        if (this.PRIMARY_DOMAINS.includes(hostname)) {
            return 'PRODUCTION';
        }
        
        // Si es localhost o IP local, usar desarrollo
        if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname.match(/^\d+\.\d+\.\d+\.\d+$/)) {
            return 'DEVELOPMENT';
        }
        
        // Por defecto, usar producción
        return 'PRODUCTION';
    },
    
    // Obtener endpoints para el entorno actual
    getEndpoints() {
        const env = this.getEnvironment();
        return this.API_ENDPOINTS[env];
    },
    
    // Verificar si estamos en un dominio principal
    isPrimaryDomain() {
        return this.PRIMARY_DOMAINS.includes(window.location.hostname);
    }
};

// Configuración global disponible
window.ENDPOINTS = window.OPENAGI_CONFIG.getEndpoints();