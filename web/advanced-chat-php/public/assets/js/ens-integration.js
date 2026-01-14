/**
 * Integración ENS para aegis-openagi.eth
 * Manejo de resolución de dominios descentralizados
 */

class ENSIntegration {
  constructor() {
    this.provider = null;
    this.ensResolver = null;
    this.domain = 'aegis-openagi.eth';
    this.initialized = false;
  }

  /**
   * Inicializar la conexión con ENS
   */
  async initialize() {
    try {
      // Verificar si MetaMask está disponible
      if (typeof window.ethereum !== 'undefined') {
        this.provider = new ethers.providers.Web3Provider(window.ethereum);
        this.ensResolver = this.provider.getResolver(this.domain);
        this.initialized = true;
        
        console.log('✅ ENS Integration inicializada para:', this.domain);
        return true;
      } else {
        console.warn('⚠️ MetaMask no detectado, usando resolución fallback');
        return false;
      }
    } catch (error) {
      console.error('❌ Error inicializando ENS:', error);
      return false;
    }
  }

  /**
   * Resolver dirección del dominio ENS
   */
  async resolveAddress() {
    if (!this.initialized) {
      const success = await this.initialize();
      if (!success) {
        console.warn('⚠️ ENS no disponible, usando fallback');
        return null;
      }
    }

    if (!this.provider) {
      console.warn('⚠️ Provider no disponible para resolución ENS');
      return null;
    }

    try {
      const address = await this.provider.resolveName(this.domain);
      console.log(`📍 ${this.domain} resuelve a:`, address);
      return address;
    } catch (error) {
      console.error('❌ Error resolviendo ENS:', error);
      return null;
    }
  }

  /**
   * Obtener contenido hash del dominio ENS
   */
  async getContentHash() {
    if (!this.ensResolver) {
      console.warn('⚠️ ENS Resolver no disponible');
      return null;
    }

    try {
      const contentHash = await this.ensResolver.getContentHash();
      console.log(`📦 Content Hash para ${this.domain}:`, contentHash);
      return contentHash;
    } catch (error) {
      console.error('❌ Error obteniendo content hash:', error);
      return null;
    }
  }

  /**
   * Obtener texto de un registro ENS
   */
  async getText(key) {
    if (!this.ensResolver) {
      console.warn('⚠️ ENS Resolver no disponible');
      return null;
    }

    try {
      const text = await this.ensResolver.getText(key);
      console.log(`📝 Texto '${key}' para ${this.domain}:`, text);
      return text;
    } catch (error) {
      console.error(`❌ Error obteniendo texto '${key}':`, error);
      return null;
    }
  }

  /**
   * Verificar si el dominio ENS está configurado correctamente
   */
  async verifyDomain() {
    const checks = {
      address: await this.resolveAddress(),
      contentHash: await this.getContentHash(),
      website: await this.getText('url'),
      description: await this.getText('description'),
      avatar: await this.getText('avatar')
    };

    console.log('🔍 Verificación de dominio ENS:', checks);
    return checks;
  }

  /**
   * Redirigir a través de gateway ENS
   */
  redirectToENS() {
    const ensGateways = [
      `https://${this.domain}.limo`,
      `https://${this.domain}.link`,
      `https://eth.limo/${this.domain}`
    ];

    // Intentar con el primer gateway disponible
    window.location.href = ensGateways[0];
  }

  /**
   * Obtener URL de gateway ENS
   */
  getENSGatewayURL() {
    return `https://${this.domain}.limo`;
  }

  /**
   * Configurar subdominios ENS
   */
  getSubdomainURL(subdomain) {
    return `https://${subdomain}.${this.domain}.limo`;
  }
}

// Instancia global de ENS
window.ensIntegration = new ENSIntegration();

// Auto-inicializar cuando se carga la página
document.addEventListener('DOMContentLoaded', async () => {
  await window.ensIntegration.initialize();
  
  // Verificar dominio si estamos en producción
  if (!window.isDevelopment()) {
    await window.ensIntegration.verifyDomain();
  }
});

// Función helper para acceso rápido a ENS
window.resolveENS = async (domain = 'aegis-openagi.eth') => {
  return await window.ensIntegration.resolveAddress();
};

console.log('🌐 Módulo ENS cargado para aegis-openagi.eth');