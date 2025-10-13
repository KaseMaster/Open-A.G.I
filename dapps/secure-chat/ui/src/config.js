export const CONTRACTS = {
  UserRegistry: "0xa85233C63b9Ee964Add6F2cffe00Fd84eb32338f", // Dirección despliegue local actualizada
  ChatRoom: "0x4A679253410272dd5232B3Ff7cF5dbB88f295319"     // Dirección despliegue local actualizada
};

// Exportar direcciones individuales para compatibilidad
export const USERREGISTRY_ADDRESS = CONTRACTS.UserRegistry;
export const CHATROOM_ADDRESS = CONTRACTS.ChatRoom;

// Configuración de IPFS para subir contenidos cifrados
// Ejemplos:
// - Local node: API_URL: "http://127.0.0.1:5001/api/v0"
// - Infura: API_URL: "https://ipfs.infura.io:5001/api/v0", BASIC_AUTH: "Basic base64(projectId:projectSecret)"
export const IPFS = {
  API_URL: "", // Deshabilitado temporalmente - usar servicio público
  BASIC_AUTH: ""
};

// Configuración de donaciones
export const DONATIONS = {
  // Dirección para donaciones ETH (cambia para producción)
  ETH_ADDRESS: "0x0000000000000000000000000000000000000000",
  
  // Token AEGIS para donaciones ERC-20
  AEGIS_TOKEN: {
    address: "0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9",
    symbol: "AEGIS",
    decimals: 18,
    name: "AEGIS Token"
  },
  
  // Faucet AEGIS para obtener tokens de prueba
  AEGIS_FAUCET: {
    address: "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9",
    cooldownHours: 24,
    amountPerRequest: "100"
  }
};

// Dirección para donaciones (ETH) - DEPRECATED: usar DONATIONS.ETH_ADDRESS
export const DONATION_ADDRESS = "0x0000000000000000000000000000000000000000";

// Gateway para previsualizar/descargar por navegador
// Ejemplo local: http://127.0.0.1:8080/ipfs/
export const IPFS_GATEWAY = "https://ipfs.io/ipfs/";