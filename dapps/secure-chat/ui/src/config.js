export const CONTRACTS = {
  UserRegistry: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512", // Dirección despliegue local
  ChatRoom: "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"     // Dirección despliegue local
};

// Configuración de IPFS para subir contenidos cifrados
// Ejemplos:
// - Local node: API_URL: "http://127.0.0.1:5001/api/v0"
// - Infura: API_URL: "https://ipfs.infura.io:5001/api/v0", BASIC_AUTH: "Basic base64(projectId:projectSecret)"
export const IPFS = {
  API_URL: "http://127.0.0.1:5001/api/v0",
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
export const IPFS_GATEWAY = "http://127.0.0.1:8080/ipfs/";