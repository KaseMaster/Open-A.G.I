const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

/**
 * Script de despliegue para AEGISToken y AEGISFaucet
 * Despliega ambos contratos y configura el faucet con tokens iniciales
 */
async function main() {
    console.log("🚀 Iniciando despliegue de contratos AEGIS...\n");
    
    // Obtener signers
    const [deployer] = await ethers.getSigners();
    console.log("📝 Desplegando contratos con la cuenta:", deployer.address);
    
    // Verificar balance del deployer
    const balance = await ethers.provider.getBalance(deployer.address);
    console.log("💰 Balance de la cuenta:", ethers.formatEther(balance), "ETH\n");
    
    if (balance < ethers.parseEther("0.01")) {
        console.warn("⚠️  ADVERTENCIA: Balance bajo, podría no ser suficiente para el despliegue\n");
    }
    
    try {
        // 1. Desplegar AEGISToken
        console.log("🪙 Desplegando AEGISToken...");
        const AEGISToken = await ethers.getContractFactory("AEGISToken");
        const aegisToken = await AEGISToken.deploy(deployer.address);
        await aegisToken.waitForDeployment();
        
        const tokenAddress = await aegisToken.getAddress();
        console.log("✅ AEGISToken desplegado en:", tokenAddress);
        
        // Verificar información del token
        const tokenInfo = await aegisToken.getTokenInfo();
        console.log("📊 Información del token:");
        console.log("   - Nombre:", tokenInfo[0]);
        console.log("   - Símbolo:", tokenInfo[1]);
        console.log("   - Decimales:", tokenInfo[2].toString());
        console.log("   - Suministro inicial:", ethers.formatEther(tokenInfo[3]), "AEGIS");
        console.log("   - Suministro máximo:", ethers.formatEther(tokenInfo[4]), "AEGIS\n");
        
        // 2. Desplegar AEGISFaucet
        console.log("🚰 Desplegando AEGISFaucet...");
        const AEGISFaucet = await ethers.getContractFactory("AEGISFaucet");
        const aegisFaucet = await AEGISFaucet.deploy(tokenAddress, deployer.address);
        await aegisFaucet.waitForDeployment();
        
        const faucetAddress = await aegisFaucet.getAddress();
        console.log("✅ AEGISFaucet desplegado en:", faucetAddress);
        
        // 3. Configurar el faucet con tokens iniciales
        console.log("\n🔧 Configurando faucet...");
        
        // Transferir tokens al faucet (1 millón de tokens para distribución)
        const faucetSupply = ethers.parseEther("1000000"); // 1M AEGIS
        console.log("💸 Transfiriendo", ethers.formatEther(faucetSupply), "AEGIS al faucet...");
        
        const transferTx = await aegisToken.transfer(faucetAddress, faucetSupply);
        await transferTx.wait();
        console.log("✅ Tokens transferidos al faucet");
        
        // Verificar balance del faucet
        const faucetBalance = await aegisToken.balanceOf(faucetAddress);
        console.log("💰 Balance del faucet:", ethers.formatEther(faucetBalance), "AEGIS");
        
        // Obtener estadísticas del faucet
        const faucetStats = await aegisFaucet.getFaucetStats();
        console.log("📊 Configuración del faucet:");
        console.log("   - Cantidad por solicitud:", ethers.formatEther(await aegisFaucet.faucetAmount()), "AEGIS");
        console.log("   - Tiempo de cooldown:", (await aegisFaucet.cooldownTime()).toString() / 3600, "horas");
        console.log("   - Límite diario:", ethers.formatEther(await aegisFaucet.maxDailyLimit()), "AEGIS");
        console.log("   - Estado:", faucetStats[6] ? "Pausado" : "Activo");
        
        // 4. Guardar direcciones de contratos
        const deploymentInfo = {
            network: await ethers.provider.getNetwork().then(n => n.name),
            chainId: await ethers.provider.getNetwork().then(n => n.chainId.toString()),
            deployer: deployer.address,
            deploymentTime: new Date().toISOString(),
            contracts: {
                AEGISToken: {
                    address: tokenAddress,
                    name: tokenInfo[0],
                    symbol: tokenInfo[1],
                    decimals: tokenInfo[2].toString(),
                    initialSupply: ethers.formatEther(tokenInfo[3]),
                    maxSupply: ethers.formatEther(tokenInfo[4])
                },
                AEGISFaucet: {
                    address: faucetAddress,
                    tokenAddress: tokenAddress,
                    faucetAmount: ethers.formatEther(await aegisFaucet.faucetAmount()),
                    cooldownHours: (await aegisFaucet.cooldownTime()).toString() / 3600,
                    dailyLimit: ethers.formatEther(await aegisFaucet.maxDailyLimit()),
                    initialBalance: ethers.formatEther(faucetBalance)
                }
            },
            gasUsed: {
                token: "Estimado en despliegue",
                faucet: "Estimado en despliegue",
                setup: "Estimado en configuración"
            }
        };
        
        // Crear directorio deployments si no existe
        const deploymentsDir = path.join(__dirname, "..", "deployments");
        if (!fs.existsSync(deploymentsDir)) {
            fs.mkdirSync(deploymentsDir, { recursive: true });
        }
        
        // Guardar información de despliegue
        const deploymentFile = path.join(deploymentsDir, `deployment-${Date.now()}.json`);
        fs.writeFileSync(deploymentFile, JSON.stringify(deploymentInfo, null, 2));
        
        // También guardar la última versión
        const latestFile = path.join(deploymentsDir, "latest.json");
        fs.writeFileSync(latestFile, JSON.stringify(deploymentInfo, null, 2));
        
        console.log("\n📄 Información de despliegue guardada en:");
        console.log("   -", deploymentFile);
        console.log("   -", latestFile);
        
        // 5. Mostrar resumen final
        console.log("\n🎉 ¡Despliegue completado exitosamente!");
        console.log("=" .repeat(50));
        console.log("📋 RESUMEN DE DESPLIEGUE:");
        console.log("=" .repeat(50));
        console.log("🪙 AEGISToken:", tokenAddress);
        console.log("🚰 AEGISFaucet:", faucetAddress);
        console.log("🌐 Red:", deploymentInfo.network);
        console.log("⛓️  Chain ID:", deploymentInfo.chainId);
        console.log("👤 Deployer:", deployer.address);
        console.log("=" .repeat(50));
        
        // 6. Instrucciones para verificación en Etherscan (si aplica)
        const network = await ethers.provider.getNetwork();
        if (network.chainId === 11155111n) { // Sepolia
            console.log("\n🔍 Para verificar en Etherscan:");
            console.log("npx hardhat verify --network sepolia", tokenAddress, deployer.address);
            console.log("npx hardhat verify --network sepolia", faucetAddress, tokenAddress, deployer.address);
        }
        
        // 7. Instrucciones de uso
        console.log("\n📖 INSTRUCCIONES DE USO:");
        console.log("=" .repeat(30));
        console.log("1. Los usuarios pueden solicitar tokens del faucet llamando a requestTokens()");
        console.log("2. Cada usuario puede solicitar", ethers.formatEther(await aegisFaucet.faucetAmount()), "AEGIS cada", (await aegisFaucet.cooldownTime()).toString() / 3600, "horas");
        console.log("3. El faucet tiene un límite diario de", ethers.formatEther(await aegisFaucet.maxDailyLimit()), "AEGIS");
        console.log("4. El owner puede pausar/despausar el faucet y ajustar parámetros");
        console.log("5. Los tokens son minteable, burnable y pausable por el owner");
        
        return {
            aegisToken: tokenAddress,
            aegisFaucet: faucetAddress,
            deploymentInfo
        };
        
    } catch (error) {
        console.error("❌ Error durante el despliegue:", error);
        throw error;
    }
}

// Función para despliegue en red específica
async function deployToNetwork(networkName) {
    console.log(`🌐 Desplegando en red: ${networkName}`);
    return await main();
}

// Ejecutar si es llamado directamente
if (require.main === module) {
    main()
        .then(() => process.exit(0))
        .catch((error) => {
            console.error("💥 Error fatal:", error);
            process.exit(1);
        });
}

module.exports = {
    main,
    deployToNetwork
};