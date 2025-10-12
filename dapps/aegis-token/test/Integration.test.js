import { expect } from "chai";
import hre from "hardhat";
import { loadFixture, time } from "@nomicfoundation/hardhat-network-helpers";

const { ethers } = hre;

describe("AEGISToken + AEGISFaucet Integration", function () {
    // Fixture completo para pruebas de integración
    async function deployIntegrationFixture() {
        const [owner, user1, user2, user3, user4, user5] = await ethers.getSigners();
        
        // Desplegar AEGISToken
        const AEGISToken = await ethers.getContractFactory("AEGISToken");
        const aegisToken = await AEGISToken.deploy(owner.address);
        
        // Desplegar AEGISFaucet
        const AEGISFaucet = await ethers.getContractFactory("AEGISFaucet");
        const aegisFaucet = await AEGISFaucet.deploy(
            await aegisToken.getAddress(),
            owner.address
        );
        
        // Configurar el faucet después del despliegue
        await aegisFaucet.configureFaucet(
            ethers.parseEther("100"), // 100 tokens por claim
            86400, // 24 horas de cooldown en segundos
            ethers.parseEther("10000") // límite diario
        );
        
        // Transferir tokens al faucet (10% del suministro inicial)
        const faucetSupply = ethers.parseEther("10000000"); // 10M tokens
        await aegisToken.transfer(await aegisFaucet.getAddress(), faucetSupply);
        
        return { 
            aegisToken, 
            aegisFaucet, 
            owner, 
            user1, 
            user2, 
            user3, 
            user4, 
            user5 
        };
    }
    
    describe("Complete Deployment Flow", function () {
        it("Should deploy and configure both contracts correctly", async function () {
            const { aegisToken, aegisFaucet, owner } = await loadFixture(deployIntegrationFixture);
            
            // Verificar AEGISToken
            const tokenInfo = await aegisToken.getTokenInfo();
            expect(tokenInfo[0]).to.equal("AEGIS Token");
            expect(tokenInfo[1]).to.equal("AEGIS");
            expect(tokenInfo[3]).to.equal(ethers.parseEther("100000000")); // 100M initial supply
            
            // Verificar AEGISFaucet
            const faucetConfig = await aegisFaucet.getFaucetConfig();
            expect(faucetConfig[0]).to.equal(await aegisToken.getAddress());
            expect(faucetConfig[1]).to.equal(ethers.parseEther("100"));
            
            // Verificar balance del faucet
            const faucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            expect(faucetBalance).to.equal(ethers.parseEther("10000000"));
            
            // Verificar balance del owner
            const ownerBalance = await aegisToken.balanceOf(owner.address);
            expect(ownerBalance).to.equal(ethers.parseEther("90000000")); // 100M - 10M
        });
        
        it("Should handle initial token distribution correctly", async function () {
            const { aegisToken, aegisFaucet, owner } = await loadFixture(deployIntegrationFixture);
            
            const totalSupply = await aegisToken.totalSupply();
            const ownerBalance = await aegisToken.balanceOf(owner.address);
            const faucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            
            // Verificar que la suma de balances = total supply
            expect(ownerBalance + faucetBalance).to.equal(totalSupply);
            expect(totalSupply).to.equal(ethers.parseEther("100000000"));
        });
    });
    
    describe("User Journey - First Time Users", function () {
        it("Should allow new users to claim tokens from faucet", async function () {
            const { aegisToken, aegisFaucet, user1 } = await loadFixture(deployIntegrationFixture);
            
            // Usuario nuevo debería poder reclamar inmediatamente
            expect(await aegisFaucet.canClaim(user1.address)).to.be.true;
            expect(await aegisToken.balanceOf(user1.address)).to.equal(0);
            
            // Reclamar tokens
            await expect(aegisFaucet.connect(user1).requestTokens())
                .to.emit(aegisFaucet, "TokensClaimed")
                .withArgs(user1.address, ethers.parseEther("100"));
            
            // Verificar balance actualizado
            expect(await aegisToken.balanceOf(user1.address)).to.equal(ethers.parseEther("100"));
            expect(await aegisFaucet.canClaim(user1.address)).to.be.false;
        });
        
        it("Should handle multiple new users claiming simultaneously", async function () {
            const { aegisToken, aegisFaucet, user1, user2, user3, user4, user5 } = await loadFixture(deployIntegrationFixture);
            
            const users = [user1, user2, user3, user4, user5];
            const claimAmount = ethers.parseEther("100");
            
            // Todos los usuarios reclaman
            for (const user of users) {
                await aegisFaucet.connect(user).requestTokens();
                expect(await aegisToken.balanceOf(user.address)).to.equal(claimAmount);
            }
            
            // Verificar balance total del faucet
            const expectedFaucetBalance = ethers.parseEther("10000000") - (claimAmount * BigInt(users.length));
            const actualFaucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            expect(actualFaucetBalance).to.equal(expectedFaucetBalance);
        });
    });
    
    describe("Token Economics and Supply Management", function () {
        it("Should maintain correct token economics after faucet operations", async function () {
            const { aegisToken, aegisFaucet, owner, user1, user2 } = await loadFixture(deployIntegrationFixture);
            
            const initialSupply = await aegisToken.totalSupply();
            
            // Usuarios reclaman del faucet
            await aegisFaucet.connect(user1).requestTokens();
            await aegisFaucet.connect(user2).requestTokens();
            
            // Owner mintea nuevos tokens
            await aegisToken.mint(owner.address, ethers.parseEther("1000000"));
            
            // Usuario quema algunos tokens
            await aegisToken.connect(user1).burn(ethers.parseEther("50"));
            
            // Verificar supply total
            const expectedSupply = initialSupply + ethers.parseEther("1000000") - ethers.parseEther("50");
            const actualSupply = await aegisToken.totalSupply();
            expect(actualSupply).to.equal(expectedSupply);
            
            // Verificar que los tokens no se perdieron, solo se redistribuyeron
            const ownerBalance = await aegisToken.balanceOf(owner.address);
            const user1Balance = await aegisToken.balanceOf(user1.address);
            const user2Balance = await aegisToken.balanceOf(user2.address);
            const faucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            
            const totalAccountedTokens = ownerBalance + user1Balance + user2Balance + faucetBalance;
            expect(totalAccountedTokens).to.equal(actualSupply);
        });
        
        it("Should handle faucet running out of tokens gracefully", async function () {
            const { aegisToken, aegisFaucet, owner, user1 } = await loadFixture(deployIntegrationFixture);
            
            // Retirar casi todos los tokens del faucet
            const faucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            const withdrawAmount = faucetBalance - ethers.parseEther("50"); // Dejar solo 50 tokens
            
            await aegisFaucet.emergencyWithdraw(withdrawAmount);
            
            // Usuario intenta reclamar 100 tokens pero solo hay 50
            await expect(aegisFaucet.connect(user1).requestTokens())
                .to.be.rejectedWith("AEGISFaucet: insufficient faucet balance");
            
            // Recargar faucet
            await aegisToken.transfer(await aegisFaucet.getAddress(), ethers.parseEther("1000"));
            
            // Ahora debería funcionar
            await expect(aegisFaucet.connect(user1).requestTokens())
                .to.emit(aegisFaucet, "TokensClaimed");
        });
    });
    
    describe("Advanced User Scenarios", function () {
        it("Should handle users with existing tokens claiming from faucet", async function () {
            const { aegisToken, aegisFaucet, owner, user1 } = await loadFixture(deployIntegrationFixture);
            
            // Owner transfiere tokens directamente al usuario
            await aegisToken.transfer(user1.address, ethers.parseEther("500"));
            expect(await aegisToken.balanceOf(user1.address)).to.equal(ethers.parseEther("500"));
            
            // Usuario también puede reclamar del faucet
            await aegisFaucet.connect(user1).requestTokens();
            expect(await aegisToken.balanceOf(user1.address)).to.equal(ethers.parseEther("600"));
        });
        
        it("Should handle complex token operations after faucet claims", async function () {
            const { aegisToken, aegisFaucet, owner, user1, user2 } = await loadFixture(deployIntegrationFixture);
            
            // Usuarios reclaman del faucet
            await aegisFaucet.connect(user1).requestTokens();
            await aegisFaucet.connect(user2).requestTokens();
            
            // Operaciones complejas de tokens
            // 1. Transferencia entre usuarios
            await aegisToken.connect(user1).transfer(user2.address, ethers.parseEther("25"));
            
            // 2. Approval y transferFrom
            await aegisToken.connect(user2).approve(user1.address, ethers.parseEther("50"));
            await aegisToken.connect(user1).transferFrom(user2.address, owner.address, ethers.parseEther("30"));
            
            // 3. Burn tokens
            await aegisToken.connect(user1).burn(ethers.parseEther("10"));
            
            // Verificar balances finales
            expect(await aegisToken.balanceOf(user1.address)).to.equal(ethers.parseEther("65")); // 100 - 25 + 30 - 10
            expect(await aegisToken.balanceOf(user2.address)).to.equal(ethers.parseEther("95")); // 100 + 25 - 30
        });
    });
    
    describe("Administrative Operations", function () {
        it("Should handle faucet configuration changes during active use", async function () {
            const { aegisToken, aegisFaucet, owner, user1, user2 } = await loadFixture(deployIntegrationFixture);
            
            // Usuario 1 reclama con configuración original
            await aegisFaucet.connect(user1).requestTokens();
            
            // Admin cambia configuración
            await aegisFaucet.configureFaucet(
                ethers.parseEther("200"), // Nueva cantidad
                12 * 60 * 60, // 12 horas
                ethers.parseEther("10000") // Mantener límite diario
            );
            
            // Usuario 2 reclama con nueva configuración
            await aegisFaucet.connect(user2).requestTokens();
            
            // Verificar diferentes cantidades reclamadas
            expect(await aegisToken.balanceOf(user1.address)).to.equal(ethers.parseEther("100"));
            expect(await aegisToken.balanceOf(user2.address)).to.equal(ethers.parseEther("200"));
            
            // Usuario 1 espera nuevo cooldown y reclama con nueva cantidad
            await time.increase(12 * 60 * 60 + 1);
            await aegisFaucet.connect(user1).requestTokens();
            
            expect(await aegisToken.balanceOf(user1.address)).to.equal(ethers.parseEther("300"));
        });
        
        it("Should handle emergency scenarios correctly", async function () {
            const { aegisToken, aegisFaucet, owner, user1, user2 } = await loadFixture(deployIntegrationFixture);
            
            // Usuarios reclaman normalmente
            await aegisFaucet.connect(user1).requestTokens();
            await aegisFaucet.connect(user2).requestTokens();
            
            // Emergencia: pausar faucet
            await aegisFaucet.pause();
            
            // Nuevos usuarios no pueden reclamar
            await expect(aegisFaucet.connect(user1).requestTokens())
                .to.be.rejectedWith("EnforcedPause");
            
            // Admin puede retirar tokens en emergencia
            const emergencyWithdraw = ethers.parseEther("1000000");
            await aegisFaucet.emergencyWithdraw(emergencyWithdraw);
            
            // Reactivar faucet
            await aegisFaucet.unpause();
            
            // Verificar que el sistema sigue funcionando
            await time.increase(86401);
            await aegisFaucet.connect(user1).requestTokens();
        });
    });
    
    describe("Security and Edge Cases", function () {
        it("Should prevent double spending and maintain consistency", async function () {
            const { aegisToken, aegisFaucet, user1 } = await loadFixture(deployIntegrationFixture);
            
            // Primera reclamación
            await aegisFaucet.connect(user1).requestTokens();
            const balanceAfterFirst = await aegisToken.balanceOf(user1.address);
            
            // Intentar reclamar inmediatamente (debería fallar)
            await expect(aegisFaucet.connect(user1).requestTokens())
                .to.be.rejectedWith("AEGISFaucet: cooldown period not elapsed");
            
            // Balance no debería cambiar
            expect(await aegisToken.balanceOf(user1.address)).to.equal(balanceAfterFirst);
            
            // Después del cooldown debería funcionar
            await time.increase(86401);
            await aegisFaucet.connect(user1).requestTokens();
            
            expect(await aegisToken.balanceOf(user1.address)).to.equal(balanceAfterFirst + ethers.parseEther("100"));
        });
        
        it("Should handle token contract being paused", async function () {
            const { aegisToken, aegisFaucet, owner, user1 } = await loadFixture(deployIntegrationFixture);
            
            // Pausar el contrato de token
            await aegisToken.pause();
            
            // Faucet no debería poder transferir tokens
            await expect(aegisFaucet.connect(user1).requestTokens())
                .to.be.rejectedWith("EnforcedPause");
            
            // Reactivar token
            await aegisToken.unpause();
            
            // Ahora debería funcionar
            await expect(aegisFaucet.connect(user1).requestTokens())
                .to.emit(aegisFaucet, "TokensClaimed");
        });
        
        it("Should maintain correct state across ownership transfers", async function () {
            const { aegisToken, aegisFaucet, owner, user1, user2 } = await loadFixture(deployIntegrationFixture);
            
            // Usuario reclama con owner original
            await aegisFaucet.connect(user1).requestTokens();
            
            // Transferir ownership del token
            await aegisToken.transferOwnership(user2.address);
            
            // Transferir ownership del faucet
            await aegisFaucet.transferOwnership(user2.address);
            
            // Nuevo owner puede administrar
            await aegisFaucet.connect(user2).configureFaucet(
                ethers.parseEther("150"),
                86400,
                ethers.parseEther("10000")
            );
            
            // Usuarios pueden seguir reclamando
            await time.increase(86401);
            await aegisFaucet.connect(user1).requestTokens();
            
            expect(await aegisToken.balanceOf(user1.address)).to.equal(ethers.parseEther("250"));
        });
    });
    
    describe("Performance and Scalability", function () {
        it("Should handle high volume of claims efficiently", async function () {
            const { aegisToken, aegisFaucet, owner } = await loadFixture(deployIntegrationFixture);
            
            // Obtener balance inicial del faucet
            const initialFaucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            
            // Crear múltiples usuarios
            const users = [];
            for (let i = 0; i < 10; i++) {
                const wallet = ethers.Wallet.createRandom().connect(ethers.provider);
                // Enviar ETH para gas
                await owner.sendTransaction({
                    to: wallet.address,
                    value: ethers.parseEther("0.1")
                });
                users.push(wallet);
            }
            
            // Todos reclaman simultáneamente
            const claimPromises = users.map(user => 
                aegisFaucet.connect(user).requestTokens()
            );
            
            await Promise.all(claimPromises);
            
            // Verificar que todos recibieron tokens
            for (const user of users) {
                const balance = await aegisToken.balanceOf(user.address);
                expect(balance).to.equal(ethers.parseEther("100"));
            }
            
            // Verificar balance del faucet
            const expectedFaucetBalance = initialFaucetBalance - (ethers.parseEther("100") * BigInt(users.length));
            const actualFaucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            expect(actualFaucetBalance).to.equal(expectedFaucetBalance);
        });
    });
    
    describe("Real-world Usage Simulation", function () {
        it("Should simulate a complete DApp ecosystem", async function () {
            const { aegisToken, aegisFaucet, owner, user1, user2, user3 } = await loadFixture(deployIntegrationFixture);
            
            // Día 1: Usuarios nuevos llegan y reclaman tokens
            await aegisFaucet.connect(user1).requestTokens();
            await aegisFaucet.connect(user2).requestTokens();
            await aegisFaucet.connect(user3).requestTokens();
            
            // Día 1: Actividad de trading/transferencias
            await aegisToken.connect(user1).transfer(user2.address, ethers.parseEther("20"));
            await aegisToken.connect(user2).transfer(user3.address, ethers.parseEther("30"));
            
            // Día 2: Usuarios reclaman nuevamente (después de cooldown)
            await time.increase(86401);
            await aegisFaucet.connect(user1).requestTokens();
            await aegisFaucet.connect(user2).requestTokens();
            
            // Día 2: Más actividad económica
            await aegisToken.connect(user3).approve(user1.address, ethers.parseEther("50"));
            await aegisToken.connect(user1).transferFrom(user3.address, user2.address, ethers.parseEther("25"));
            
            // Día 3: Admin ajusta parámetros del faucet
            await aegisFaucet.configureFaucet(
                ethers.parseEther("75"), // Reducir cantidad
                48 * 60 * 60, // Aumentar cooldown a 48 horas
                ethers.parseEther("5000") // Ajustar límite diario
            );
            
            // Día 5: Usuarios reclaman con nuevos parámetros (después de 48h cooldown)
            await time.increase(48 * 60 * 60 + 1);
            await aegisFaucet.connect(user1).requestTokens();
            
            // Verificar estado final del ecosistema
            const totalSupply = await aegisToken.totalSupply();
            const user1Balance = await aegisToken.balanceOf(user1.address);
            const user2Balance = await aegisToken.balanceOf(user2.address);
            const user3Balance = await aegisToken.balanceOf(user3.address);
            const faucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            const ownerBalance = await aegisToken.balanceOf(owner.address);
            
            // Verificar conservación de tokens
            const totalBalance = user1Balance + user2Balance + user3Balance + faucetBalance + ownerBalance;
            expect(totalBalance).to.equal(totalSupply);
            
            // Verificar que el ecosistema está funcionando
            expect(user1Balance).to.be.greaterThan(0);
            expect(user2Balance).to.be.greaterThan(0);
            expect(user3Balance).to.be.greaterThan(0);
            expect(faucetBalance).to.be.lessThan(ethers.parseEther("10000000")); // Se han distribuido tokens
        });
    });
});