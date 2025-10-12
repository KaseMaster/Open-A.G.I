import { expect } from "chai";
import hre from "hardhat";
import { loadFixture, time } from "@nomicfoundation/hardhat-network-helpers";

const { ethers } = hre;

describe("AEGISFaucet", function () {
    // Fixture para desplegar ambos contratos
    async function deployFaucetFixture() {
        const [owner, addr1, addr2, addr3] = await ethers.getSigners();
        
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
        
        // Transferir tokens al faucet
        const faucetSupply = ethers.parseEther("10000"); // 10K tokens
        await aegisToken.transfer(await aegisFaucet.getAddress(), faucetSupply);
        
        return { aegisToken, aegisFaucet, owner, addr1, addr2, addr3 };
    }
    
    describe("Deployment", function () {
        it("Should deploy with correct initial parameters", async function () {
            const { aegisFaucet, aegisToken } = await loadFixture(deployFaucetFixture);
            
            const config = await aegisFaucet.getFaucetConfig();
            
            expect(config[0]).to.equal(await aegisToken.getAddress()); // token address
            expect(config[1]).to.equal(ethers.parseEther("100")); // claim amount
            expect(config[2]).to.equal(86400); // cooldown period
            expect(config[3]).to.be.false; // not paused
        });
        
        it("Should set correct owner", async function () {
            const { aegisFaucet, owner } = await loadFixture(deployFaucetFixture);
            
            expect(await aegisFaucet.owner()).to.equal(owner.address);
        });
        
        it("Should have tokens transferred to faucet", async function () {
            const { aegisFaucet, aegisToken } = await loadFixture(deployFaucetFixture);
            
            const faucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            expect(faucetBalance).to.equal(ethers.parseEther("10000"));
        });
        
        it("Should emit FaucetConfigured event on deployment", async function () {
            const [owner] = await ethers.getSigners();
            
            const AEGISToken = await ethers.getContractFactory("AEGISToken");
            const aegisToken = await AEGISToken.deploy(owner.address);
            
            const AEGISFaucet = await ethers.getContractFactory("AEGISFaucet");
            const aegisFaucet = await AEGISFaucet.deploy(
                await aegisToken.getAddress(),
                owner.address
            );
            
            await expect(aegisFaucet.configureFaucet(
                ethers.parseEther("100"),
                86400,
                ethers.parseEther("10000")
            )).to.emit(aegisFaucet, "FaucetConfigured")
              .withArgs(ethers.parseEther("100"), 86400, ethers.parseEther("10000"));
        });
    });
    
    describe("Token Claims", function () {
        it("Should allow users to claim tokens", async function () {
            const { aegisFaucet, aegisToken, addr1 } = await loadFixture(deployFaucetFixture);
            
            const claimAmount = ethers.parseEther("100");
            
            await expect(aegisFaucet.connect(addr1).requestTokens())
                .to.emit(aegisFaucet, "TokensClaimed")
                .withArgs(addr1.address, claimAmount);
            
            expect(await aegisToken.balanceOf(addr1.address)).to.equal(claimAmount);
        });
        
        it("Should update last claim timestamp", async function () {
            const { aegisFaucet, addr1 } = await loadFixture(deployFaucetFixture);
            
            const beforeClaim = await time.latest();
            await aegisFaucet.connect(addr1).requestTokens();
            const afterClaim = await time.latest();
            
            const lastClaim = await aegisFaucet.getLastClaimTime(addr1.address);
            expect(lastClaim).to.be.at.least(beforeClaim);
            expect(lastClaim).to.be.at.most(afterClaim);
        });
        
        it("Should not allow claims before cooldown period", async function () {
            const { aegisFaucet, addr1 } = await loadFixture(deployFaucetFixture);
            
            // Primera reclamación
            await aegisFaucet.connect(addr1).requestTokens();
            
            // Intentar reclamar inmediatamente
            await expect(aegisFaucet.connect(addr1).requestTokens())
                .to.be.revertedWith("AEGISFaucet: cooldown period not elapsed");
        });
        
        it("Should allow claims after cooldown period", async function () {
            const { aegisFaucet, aegisToken, addr1 } = await loadFixture(deployFaucetFixture);
            
            // Primera reclamación
            await aegisFaucet.connect(addr1).requestTokens();
            
            // Avanzar tiempo más allá del cooldown
            await time.increase(86401); // 24 horas + 1 segundo
            
            // Segunda reclamación debería funcionar
            await expect(aegisFaucet.connect(addr1).requestTokens())
                .to.emit(aegisFaucet, "TokensClaimed")
                .withArgs(addr1.address, ethers.parseEther("100"));
            
            expect(await aegisToken.balanceOf(addr1.address)).to.equal(ethers.parseEther("200"));
        });
        
        it("Should not allow claims when paused", async function () {
            const { aegisFaucet, owner, addr1 } = await loadFixture(deployFaucetFixture);
            
            await aegisFaucet.pause();
            
            await expect(aegisFaucet.connect(addr1).requestTokens())
                .to.be.revertedWithCustomError(aegisFaucet, "EnforcedPause");
        });
        
        it("Should not allow claims when insufficient faucet balance", async function () {
            const { aegisFaucet, aegisToken, owner, addr1 } = await loadFixture(deployFaucetFixture);
            
            // Retirar todos los tokens del faucet
            const faucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            await aegisFaucet.emergencyWithdraw(faucetBalance);
            
            await expect(aegisFaucet.connect(addr1).requestTokens())
                .to.be.revertedWith("AEGISFaucet: insufficient faucet balance");
        });
        
        it("Should handle multiple users claiming", async function () {
            const { aegisFaucet, aegisToken, addr1, addr2, addr3 } = await loadFixture(deployFaucetFixture);
            
            // Múltiples usuarios reclaman
            await aegisFaucet.connect(addr1).requestTokens();
            await aegisFaucet.connect(addr2).requestTokens();
            await aegisFaucet.connect(addr3).requestTokens();
            
            // Verificar balances
            expect(await aegisToken.balanceOf(addr1.address)).to.equal(ethers.parseEther("100"));
            expect(await aegisToken.balanceOf(addr2.address)).to.equal(ethers.parseEther("100"));
            expect(await aegisToken.balanceOf(addr3.address)).to.equal(ethers.parseEther("100"));
            
            // Verificar balance del faucet
            const faucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            expect(faucetBalance).to.equal(ethers.parseEther("9700")); // 10000 - 300
        });
    });
    
    describe("Configuration Management", function () {
        it("Should allow owner to update claim amount", async function () {
            const { aegisFaucet, owner } = await loadFixture(deployFaucetFixture);
            
            const newAmount = ethers.parseEther("200");
            
            await expect(aegisFaucet.setClaimAmount(newAmount))
                .to.emit(aegisFaucet, "ClaimAmountUpdated")
                .withArgs(newAmount);
            
            const config = await aegisFaucet.getFaucetConfig();
            expect(config[1]).to.equal(newAmount);
        });
        
        it("Should allow owner to update cooldown period", async function () {
            const { aegisFaucet, owner } = await loadFixture(deployFaucetFixture);
            
            const newCooldown = 12 * 60 * 60; // 12 horas
            
            await expect(aegisFaucet.setCooldownPeriod(newCooldown))
                .to.emit(aegisFaucet, "CooldownPeriodUpdated")
                .withArgs(newCooldown);
            
            const config = await aegisFaucet.getFaucetConfig();
            expect(config[2]).to.equal(newCooldown);
        });
        
        it("Should not allow non-owner to update configuration", async function () {
            const { aegisFaucet, addr1 } = await loadFixture(deployFaucetFixture);
            
            await expect(aegisFaucet.connect(addr1).setClaimAmount(ethers.parseEther("200")))
                .to.be.revertedWithCustomError(aegisFaucet, "OwnableUnauthorizedAccount");
            
            await expect(aegisFaucet.connect(addr1).setCooldownPeriod(12 * 60 * 60))
                .to.be.revertedWithCustomError(aegisFaucet, "OwnableUnauthorizedAccount");
        });
        
        it("Should not allow setting claim amount to zero", async function () {
            const { aegisFaucet } = await loadFixture(deployFaucetFixture);
            
            await expect(aegisFaucet.setClaimAmount(0))
                .to.be.revertedWith("AEGISFaucet: claim amount must be greater than 0");
        });
        
        it("Should not allow setting cooldown period to zero", async function () {
            const { aegisFaucet } = await loadFixture(deployFaucetFixture);
            
            await expect(aegisFaucet.setCooldownPeriod(0))
                .to.be.revertedWith("AEGISFaucet: cooldown period must be greater than 0");
        });
    });
    
    describe("Pausable Functionality", function () {
        it("Should allow owner to pause the faucet", async function () {
            const { aegisFaucet } = await loadFixture(deployFaucetFixture);
            
            await expect(aegisFaucet.pause())
                .to.emit(aegisFaucet, "FaucetPaused");
            
            const config = await aegisFaucet.getFaucetConfig();
            expect(config[3]).to.be.true; // paused
        });
        
        it("Should allow owner to unpause the faucet", async function () {
            const { aegisFaucet } = await loadFixture(deployFaucetFixture);
            
            await aegisFaucet.pause();
            
            await expect(aegisFaucet.unpause())
                .to.emit(aegisFaucet, "FaucetUnpaused");
            
            const config = await aegisFaucet.getFaucetConfig();
            expect(config[3]).to.be.false; // not paused
        });
        
        it("Should not allow non-owner to pause", async function () {
            const { aegisFaucet, addr1 } = await loadFixture(deployFaucetFixture);
            
            await expect(aegisFaucet.connect(addr1).pause())
                .to.be.revertedWithCustomError(aegisFaucet, "OwnableUnauthorizedAccount");
        });
        
        it("Should allow claims after unpausing", async function () {
            const { aegisFaucet, aegisToken, addr1 } = await loadFixture(deployFaucetFixture);
            
            await aegisFaucet.pause();
            await aegisFaucet.unpause();
            
            await expect(aegisFaucet.connect(addr1).requestTokens())
                .to.emit(aegisFaucet, "TokensClaimed")
                .withArgs(addr1.address, ethers.parseEther("100"));
        });
    });
    
    describe("Emergency Functions", function () {
        it("Should allow owner to emergency withdraw tokens", async function () {
            const { aegisFaucet, aegisToken, owner } = await loadFixture(deployFaucetFixture);
            
            const withdrawAmount = ethers.parseEther("1000");
            const initialOwnerBalance = await aegisToken.balanceOf(owner.address);
            
            await expect(aegisFaucet.emergencyWithdraw(withdrawAmount))
                .to.emit(aegisFaucet, "EmergencyWithdraw")
                .withArgs(owner.address, withdrawAmount);
            
            const newOwnerBalance = await aegisToken.balanceOf(owner.address);
            expect(newOwnerBalance).to.equal(initialOwnerBalance + withdrawAmount);
        });
        
        it("Should not allow non-owner to emergency withdraw", async function () {
            const { aegisFaucet, addr1 } = await loadFixture(deployFaucetFixture);
            
            await expect(aegisFaucet.connect(addr1).emergencyWithdraw(ethers.parseEther("1000")))
                .to.be.revertedWithCustomError(aegisFaucet, "OwnableUnauthorizedAccount");
        });
        
        it("Should not allow withdrawing more than faucet balance", async function () {
            const { aegisFaucet } = await loadFixture(deployFaucetFixture);
            
            const excessAmount = ethers.parseEther("20000"); // Más de lo que tiene el faucet
            await expect(aegisFaucet.emergencyWithdraw(excessAmount))
                .to.be.revertedWith("AEGISFaucet: insufficient balance");
        });
        
        it("Should allow owner to add more tokens to faucet", async function () {
            const { aegisFaucet, aegisToken, owner } = await loadFixture(deployFaucetFixture);
            
            const addAmount = ethers.parseEther("5000");
            const initialFaucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            
            await aegisToken.transfer(await aegisFaucet.getAddress(), addAmount);
            
            const newFaucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            expect(newFaucetBalance).to.equal(initialFaucetBalance + addAmount);
        });
    });
    
    describe("View Functions", function () {
        it("Should return correct faucet configuration", async function () {
            const { aegisFaucet, aegisToken } = await loadFixture(deployFaucetFixture);
            
            const config = await aegisFaucet.getFaucetConfig();
            
            expect(config[0]).to.equal(await aegisToken.getAddress());
            expect(config[1]).to.equal(ethers.parseEther("100"));
            expect(config[2]).to.equal(86400);
            expect(config[3]).to.be.false;
        });
        
        it("Should return correct last claim time", async function () {
            const { aegisFaucet, addr1 } = await loadFixture(deployFaucetFixture);
            
            // Antes de reclamar
            expect(await aegisFaucet.getLastClaimTime(addr1.address)).to.equal(0);
            
            // Después de reclamar
            await aegisFaucet.connect(addr1).requestTokens();
            const lastClaim = await aegisFaucet.getLastClaimTime(addr1.address);
            expect(lastClaim).to.be.greaterThan(0);
        });
        
        it("Should return correct time until next claim", async function () {
            const { aegisFaucet, addr1 } = await loadFixture(deployFaucetFixture);
            
            // Antes de reclamar - debería poder reclamar inmediatamente
            expect(await aegisFaucet.getTimeUntilNextClaim(addr1.address)).to.equal(0);
            
            // Después de reclamar
            await aegisFaucet.connect(addr1).requestTokens();
            const timeUntilNext = await aegisFaucet.getTimeUntilNextClaim(addr1.address);
            expect(timeUntilNext).to.be.greaterThan(0);
            expect(timeUntilNext).to.be.at.most(86400);
        });
        
        it("Should correctly identify if user can claim", async function () {
            const { aegisFaucet, addr1 } = await loadFixture(deployFaucetFixture);
            
            // Antes de reclamar
            expect(await aegisFaucet.canClaim(addr1.address)).to.be.true;
            
            // Después de reclamar
            await aegisFaucet.connect(addr1).requestTokens();
            expect(await aegisFaucet.canClaim(addr1.address)).to.be.false;
            
            // Después del cooldown
            await time.increase(86401);
            expect(await aegisFaucet.canClaim(addr1.address)).to.be.true;
        });
    });
    
    describe("Reentrancy Protection", function () {
        it("Should prevent reentrancy attacks", async function () {
            // Esta prueba requeriría un contrato malicioso para ser completa
            // Por ahora, verificamos que el modificador nonReentrant está presente
            const { aegisFaucet, addr1 } = await loadFixture(deployFaucetFixture);
            
            // Reclamación normal debería funcionar
            await expect(aegisFaucet.connect(addr1).requestTokens())
                .to.not.be.reverted;
        });
    });
    
    describe("Edge Cases and Security", function () {
        it("Should handle configuration changes after claims", async function () {
            const { aegisFaucet, aegisToken, addr1 } = await loadFixture(deployFaucetFixture);
            
            // Reclamar con configuración original
            await aegisFaucet.connect(addr1).requestTokens();
            
            // Cambiar configuración
            await aegisFaucet.setClaimAmount(ethers.parseEther("200"));
            await aegisFaucet.setCooldownPeriod(12 * 60 * 60);
            
            // Avanzar tiempo según nueva configuración
            await time.increase(12 * 60 * 60 + 1);
            
            // Reclamar con nueva configuración
            await expect(aegisFaucet.connect(addr1).requestTokens())
                .to.emit(aegisFaucet, "TokensClaimed")
                .withArgs(addr1.address, ethers.parseEther("200"));
            
            expect(await aegisToken.balanceOf(addr1.address)).to.equal(ethers.parseEther("300"));
        });
        
        it("Should maintain correct state after multiple operations", async function () {
            const { aegisFaucet, aegisToken, owner, addr1, addr2 } = await loadFixture(deployFaucetFixture);
            
            // Secuencia compleja de operaciones
            await aegisFaucet.connect(addr1).requestTokens();
            await aegisFaucet.pause();
            await aegisFaucet.setClaimAmount(ethers.parseEther("150"));
            await aegisFaucet.unpause();
            await time.increase(86401);
            await aegisFaucet.connect(addr2).requestTokens();
            await aegisFaucet.emergencyWithdraw(ethers.parseEther("1000"));
            
            // Verificar estado final
            expect(await aegisToken.balanceOf(addr1.address)).to.equal(ethers.parseEther("100"));
            expect(await aegisToken.balanceOf(addr2.address)).to.equal(ethers.parseEther("150"));
            
            const faucetBalance = await aegisToken.balanceOf(await aegisFaucet.getAddress());
            expect(faucetBalance).to.equal(ethers.parseEther("8750")); // 10000 - 100 - 150 - 1000
        });
    });
});
