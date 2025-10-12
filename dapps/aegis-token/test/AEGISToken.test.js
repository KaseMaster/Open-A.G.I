import { expect } from "chai";
import hre from "hardhat";
import { loadFixture } from "@nomicfoundation/hardhat-network-helpers";

const { ethers } = hre;

describe("AEGISToken", function () {
    // Fixture para desplegar el contrato
    async function deployAEGISTokenFixture() {
        const [owner, addr1, addr2, addr3] = await ethers.getSigners();
        
        const AEGISToken = await ethers.getContractFactory("AEGISToken");
        const aegisToken = await AEGISToken.deploy(owner.address);
        
        return { aegisToken, owner, addr1, addr2, addr3 };
    }
    
    describe("Deployment", function () {
        it("Should deploy with correct initial parameters", async function () {
            const { aegisToken, owner } = await loadFixture(deployAEGISTokenFixture);
            
            const tokenInfo = await aegisToken.getTokenInfo();
            
            expect(tokenInfo[0]).to.equal("AEGIS Token"); // name
            expect(tokenInfo[1]).to.equal("AEGIS"); // symbol
            expect(tokenInfo[2]).to.equal(18); // decimals
            expect(tokenInfo[3]).to.equal(ethers.parseEther("100000000")); // initial supply
            expect(tokenInfo[4]).to.equal(ethers.parseEther("1000000000")); // max supply
        });
        
        it("Should assign initial supply to owner", async function () {
            const { aegisToken, owner } = await loadFixture(deployAEGISTokenFixture);
            
            const ownerBalance = await aegisToken.balanceOf(owner.address);
            const totalSupply = await aegisToken.totalSupply();
            
            expect(ownerBalance).to.equal(ethers.parseEther("100000000"));
            expect(totalSupply).to.equal(ethers.parseEther("100000000"));
        });
        
        it("Should set correct owner", async function () {
            const { aegisToken, owner } = await loadFixture(deployAEGISTokenFixture);
            
            expect(await aegisToken.owner()).to.equal(owner.address);
        });
        
        it("Should emit TokensMinted event on deployment", async function () {
            const [owner] = await ethers.getSigners();
            const AEGISToken = await ethers.getContractFactory("AEGISToken");
            
            await expect(AEGISToken.deploy(owner.address))
                .to.emit(AEGISToken, "TokensMinted")
                .withArgs(owner.address, ethers.parseEther("100000000"));
        });
    });
    
    describe("Minting", function () {
        it("Should allow owner to mint tokens", async function () {
            const { aegisToken, owner, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            const mintAmount = ethers.parseEther("1000");
            await expect(aegisToken.mint(addr1.address, mintAmount))
                .to.emit(aegisToken, "TokensMinted")
                .withArgs(addr1.address, mintAmount);
            
            expect(await aegisToken.balanceOf(addr1.address)).to.equal(mintAmount);
        });
        
        it("Should not allow non-owner to mint tokens", async function () {
            const { aegisToken, addr1, addr2 } = await loadFixture(deployAEGISTokenFixture);
            
            const mintAmount = ethers.parseEther("1000");
            await expect(aegisToken.connect(addr1).mint(addr2.address, mintAmount))
                .to.be.revertedWithCustomError(aegisToken, "OwnableUnauthorizedAccount");
        });
        
        it("Should not allow minting to zero address", async function () {
            const { aegisToken } = await loadFixture(deployAEGISTokenFixture);
            
            const mintAmount = ethers.parseEther("1000");
            await expect(aegisToken.mint(ethers.ZeroAddress, mintAmount))
                .to.be.revertedWith("AEGISToken: mint to zero address");
        });
        
        it("Should not allow minting zero amount", async function () {
            const { aegisToken, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            await expect(aegisToken.mint(addr1.address, 0))
                .to.be.revertedWith("AEGISToken: mint amount must be greater than 0");
        });
        
        it("Should not allow minting beyond max supply", async function () {
            const { aegisToken, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            // Intentar mintear más allá del suministro máximo
            const excessAmount = ethers.parseEther("900000001"); // 900M + 1 (ya hay 100M iniciales)
            await expect(aegisToken.mint(addr1.address, excessAmount))
                .to.be.revertedWith("AEGISToken: would exceed max supply");
        });
        
        it("Should update total supply correctly after minting", async function () {
            const { aegisToken, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            const initialSupply = await aegisToken.totalSupply();
            const mintAmount = ethers.parseEther("1000");
            
            await aegisToken.mint(addr1.address, mintAmount);
            
            const newSupply = await aegisToken.totalSupply();
            expect(newSupply).to.equal(initialSupply + mintAmount);
        });
    });
    
    describe("Burning", function () {
        it("Should allow token holders to burn their tokens", async function () {
            const { aegisToken, owner } = await loadFixture(deployAEGISTokenFixture);
            
            const burnAmount = ethers.parseEther("1000");
            const initialBalance = await aegisToken.balanceOf(owner.address);
            
            await expect(aegisToken.burn(burnAmount))
                .to.emit(aegisToken, "TokensBurned")
                .withArgs(owner.address, burnAmount);
            
            const newBalance = await aegisToken.balanceOf(owner.address);
            expect(newBalance).to.equal(initialBalance - burnAmount);
        });
        
        it("Should allow burning tokens from another account with allowance", async function () {
            const { aegisToken, owner, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            const burnAmount = ethers.parseEther("1000");
            
            // Dar allowance
            await aegisToken.approve(addr1.address, burnAmount);
            
            // Quemar desde otra cuenta
            await expect(aegisToken.connect(addr1).burnFrom(owner.address, burnAmount))
                .to.emit(aegisToken, "TokensBurned")
                .withArgs(owner.address, burnAmount);
        });
        
        it("Should not allow burning more tokens than balance", async function () {
            const { aegisToken, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            const burnAmount = ethers.parseEther("1000");
            await expect(aegisToken.connect(addr1).burn(burnAmount))
                .to.be.revertedWithCustomError(aegisToken, "ERC20InsufficientBalance");
        });
        
        it("Should update total supply correctly after burning", async function () {
            const { aegisToken, owner } = await loadFixture(deployAEGISTokenFixture);
            
            const initialSupply = await aegisToken.totalSupply();
            const burnAmount = ethers.parseEther("1000");
            
            await aegisToken.burn(burnAmount);
            
            const newSupply = await aegisToken.totalSupply();
            expect(newSupply).to.equal(initialSupply - burnAmount);
        });
    });
    
    describe("Pausable Functionality", function () {
        it("Should allow owner to pause the contract", async function () {
            const { aegisToken } = await loadFixture(deployAEGISTokenFixture);
            
            await expect(aegisToken.pause())
                .to.emit(aegisToken, "ContractPaused");
            
            expect(await aegisToken.paused()).to.be.true;
        });
        
        it("Should allow owner to unpause the contract", async function () {
            const { aegisToken } = await loadFixture(deployAEGISTokenFixture);
            
            await aegisToken.pause();
            
            await expect(aegisToken.unpause())
                .to.emit(aegisToken, "ContractUnpaused");
            
            expect(await aegisToken.paused()).to.be.false;
        });
        
        it("Should not allow non-owner to pause", async function () {
            const { aegisToken, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            await expect(aegisToken.connect(addr1).pause())
                .to.be.revertedWithCustomError(aegisToken, "OwnableUnauthorizedAccount");
        });
        
        it("Should prevent transfers when paused", async function () {
            const { aegisToken, owner, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            await aegisToken.pause();
            
            const transferAmount = ethers.parseEther("1000");
            await expect(aegisToken.transfer(addr1.address, transferAmount))
                .to.be.revertedWithCustomError(aegisToken, "EnforcedPause");
        });
        
        it("Should allow transfers when unpaused", async function () {
            const { aegisToken, owner, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            await aegisToken.pause();
            await aegisToken.unpause();
            
            const transferAmount = ethers.parseEther("1000");
            await expect(aegisToken.transfer(addr1.address, transferAmount))
                .to.not.be.reverted;
        });
    });
    
    describe("Standard ERC20 Functionality", function () {
        it("Should transfer tokens between accounts", async function () {
            const { aegisToken, owner, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            const transferAmount = ethers.parseEther("1000");
            
            await expect(aegisToken.transfer(addr1.address, transferAmount))
                .to.emit(aegisToken, "Transfer")
                .withArgs(owner.address, addr1.address, transferAmount);
            
            expect(await aegisToken.balanceOf(addr1.address)).to.equal(transferAmount);
        });
        
        it("Should handle allowances correctly", async function () {
            const { aegisToken, owner, addr1, addr2 } = await loadFixture(deployAEGISTokenFixture);
            
            const allowanceAmount = ethers.parseEther("1000");
            
            await aegisToken.approve(addr1.address, allowanceAmount);
            expect(await aegisToken.allowance(owner.address, addr1.address)).to.equal(allowanceAmount);
            
            const transferAmount = ethers.parseEther("500");
            await aegisToken.connect(addr1).transferFrom(owner.address, addr2.address, transferAmount);
            
            expect(await aegisToken.balanceOf(addr2.address)).to.equal(transferAmount);
            expect(await aegisToken.allowance(owner.address, addr1.address)).to.equal(allowanceAmount - transferAmount);
        });
    });
    
    describe("Emergency Functions", function () {
        it("Should allow owner to withdraw accidentally sent ETH", async function () {
            const { aegisToken, owner } = await loadFixture(deployAEGISTokenFixture);
            
            // Simular ETH enviado accidentalmente (esto normalmente no debería pasar)
            // En un test real, necesitaríamos un contrato auxiliar para enviar ETH
            // Por ahora, solo verificamos que la función existe y revierte correctamente
            await expect(aegisToken.emergencyWithdrawETH())
                .to.be.revertedWith("AEGISToken: no ETH to withdraw");
        });
        
        it("Should not allow non-owner to call emergency functions", async function () {
            const { aegisToken, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            await expect(aegisToken.connect(addr1).emergencyWithdrawETH())
                .to.be.revertedWithCustomError(aegisToken, "OwnableUnauthorizedAccount");
        });
        
        it("Should not allow withdrawing AEGIS tokens via emergencyWithdrawToken", async function () {
            const { aegisToken } = await loadFixture(deployAEGISTokenFixture);
            
            const tokenAddress = await aegisToken.getAddress();
            await expect(aegisToken.emergencyWithdrawToken(tokenAddress, 1000))
                .to.be.revertedWith("AEGISToken: cannot withdraw AEGIS tokens");
        });
    });
    
    describe("Permit Functionality", function () {
        it("Should have correct domain separator", async function () {
            const { aegisToken } = await loadFixture(deployAEGISTokenFixture);
            
            const domain = await aegisToken.DOMAIN_SEPARATOR();
            expect(domain).to.not.equal(ethers.ZeroHash);
        });
        
        it("Should support EIP-2612 permit", async function () {
            const { aegisToken, owner, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            // Esta es una prueba básica - una implementación completa requeriría
            // generar firmas válidas usando la clave privada
            const nonce = await aegisToken.nonces(owner.address);
            expect(nonce).to.equal(0);
        });
    });
    
    describe("Access Control", function () {
        it("Should transfer ownership correctly", async function () {
            const { aegisToken, owner, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            await aegisToken.transferOwnership(addr1.address);
            expect(await aegisToken.owner()).to.equal(addr1.address);
        });
        
        it("Should allow new owner to mint after ownership transfer", async function () {
            const { aegisToken, owner, addr1, addr2 } = await loadFixture(deployAEGISTokenFixture);
            
            await aegisToken.transferOwnership(addr1.address);
            
            const mintAmount = ethers.parseEther("1000");
            await expect(aegisToken.connect(addr1).mint(addr2.address, mintAmount))
                .to.emit(aegisToken, "TokensMinted")
                .withArgs(addr2.address, mintAmount);
        });
        
        it("Should not allow old owner to mint after ownership transfer", async function () {
            const { aegisToken, owner, addr1, addr2 } = await loadFixture(deployAEGISTokenFixture);
            
            await aegisToken.transferOwnership(addr1.address);
            
            const mintAmount = ethers.parseEther("1000");
            await expect(aegisToken.mint(addr2.address, mintAmount))
                .to.be.revertedWithCustomError(aegisToken, "OwnableUnauthorizedAccount");
        });
    });
    
    describe("Edge Cases and Security", function () {
        it("Should handle maximum values correctly", async function () {
            const { aegisToken, addr1 } = await loadFixture(deployAEGISTokenFixture);
            
            // Mintear hasta el máximo permitido
            const remainingSupply = ethers.parseEther("900000000"); // 1B - 100M inicial
            await aegisToken.mint(addr1.address, remainingSupply);
            
            const totalSupply = await aegisToken.totalSupply();
            expect(totalSupply).to.equal(ethers.parseEther("1000000000"));
        });
        
        it("Should maintain correct state after multiple operations", async function () {
            const { aegisToken, owner, addr1, addr2 } = await loadFixture(deployAEGISTokenFixture);
            
            // Secuencia de operaciones complejas
            await aegisToken.mint(addr1.address, ethers.parseEther("1000"));
            await aegisToken.transfer(addr2.address, ethers.parseEther("500"));
            await aegisToken.connect(addr1).burn(ethers.parseEther("200"));
            
            // Verificar balances finales
            expect(await aegisToken.balanceOf(owner.address)).to.equal(ethers.parseEther("99999500"));
            expect(await aegisToken.balanceOf(addr1.address)).to.equal(ethers.parseEther("800"));
            expect(await aegisToken.balanceOf(addr2.address)).to.equal(ethers.parseEther("500"));
            expect(await aegisToken.totalSupply()).to.equal(ethers.parseEther("99999800"));
        });
    });
});
