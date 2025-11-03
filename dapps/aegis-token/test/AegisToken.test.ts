import { expect } from "chai";
import { ethers } from "hardhat";

describe("AEGISToken", function () {
  it("Should deploy the token with correct name and symbol", async function () {
    const [owner] = await ethers.getSigners();
    const AEGISToken = await ethers.getContractFactory("AEGISToken");
    const token = await AEGISToken.deploy(owner.address);
    await token.waitForDeployment();

    expect(await token.name()).to.equal("AEGIS");
    expect(await token.symbol()).to.equal("AEGIS");
  });

  it("Should have correct initial supply", async function () {
    const [owner] = await ethers.getSigners();
    const AEGISToken = await ethers.getContractFactory("AEGISToken");
    const token = await AEGISToken.deploy(owner.address);
    await token.waitForDeployment();

    const initialSupply = await token.totalSupply();
    expect(initialSupply).to.equal(ethers.parseEther("1000000"));
  });

  it("Should allow owner to mint new tokens", async function () {
    const [owner, addr1] = await ethers.getSigners();
    const AEGISToken = await ethers.getContractFactory("AEGISToken");
    const token = await AEGISToken.deploy(owner.address);
    await token.waitForDeployment();

    const mintAmount = ethers.parseEther("1000");
    await token.mint(addr1.address, mintAmount);

    const balance = await token.balanceOf(addr1.address);
    expect(balance).to.equal(mintAmount);
  });
});