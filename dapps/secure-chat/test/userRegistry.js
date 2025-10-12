const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("UserRegistry", function () {
  it("registra y lee clave pública", async function () {
    const [alice] = await ethers.getSigners();
    const UserRegistry = await ethers.getContractFactory("UserRegistry");
    const registry = await UserRegistry.deploy();
    await registry.deployed();

    const key = "base64-x25519-public-key";
    await expect(registry.connect(alice).setPublicKey(key))
      .to.emit(registry, "UserKeyUpdated")
      .withArgs(alice.address, key);

    expect(await registry.getPublicKey(alice.address)).to.equal(key);
  });
});