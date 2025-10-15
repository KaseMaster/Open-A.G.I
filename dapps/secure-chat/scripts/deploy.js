const { ethers } = require('hardhat');

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log('Deploying with account:', deployer.address);

  const ChatRoom = await ethers.getContractFactory('ChatRoom');
  const chat = await ChatRoom.deploy();
  await chat.waitForDeployment();
  console.log('ChatRoom deployed at:', await chat.getAddress());

  const AEGISToken = await ethers.getContractFactory('AEGISToken');
  const aegis = await AEGISToken.deploy(deployer.address);
  await aegis.waitForDeployment();
  console.log('AEGISToken deployed at:', await aegis.getAddress());

  // Optional: configure faucet params (defaults are fine for demo)
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});