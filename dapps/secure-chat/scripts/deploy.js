const { ethers } = require("hardhat");

async function main() {
  console.log("Deploying UserRegistry and ChatRoom...");

  const UserRegistry = await ethers.getContractFactory("UserRegistry");
  const userRegistry = await UserRegistry.deploy();
  await userRegistry.waitForDeployment();
  const userRegistryAddr = await userRegistry.getAddress();
  console.log("UserRegistry deployed at:", userRegistryAddr);

  const ChatRoom = await ethers.getContractFactory("ChatRoom");
  const chatRoom = await ChatRoom.deploy();
  await chatRoom.waitForDeployment();
  const chatRoomAddr = await chatRoom.getAddress();
  console.log("ChatRoom deployed at:", chatRoomAddr);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});