const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ChatRoom", function () {
  let chatRoom;
  let owner;
  let addr1;

  beforeEach(async function () {
    [owner, addr1] = await ethers.getSigners();
    const ChatRoom = await ethers.getContractFactory("ChatRoom");
    chatRoom = await ChatRoom.deploy();
    // In ethers v6, deploy() returns a Contract, but we might need to await waitForDeployment() if using recent hardhat-ethers
    // However, usually await factory.deploy() resolves to the contract instance directly in hardhat-ethers v3
  });

  it("Should create a room", async function () {
    await chatRoom.createRoom("General", "QmHash");
    const rooms = await chatRoom.getRooms();
    expect(rooms.length).to.equal(1);
    expect(rooms[0].name).to.equal("General");
  });

  it("Should allow sending messages", async function () {
    await chatRoom.createRoom("General", "QmHash");
    await chatRoom.connect(addr1).sendMessage(0, "QmMessageHash");
    const messages = await chatRoom.getRoomMessages(0);
    expect(messages.length).to.equal(1);
    expect(messages[0].sender).to.equal(addr1.address);
  });
});
