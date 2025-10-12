const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ChatRoom", function () {
  it("crea sala y emite mensajes", async function () {
    const [admin, bob] = await ethers.getSigners();
    const ChatRoom = await ethers.getContractFactory("ChatRoom");
    const chat = await ChatRoom.deploy();
    await chat.deployed();

    const tx = await chat.connect(admin).createRoom([bob.address]);
    const receipt = await tx.wait();
    const createdEvent = receipt.events.find(e => e.event === "RoomCreated");
    const roomId = createdEvent.args.roomId.toNumber();

    expect(await chat.isParticipant(roomId, admin.address)).to.equal(true);
    expect(await chat.isParticipant(roomId, bob.address)).to.equal(true);

    await expect(chat.connect(bob).postMessage(roomId, "cid123", ethers.utils.formatBytes32String("hash")))
      .to.emit(chat, "MessagePosted");
  });
});