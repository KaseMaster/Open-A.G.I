const { ethers } = require('hardhat');
const fs = require('fs');
const path = require('path');

async function main() {
  const hashPath = path.join(__dirname, '..', 'ipfs_hash.txt');
  if (!fs.existsSync(hashPath)) {
    throw new Error('ipfs_hash.txt no encontrado. Genera un hash de IPFS primero.');
  }
  const ipfsHash = fs.readFileSync(hashPath, 'utf8').trim();

  const chatAddress = '0x5FbDB2315678afecb367f032d93F642f64180aa3';
  const [signer] = await ethers.getSigners();
  console.log('Usando cuenta:', signer.address);

  const chat = await ethers.getContractAt('ChatRoom', chatAddress, signer);

  console.log('Creando sala de prueba con hash:', ipfsHash);
  const txCreate = await chat.createRoom('Sala Demo', ipfsHash);
  await txCreate.wait();

  const rooms = await chat.getRooms();
  const lastRoom = rooms[rooms.length - 1];
  const roomId = Number(lastRoom.id);
  console.log('Sala creada con ID:', roomId, 'Nombre:', lastRoom.name);

  console.log('Uniendo miembro y enviando mensaje...');
  const txJoin = await chat.joinRoom(roomId);
  await txJoin.wait();

  const txMsg = await chat.sendMessage(roomId, ipfsHash);
  await txMsg.wait();

  const messages = await chat.getRoomMessages(roomId);
  console.log('Mensajes en sala', roomId, messages.map(m => ({ sender: m.sender, ipfsHash: m.ipfsHash, ts: Number(m.timestamp) })));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});