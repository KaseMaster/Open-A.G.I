console.log('🔍 SIMULANDO EXACTAMENTE LA FUNCIÓN loadActiveRooms() DE LA UI...');

const { ethers } = require('ethers');
const { CHATROOM_ADDRESS } = require('./src/config.js');
const fs = require('fs');

async function simulateUILoadActiveRooms() {
  try {
    const provider = new ethers.JsonRpcProvider('http://localhost:8545');
    const signer = await provider.getSigner();
    const currentUser = await signer.getAddress();
    
    const chatAbi = JSON.parse(fs.readFileSync('../artifacts/contracts/ChatRoom.sol/ChatRoom.json', 'utf8')).abi;
    const chat = new ethers.Contract(CHATROOM_ADDRESS, chatAbi, signer);
    
    console.log('✅ Contratos cargados correctamente');
    console.log('👤 Usuario actual:', currentUser);
    
    if (!chat || !provider) {
      console.log('❌ chat o provider no están disponibles');
      return;
    }
    
    console.log('✅ chat y provider están disponibles');
    console.log('📊 Cargando salas activas...');
    
    const latest = await provider.getBlockNumber();
    console.log('📊 Bloque actual:', latest);
    
    const filter = chat.filters.RoomCreated();
    const events = await chat.queryFilter(filter, Math.max(0, latest - 10000), latest);
    console.log('📋 Encontrados', events.length, 'eventos RoomCreated');
    
    const roomsData = [];
    
    for (const event of events) {
      const { roomId, admin, participants } = event.args;
      const roomIdNum = Number(roomId);
      
      console.log('');
      console.log('🏠 Procesando sala', roomIdNum, ', admin:', admin);
      console.log('   Participantes iniciales:', participants.length);
      
      try {
        console.log('   🔍 Verificando si usuario es participante...');
        const isParticipant = await chat.isParticipant(roomIdNum, currentUser);
        console.log('   ✅ isParticipant resultado:', isParticipant);
        
        if (isParticipant) {
          const roomData = {
            id: roomIdNum,
            admin: admin,
            participants: participants.length + 1,
            blockNumber: event.blockNumber,
            status: 'Activa'
          };
          
          roomsData.push(roomData);
          console.log('   ✅ Sala', roomIdNum, 'agregada a la lista');
          console.log('   📊 Datos de sala:', JSON.stringify(roomData, null, 6));
        } else {
          console.log('   ❌ Usuario no es participante de sala', roomIdNum);
        }
      } catch (e) {
        console.log('   ⚠️ Error verificando participación en sala', roomIdNum, ':', e.message);
      }
    }
    
    roomsData.sort((a, b) => b.blockNumber - a.blockNumber);
    console.log('');
    console.log('📊 Salas ordenadas por bloque (más recientes primero)');
    
    console.log('');
    console.log('🎯 RESULTADO FINAL - setActiveRooms() recibiría:');
    console.log('   Cantidad de salas:', roomsData.length);
    console.log('   Datos completos:', JSON.stringify(roomsData, null, 2));
    
    console.log('');
    console.log('✅ Salas activas cargadas:', roomsData.length);
    
    if (roomsData.length === 0) {
      console.log('');
      console.log('⚠️ PROBLEMA: activeRooms estará vacío en la UI');
      console.log('   La UI mostrará: "No hay salas activas disponibles."');
    } else {
      console.log('');
      console.log('✅ SUCCESS: activeRooms tendrá', roomsData.length, 'elemento(s)');
      console.log('   La UI debería mostrar las salas correctamente');
    }
    
  } catch (e) {
    console.error('❌ Error en simulación:', e.message);
    console.error('Stack:', e.stack);
  }
}

simulateUILoadActiveRooms();