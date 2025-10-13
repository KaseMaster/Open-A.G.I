console.log('🔍 PROBANDO CONEXIÓN COMPLETA DESDE LA UI...');

const { ethers } = require('ethers');
const { CONTRACTS } = require('./src/config.js');
const fs = require('fs');

async function testUIConnection() {
  try {
    console.log('🚀 Iniciando prueba de conexión UI...');
    
    // Simular el flujo de conexión de la UI
    console.log('1️⃣ Conectando al provider...');
    const provider = new ethers.JsonRpcProvider('http://localhost:8545');
    const signer = await provider.getSigner();
    const address = await signer.getAddress();
    
    console.log('✅ Provider conectado');
    console.log('👤 Dirección:', address);
    
    // Verificar configuración de contratos
    console.log('');
    console.log('2️⃣ Verificando configuración de contratos...');
    console.log('📋 CONTRACTS.UserRegistry:', CONTRACTS.UserRegistry);
    console.log('📋 CONTRACTS.ChatRoom:', CONTRACTS.ChatRoom);
    
    if (!CONTRACTS.ChatRoom) {
      console.log('❌ PROBLEMA: CONTRACTS.ChatRoom no está definido');
      return;
    }
    
    // Cargar ABIs
    console.log('');
    console.log('3️⃣ Cargando ABIs...');
    const chatAbi = JSON.parse(fs.readFileSync('../artifacts/contracts/ChatRoom.sol/ChatRoom.json', 'utf8')).abi;
    console.log('✅ ChatRoom ABI cargado');
    
    // Instanciar contratos
    console.log('');
    console.log('4️⃣ Instanciando contratos...');
    const chat = new ethers.Contract(CONTRACTS.ChatRoom, chatAbi, signer);
    console.log('✅ ChatRoom contract instanciado');
    
    // Simular el useEffect que carga salas activas
    console.log('');
    console.log('5️⃣ Simulando useEffect [chat, provider]...');
    
    if (chat && provider) {
      console.log('✅ Condiciones cumplidas: chat && provider = true');
      console.log('🔄 Ejecutando loadActiveRooms()...');
      
      // Simular loadActiveRooms exactamente como en la UI
      const latest = await provider.getBlockNumber();
      console.log('📊 Bloque actual:', latest);
      
      const filter = chat.filters.RoomCreated();
      const events = await chat.queryFilter(filter, Math.max(0, latest - 10000), latest);
      console.log(`📋 Encontrados ${events.length} eventos RoomCreated`);
      
      const roomsData = [];
      const currentUser = await signer.getAddress();
      
      for (const event of events) {
        const { roomId, admin, participants } = event.args;
        const roomIdNum = Number(roomId);
        
        console.log(`🏠 Procesando sala ${roomIdNum}, admin: ${admin}`);
        
        try {
          const isParticipant = await chat.isParticipant(roomIdNum, currentUser);
          
          if (isParticipant) {
            roomsData.push({
              id: roomIdNum,
              admin: admin,
              participants: participants.length + 1,
              blockNumber: event.blockNumber,
              status: 'Activa'
            });
            console.log(`✅ Sala ${roomIdNum} agregada a la lista`);
          } else {
            console.log(`❌ Usuario no es participante de sala ${roomIdNum}`);
          }
        } catch (e) {
          console.warn(`⚠️ Error verificando participación en sala ${roomIdNum}:`, e.message);
        }
      }
      
      roomsData.sort((a, b) => b.blockNumber - a.blockNumber);
      
      console.log('');
      console.log('🎯 RESULTADO FINAL:');
      console.log(`✅ Salas activas cargadas: ${roomsData.length}`);
      console.log('📊 Datos que se pasarían a setActiveRooms:');
      console.log(JSON.stringify(roomsData, null, 2));
      
      // Simular el renderizado
      console.log('');
      console.log('6️⃣ Simulando renderizado de RoomsView...');
      
      if (roomsData.length === 0) {
        console.log('🖼️ UI mostraría: "No hay salas activas disponibles."');
      } else {
        console.log(`🖼️ UI mostraría: ${roomsData.length} tarjetas de sala`);
        roomsData.forEach(room => {
          console.log(`   - Sala #${room.id} (${room.participants} participantes, Admin: ${room.admin === currentUser ? 'Tú' : room.admin.slice(0,6)+'...'})`);
        });
      }
      
    } else {
      console.log('❌ Condiciones NO cumplidas:');
      console.log('   chat:', !!chat);
      console.log('   provider:', !!provider);
    }
    
    console.log('');
    console.log('✅ Prueba de conexión UI completada');
    
  } catch (error) {
    console.error('❌ Error en prueba de conexión UI:', error.message);
    console.error('Stack:', error.stack);
  }
}

testUIConnection();