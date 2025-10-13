console.log('🔍 AGREGANDO LOGS DE DEBUG AL COMPONENTE REACT...');

const fs = require('fs');

// Leer el archivo App.jsx
const appPath = './src/App.jsx';
let content = fs.readFileSync(appPath, 'utf8');

// Verificar si ya tiene logs de debug
if (content.includes('DEBUG_REACT_STATE')) {
  console.log('⚠️ Los logs de debug ya están presentes');
  return;
}

// Agregar logs de debug al useEffect
const useEffectPattern = /useEffect\(\(\) => \{\s*\/\/ Cargar salas activas cuando se conecta el contrato\s*if \(chat && provider\) \{\s*loadActiveRooms\(\)\s*\}\s*\}, \[chat, provider\]\)/;

const debugUseEffect = `useEffect(() => {
    // DEBUG_REACT_STATE: Cargar salas activas cuando se conecta el contrato
    console.log('🔄 useEffect [chat, provider] ejecutado');
    console.log('   chat:', !!chat);
    console.log('   provider:', !!provider);
    
    if (chat && provider) {
      console.log('✅ Condiciones cumplidas, ejecutando loadActiveRooms()');
      loadActiveRooms()
    } else {
      console.log('❌ Condiciones NO cumplidas para loadActiveRooms');
    }
  }, [chat, provider])`;

content = content.replace(useEffectPattern, debugUseEffect);

// Agregar logs de debug al loadActiveRooms
const loadActiveRoomsPattern = /const loadActiveRooms = async \(\) => \{/;

const debugLoadActiveRooms = `const loadActiveRooms = async () => {
    console.log('🔄 DEBUG_REACT_STATE: loadActiveRooms iniciado');
    console.log('   Estado actual activeRooms.length:', activeRooms.length);
    console.log('   loadingRooms:', loadingRooms);`;

content = content.replace(loadActiveRoomsPattern, debugLoadActiveRooms);

// Agregar log después de setActiveRooms
const setActiveRoomsPattern = /setActiveRooms\(roomsData\)/;

const debugSetActiveRooms = `setActiveRooms(roomsData)
      console.log('✅ DEBUG_REACT_STATE: setActiveRooms ejecutado');
      console.log('   Nuevos datos:', roomsData);
      console.log('   Cantidad de salas:', roomsData.length);`;

content = content.replace(setActiveRoomsPattern, debugSetActiveRooms);

// Agregar log en el renderizado de RoomsView
const roomsViewPattern = /const RoomsView = \(\) => \(/;

const debugRoomsView = `const RoomsView = () => {
    console.log('🖼️ DEBUG_REACT_STATE: RoomsView renderizado');
    console.log('   activeRooms.length:', activeRooms.length);
    console.log('   loadingRooms:', loadingRooms);
    console.log('   activeRooms data:', activeRooms);
    
    return (`;

content = content.replace(roomsViewPattern, debugRoomsView);

// Cerrar el return que agregamos
const roomsViewEndPattern = /}\s*\/\/ Componente de salas activas/;
content = content.replace(roomsViewEndPattern, `  )
  } // Componente de salas activas`);

// Escribir el archivo modificado
fs.writeFileSync(appPath, content);

console.log('✅ Logs de debug agregados a App.jsx');
console.log('');
console.log('📋 LOGS AGREGADOS:');
console.log('1. useEffect [chat, provider] - verifica condiciones');
console.log('2. loadActiveRooms() - inicio y estado actual');
console.log('3. setActiveRooms() - datos nuevos');
console.log('4. RoomsView - renderizado y estado');
console.log('');
console.log('🔄 Reinicia el servidor de desarrollo para ver los logs');
console.log('💡 Abre la consola del navegador para ver los mensajes de debug');