console.log('🔍 VERIFICANDO ESTADO DE LA UI Y LOGS DE CONSOLA...');

const puppeteer = require('puppeteer');

async function debugUIState() {
  let browser;
  try {
    console.log('🚀 Iniciando navegador...');
    browser = await puppeteer.launch({
      headless: false,
      devtools: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    
    // Capturar logs de consola
    const consoleLogs = [];
    page.on('console', msg => {
      const type = msg.type();
      const text = msg.text();
      consoleLogs.push({ type, text, timestamp: new Date().toISOString() });
      console.log(`📝 [${type.toUpperCase()}] ${text}`);
    });
    
    // Capturar errores
    page.on('pageerror', error => {
      console.log('❌ ERROR DE PÁGINA:', error.message);
    });
    
    page.on('requestfailed', request => {
      console.log('❌ REQUEST FAILED:', request.url(), request.failure().errorText);
    });
    
    console.log('🌐 Navegando a la aplicación...');
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });
    
    // Esperar un poco para que la aplicación se cargue completamente
    await page.waitForTimeout(3000);
    
    console.log('');
    console.log('🔍 VERIFICANDO ESTADO DE LA UI...');
    
    // Verificar si el botón de conectar existe
    const connectButton = await page.$('button:contains("Conectar")');
    console.log('🔌 Botón conectar encontrado:', !!connectButton);
    
    // Verificar si hay salas activas mostradas
    const activeRoomsSection = await page.$('[data-testid="active-rooms"], .active-rooms, h3:contains("Salas Activas")');
    console.log('🏠 Sección de salas activas encontrada:', !!activeRoomsSection);
    
    // Verificar si hay elementos de sala
    const roomCards = await page.$$('.room-card, [data-testid="room-card"]');
    console.log('🏠 Tarjetas de sala encontradas:', roomCards.length);
    
    // Verificar el texto de "No hay salas"
    const noRoomsText = await page.$('text="No hay salas activas disponibles"');
    console.log('❌ Mensaje "No hay salas" visible:', !!noRoomsText);
    
    // Verificar si hay indicadores de carga
    const loadingIndicators = await page.$$('.loading, [data-testid="loading"], .spinner');
    console.log('⏳ Indicadores de carga encontrados:', loadingIndicators.length);
    
    console.log('');
    console.log('📊 RESUMEN DE LOGS DE CONSOLA:');
    const errorLogs = consoleLogs.filter(log => log.type === 'error');
    const warningLogs = consoleLogs.filter(log => log.type === 'warning');
    const infoLogs = consoleLogs.filter(log => log.type === 'log' || log.type === 'info');
    
    console.log(`❌ Errores: ${errorLogs.length}`);
    console.log(`⚠️ Advertencias: ${warningLogs.length}`);
    console.log(`ℹ️ Info/Log: ${infoLogs.length}`);
    
    if (errorLogs.length > 0) {
      console.log('');
      console.log('❌ ERRORES ENCONTRADOS:');
      errorLogs.forEach((log, i) => {
        console.log(`${i + 1}. [${log.timestamp}] ${log.text}`);
      });
    }
    
    if (warningLogs.length > 0) {
      console.log('');
      console.log('⚠️ ADVERTENCIAS ENCONTRADAS:');
      warningLogs.forEach((log, i) => {
        console.log(`${i + 1}. [${log.timestamp}] ${log.text}`);
      });
    }
    
    // Buscar logs específicos de loadActiveRooms
    const loadActiveRoomsLogs = consoleLogs.filter(log => 
      log.text.includes('Cargando salas activas') || 
      log.text.includes('Salas activas cargadas') ||
      log.text.includes('Encontrados') && log.text.includes('eventos RoomCreated')
    );
    
    console.log('');
    console.log('🏠 LOGS DE LOADACTIVERROOMS:');
    if (loadActiveRoomsLogs.length > 0) {
      loadActiveRoomsLogs.forEach((log, i) => {
        console.log(`${i + 1}. [${log.timestamp}] ${log.text}`);
      });
    } else {
      console.log('❌ No se encontraron logs de loadActiveRooms');
    }
    
    // Esperar un poco más para capturar logs adicionales
    console.log('');
    console.log('⏳ Esperando logs adicionales...');
    await page.waitForTimeout(5000);
    
    console.log('');
    console.log('✅ Verificación de UI completada');
    
  } catch (error) {
    console.error('❌ Error durante la verificación:', error.message);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

debugUIState();