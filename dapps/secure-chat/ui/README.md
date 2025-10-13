# 🛡️ AEGIS Secure Chat

Una aplicación de chat segura y descentralizada construida con React, Web3 y tecnologías de cifrado avanzadas. AEGIS Secure Chat ofrece comunicación privada y segura con una interfaz moderna y elegante.

## ✨ Características Principales

### 🔐 **Seguridad Avanzada**
- **Cifrado End-to-End**: Comunicaciones completamente privadas
- **Autenticación Web3**: Conexión segura con wallets de Ethereum
- **Almacenamiento Descentralizado**: Datos distribuidos en IPFS
- **Protocolos Criptográficos**: Implementación de NaCl para máxima seguridad

### 🎨 **Interfaz Moderna**
- **Diseño Glassmorphism**: Efectos de cristal y transparencias elegantes
- **Sistema de Temas**: Modo claro y oscuro optimizados
- **Responsive Design**: Adaptación perfecta a todos los dispositivos
- **Animaciones Fluidas**: Transiciones suaves y micro-interacciones

### 🌐 **Tecnología Web3**
- **Integración Ethereum**: Soporte completo para wallets Web3
- **Smart Contracts**: Lógica descentralizada en blockchain
- **IPFS Integration**: Almacenamiento distribuido de mensajes
- **Token AEGIS**: Sistema de recompensas integrado

## 🚀 Inicio Rápido

### Prerrequisitos
- Node.js 18+ 
- npm o yarn
- Wallet Web3 (MetaMask recomendado)
- Conexión a Internet

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/aegis-secure-chat.git

# Navegar al directorio
cd aegis-secure-chat/dapps/secure-chat/ui

# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm run dev
```

### Configuración

1. **Configurar Variables de Entorno**
   ```bash
   cp .env.example .env
   # Editar .env con tus configuraciones
   ```

2. **Conectar Wallet**
   - Abrir la aplicación en `http://localhost:5173`
   - Hacer clic en "Conectar Wallet"
   - Autorizar la conexión en tu wallet Web3

3. **Comenzar a Chatear**
   - Crear o unirse a una sala de chat
   - Disfrutar de comunicación segura y privada

## 🎨 Mejoras Visuales Recientes

### **Sistema de Centrado Perfecto**
- ✅ Contenido centrado en todas las resoluciones de pantalla
- ✅ Márgenes automáticos elegantes para mejor legibilidad
- ✅ Proporciones equilibradas en desktop, tablet y móvil

### **Efectos Glassmorphism**
- ✅ Transparencias elegantes con efecto cristal
- ✅ Sombras dinámicas multicapa para profundidad visual
- ✅ Fondos difuminados con `backdrop-filter`

### **Animaciones Avanzadas**
- ✅ Transiciones fluidas de 300-500ms
- ✅ Efectos hover sofisticados
- ✅ Micro-interacciones en todos los elementos

### **Responsive Design Optimizado**
- ✅ Funciones `clamp()` para escalado inteligente
- ✅ Viewport dinámico con `100dvh`
- ✅ Breakpoints inteligentes para todos los dispositivos

## 📱 Compatibilidad de Dispositivos

| Dispositivo | Resolución | Estado | Características |
|-------------|------------|--------|-----------------|
| 📱 **Móvil** | 320px - 767px | ✅ Optimizado | Navegación táctil, menú hamburguesa |
| 📟 **Tablet** | 768px - 1023px | ✅ Optimizado | Layout híbrido, sidebar colapsable |
| 🖥️ **Desktop** | 1024px+ | ✅ Optimizado | Sidebar fijo, hover effects completos |

## 🛠️ Tecnologías Utilizadas

### **Frontend**
- **React 18**: Biblioteca de interfaz de usuario
- **Vite**: Herramienta de construcción rápida
- **CSS3**: Estilos modernos con Flexbox y Grid
- **JavaScript ES6+**: Sintaxis moderna

### **Web3 & Blockchain**
- **ethers.js**: Interacción con Ethereum
- **Web3Modal**: Conexión de wallets
- **IPFS**: Almacenamiento descentralizado
- **Smart Contracts**: Lógica en Solidity

### **Criptografía**
- **NaCl (TweetNaCl)**: Cifrado de mensajes
- **Base58**: Codificación de datos
- **Hashing**: Funciones de hash seguras

## 📂 Estructura del Proyecto

```
ui/
├── src/
│   ├── components/          # Componentes React reutilizables
│   ├── assets/             # Recursos estáticos
│   ├── abi/                # ABIs de Smart Contracts
│   ├── App.jsx             # Componente principal
│   ├── App.css             # Estilos principales
│   ├── styles.css          # Estilos globales
│   ├── config.js           # Configuración de la app
│   ├── crypto.js           # Utilidades criptográficas
│   └── ipfs.js             # Integración con IPFS
├── public/                 # Archivos públicos
├── screenshots/            # Capturas de pantalla
├── VISUAL_IMPROVEMENTS.md  # Documentación de mejoras
└── README.md              # Este archivo
```

## 🎯 Scripts Disponibles

```bash
# Desarrollo
npm run dev          # Servidor de desarrollo
npm run build        # Construcción para producción
npm run preview      # Vista previa de producción

# Calidad de Código
npm run lint         # Linting con ESLint
npm run lint:fix     # Corregir errores de linting automáticamente

# Testing
npm run test         # Ejecutar tests
npm run test:ui      # Tests con interfaz visual
```

## 🔧 Configuración Avanzada

### **Variables de Entorno**
```env
VITE_INFURA_PROJECT_ID=tu_project_id
VITE_IPFS_GATEWAY=https://ipfs.io/ipfs/
VITE_NETWORK_ID=1
VITE_CONTRACT_ADDRESS=0x...
```

### **Personalización de Temas**
```css
:root {
  --primary-color: #2563eb;
  --secondary-color: #7c3aed;
  --accent-color: #f59e0b;
  /* Personaliza más colores en src/styles.css */
}
```

## 🤝 Contribuir

1. **Fork** el repositorio
2. **Crear** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abrir** un Pull Request

### **Guías de Contribución**
- Seguir las convenciones de código existentes
- Añadir tests para nuevas funcionalidades
- Actualizar documentación cuando sea necesario
- Respetar el sistema de temas y responsive design

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

### **Documentación**
- 📖 [Guía de Mejoras Visuales](./VISUAL_IMPROVEMENTS.md)
- 📸 [Capturas de Pantalla](./screenshots/)
- 🔧 [Configuración Avanzada](./docs/advanced-config.md)

### **Comunidad**
- 💬 [Discord](https://discord.gg/aegis-chat)
- 🐦 [Twitter](https://twitter.com/aegis_secure)
- 📧 [Email](mailto:support@aegis-chat.com)

### **Reportar Issues**
Si encuentras algún problema:
1. Revisar [issues existentes](https://github.com/tu-usuario/aegis-secure-chat/issues)
2. Crear un [nuevo issue](https://github.com/tu-usuario/aegis-secure-chat/issues/new) con detalles
3. Incluir capturas de pantalla si es relevante

## 🎉 Agradecimientos

- **React Team** - Por la increíble biblioteca
- **Vite Team** - Por las herramientas de desarrollo rápidas
- **Ethereum Foundation** - Por la infraestructura Web3
- **IPFS Team** - Por el almacenamiento descentralizado
- **Comunidad Open Source** - Por las contribuciones y feedback

---

**AEGIS Secure Chat** - Comunicación privada, segura y descentralizada para la era Web3.

*Construido con ❤️ por la comunidad AEGIS*
