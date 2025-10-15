# SecureChat UI

Esta UI está construida con Vite + React. La aplicación interactúa con la API de IPFS (`localhost:5001`) y con contratos en la red local de Hardhat.

## IPFS en desarrollo: evitar CORS

Los navegadores bloquean peticiones cross-origin si el servidor no incluye cabeceras CORS apropiadas. La API de IPFS requiere peticiones `POST` y, por defecto, no permite el origen del dev server. Hay dos opciones para habilitar el acceso desde la UI:

### Opción A (recomendada): Proxy de Vite

- Ya está configurado un proxy en `vite.config.js` que mapea `\`/ipfs-api` a `http://localhost:5001`.
- El cliente IPFS en `src/App.jsx` usa `create({ url: '/ipfs-api' })`.
- Ventaja: no necesitas cambiar nada en IPFS Desktop; el navegador ve una ruta del mismo origen y no aplica CORS.

Uso:

1. Asegúrate de que IPFS Desktop/daemon esté corriendo (API en `http://localhost:5001`).
2. Arranca el dev server: `npm run dev` en este folder (`ui`).
3. La UI consumirá IPFS usando `/ipfs-api/api/v0/...` vía proxy.

### Opción B: Configurar CORS en IPFS

Si prefieres llamar a `http://localhost:5001` directamente desde el navegador sin proxy, debes habilitar CORS en la API de IPFS.

Desde PowerShell (Windows):

```powershell
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["http://localhost:5174","http://127.0.0.1:5174"]'
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["GET","POST","PUT","OPTIONS"]'
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Headers '["Authorization","Content-Type"]'

# Reinicia el daemon de IPFS (o IPFS Desktop)
ipfs shutdown
ipfs daemon
```

Notas:
- Si usas IPFS Desktop, reinícialo tras cambiar la configuración.
- Asegúrate de incluir el puerto correcto del dev server (por defecto Vite usa `5173` o `5174`).

## Indicadores de estado (StatusBar)

La UI incluye una barra de estado superior que muestra:
- IPFS: verde si la API responde a `version`, rojo si no.
- Red: verde cuando el provider está disponible tras conectar la wallet.
- Cuenta y balance AEGIS.

## Contratos y red local

- Asegúrate de tener Hardhat en `localhost:8545`.
- Conecta MetaMask a la red local y usa la faucet/token según las funciones de perfil.

## Problemas comunes

- Error de `net::ERR_FAILED` en llamadas a `http://localhost:5001/api/v0/...`:
  - Usa la Opción A (proxy de Vite) o habilita CORS (Opción B).
- IPFS sin conexión (StatusBar rojo):
  - Verifica que el daemon está activo y la API accesible.
  - Si estás usando el proxy, que el dev server esté corriendo y que `vite.config.js` no haya sido cacheado (reinicia `npm run dev`).
