import { useEffect, useState } from 'react'
import './App.css'
import './styles.css'
import { ethers } from 'ethers'
import { CONTRACTS, IPFS, DONATIONS, DONATION_ADDRESS, IPFS_GATEWAY } from './config'
import UserRegistryAbi from './abi/UserRegistry.json'
import ChatRoomAbi from './abi/ChatRoom.json'
import AEGISTokenAbi from './abi/AEGISToken.json'
import AEGISFaucetAbi from './abi/AEGISFaucet.json'
import { ensureLocalKeypair, getPublicKeyBase64, getSecretKeyUint8, loadRecipientPublicKey, encryptBytes, decryptBytes } from './crypto'
import { uploadBytesToIPFS, fetchBytesFromIPFS, ipfsInfo, pinAdd, pinRm, isPinned } from './ipfs'

function App() {
  const [provider, setProvider] = useState(null)
  const [signer, setSigner] = useState(null)
  const [address, setAddress] = useState('')
  const [publicKey, setPublicKey] = useState('')
  const [registry, setRegistry] = useState(null)
  const [chat, setChat] = useState(null)
  const [roomId, setRoomId] = useState('')
  const [inviteAddr, setInviteAddr] = useState('')
  const [cid, setCid] = useState('')
  const [contentHash, setContentHash] = useState('')
  const [status, setStatus] = useState('')
  const [recipientAddr, setRecipientAddr] = useState('')
  const [recipientPubKey, setRecipientPubKey] = useState('')
  const [messages, setMessages] = useState([])
  const [fileToShare, setFileToShare] = useState(null)
  const [decryptedPreview, setDecryptedPreview] = useState('')
  const [loadingFeed, setLoadingFeed] = useState(false)
  const [decryptedBytes, setDecryptedBytes] = useState(null)
  const [decryptedFilename, setDecryptedFilename] = useState('')
  const [darkMode, setDarkMode] = useState(false)
  const [networkInfo, setNetworkInfo] = useState(null)
  const [ipfsStatus, setIpfsStatus] = useState(null)
  const [donateAmountEth, setDonateAmountEth] = useState('')
  const [donateAmountAegis, setDonateAmountAegis] = useState('')
  const [donateStatus, setDonateStatus] = useState('')
  const [aegisToken, setAegisToken] = useState(null)
  const [aegisFaucet, setAegisFaucet] = useState(null)
  const [aegisBalance, setAegisBalance] = useState('0')
  const [faucetStatus, setFaucetStatus] = useState('')
  const [lastRequestTime, setLastRequestTime] = useState(0)
  const [toasts, setToasts] = useState([])
  const [actionStates, setActionStates] = useState({})

  useEffect(() => {
    if (window.ethereum) {
      const prov = new ethers.BrowserProvider(window.ethereum)
      setProvider(prov)
    }
  }, [])

  const connect = async () => {
    if (!provider) return
    await provider.send('eth_requestAccounts', [])
    const s = await provider.getSigner()
    setSigner(s)
    const addr = await s.getAddress()
    setAddress(addr)
    setStatus('Conectado')
    try {
      const net = await provider.getNetwork()
      setNetworkInfo({ chainId: Number(net.chainId), name: net.name })
    } catch (_) {}
    // Preparar claves locales
    ensureLocalKeypair()
    // Instanciar contratos si hay direcciones configuradas
    // Instanciar contratos si hay direcciones configuradas
    if (CONTRACTS.UserRegistry) {
      setRegistry(new ethers.Contract(CONTRACTS.UserRegistry, UserRegistryAbi, s))
    }
    if (CONTRACTS.ChatRoom) {
      const c = new ethers.Contract(CONTRACTS.ChatRoom, ChatRoomAbi, s)
      setChat(c)
      // Suscribir eventos para feed
      c.on('MessagePosted', (evRoomId, author, evCid, evHash) => {
        setMessages(prev => [{ type: 'message', roomId: Number(evRoomId), author, cid: evCid, hash: evHash }, ...prev])
      })
      c.on('FileShared', (evRoomId, author, evCid, evHash) => {
        setMessages(prev => [{ type: 'file', roomId: Number(evRoomId), author, cid: evCid, hash: evHash }, ...prev])
      })
    }
    
    // Instanciar contratos AEGIS
    if (DONATIONS.AEGIS_TOKEN.address) {
      const token = new ethers.Contract(DONATIONS.AEGIS_TOKEN.address, AEGISTokenAbi, s)
      setAegisToken(token)
      // Obtener balance inicial
      try {
        const balance = await token.balanceOf(addr)
        setAegisBalance(ethers.formatEther(balance))
      } catch (e) {
        console.error('Error obteniendo balance AEGIS:', e)
      }
    }
    
    if (DONATIONS.AEGIS_FAUCET.address) {
      const faucet = new ethers.Contract(DONATIONS.AEGIS_FAUCET.address, AEGISFaucetAbi, s)
      setAegisFaucet(faucet)
      // Obtener tiempo de última solicitud
      try {
        const lastTime = await faucet.getLastRequestTime(addr)
        setLastRequestTime(Number(lastTime))
      } catch (e) {
        console.error('Error obteniendo tiempo de faucet:', e)
      }
    }
    
    // Consultar estado de IPFS
    try {
      const info = await ipfsInfo()
      setIpfsStatus(info)
    } catch (_) {}
  }

  const publishKey = async () => {
    if (!registry) return setStatus('Configura la dirección de UserRegistry en src/config.js')
    try {
      const pub = publicKey || getPublicKeyBase64()
      const tx = await registry.setPublicKey(pub)
      await tx.wait()
      setStatus('Clave pública publicada')
    } catch (e) {
      console.error(e)
      setStatus('Error publicando clave')
    }
  }

  const createRoom = async () => {
    if (!chat) return setStatus('Configura la dirección de ChatRoom en src/config.js')
    try {
      const tx = await chat.createRoom(inviteAddr ? [inviteAddr] : [])
      await tx.wait()
      // Opcional: leer RoomCreated del receipt
      setStatus('Sala creada')
    } catch (e) {
      console.error(e)
      setStatus('Error creando sala')
    }
  }

  const postMsg = async () => {
    if (!chat) return setStatus('Configura la dirección de ChatRoom en src/config.js')
    try {
      // Obtener clave del destinatario
      let destPub = recipientPubKey
      if (!destPub && recipientAddr && registry) {
        destPub = await loadRecipientPublicKey(registry, recipientAddr)
        setRecipientPubKey(destPub || '')
      }
      if (!destPub) return setStatus('Falta clave pública del destinatario')
      // Simular contenido a cifrar si no hay archivo (usar contentHash como texto)
      const content = new TextEncoder().encode(contentHash || `msg-${Date.now()}`)
      const encrypted = encryptBytes(content, destPub, getSecretKeyUint8())
      // Subir a IPFS (si está configurado)
      if (!IPFS.API_URL) {
        setStatus('IPFS no configurado (src/config.js). Se publica sólo metadato hash')
      }
      const cidValue = IPFS.API_URL ? await uploadBytesToIPFS(encrypted) : (cid || '')
      const hashBytes32 = ethers.keccak256(encrypted)
      const tx = await chat.postMessage(Number(roomId), cidValue || '', hashBytes32)
      await tx.wait()
      setStatus(`Mensaje publicado. CID=${cidValue || '-'} hash=${hashBytes32}`)
    } catch (e) {
      console.error(e)
      setStatus('Error publicando mensaje')
    }
  }

  const shareFile = async () => {
    if (!chat) return setStatus('Configura la dirección de ChatRoom en src/config.js')
    try {
      if (!fileToShare) return setStatus('Selecciona un archivo')
      // Obtener clave del destinatario
      let destPub = recipientPubKey
      if (!destPub && recipientAddr && registry) {
        destPub = await loadRecipientPublicKey(registry, recipientAddr)
        setRecipientPubKey(destPub || '')
      }
      if (!destPub) return setStatus('Falta clave pública del destinatario')
      const buf = await fileToShare.arrayBuffer()
      const plain = new Uint8Array(buf)
      const encrypted = encryptBytes(plain, destPub, getSecretKeyUint8())
      if (!IPFS.API_URL) {
        setStatus('IPFS no configurado (src/config.js). No se puede subir archivo')
        return
      }
      const cidValue = await uploadBytesToIPFS(encrypted)
      const hashBytes32 = ethers.keccak256(encrypted)
      const tx = await chat.shareFile(Number(roomId), cidValue, hashBytes32, fileToShare.name, fileToShare.size)
      await tx.wait()
      setStatus(`Archivo compartido. CID=${cidValue} hash=${hashBytes32}`)
      setFileToShare(null)
    } catch (e) {
      console.error(e)
      setStatus('Error compartiendo archivo')
    }
  }

  const decryptFromCID = async (cidValue, authorAddr, filename = '', itemId = null) => {
    if (itemId) setActionState(itemId, 'decrypt', 'loading')
    try {
      if (!cidValue) {
        setStatus('No hay CID para descifrar')
        if (itemId) setActionState(itemId, 'decrypt', 'error')
        return
      }
      const bytes = await fetchBytesFromIPFS(cidValue)
      // Obtener clave pública del remitente
      let senderPub = ''
      if (registry) senderPub = await loadRecipientPublicKey(registry, authorAddr)
      if (!senderPub) {
        setStatus('El remitente no ha publicado su clave pública')
        if (itemId) setActionState(itemId, 'decrypt', 'error')
        return
      }
      const plain = decryptBytes(bytes, senderPub, getSecretKeyUint8())
      if (!plain) {
        setStatus('No se pudo descifrar (clave/nounce incorrectos)')
        if (itemId) setActionState(itemId, 'decrypt', 'error')
        return
      }
      // Intentar interpretar como texto
      setDecryptedBytes(plain)
      setDecryptedFilename(filename || (plain ? 'contenido.bin' : ''))
      // Intentamos decodificar texto para vista previa
      try {
        const text = new TextDecoder().decode(plain)
        setDecryptedPreview(text)
      } catch (_) {
        setDecryptedPreview('(Contenido binario)')
      }
      setStatus('Contenido descifrado correctamente')
      addToast('Contenido descifrado correctamente', 'success')
      if (itemId) {
        setActionState(itemId, 'decrypt', 'success')
        setTimeout(() => setActionState(itemId, 'decrypt', 'idle'), 2000)
      }
    } catch (e) {
      console.error(e)
      setStatus('Error descifrando desde IPFS')
      addToast('Error descifrando desde IPFS', 'error')
      if (itemId) {
        setActionState(itemId, 'decrypt', 'error')
        setTimeout(() => setActionState(itemId, 'decrypt', 'idle'), 3000)
      }
    }
  }

  const loadHistory = async () => {
    try {
      if (!chat || !provider || !roomId) return setStatus('Configura la sala y el proveedor')
      setLoadingFeed(true)
      const room = Number(roomId)
      const latest = await provider.getBlockNumber()
      const filterMsg = chat.filters.MessagePosted(room)
      const filterFile = chat.filters.FileShared(room)
      const [msgs, files] = await Promise.all([
        chat.queryFilter(filterMsg, Math.max(0, latest - 5000), latest),
        chat.queryFilter(filterFile, Math.max(0, latest - 5000), latest)
      ])
      const items = []
      for (const ev of msgs) {
        const { roomId: r, author, cid, contentHash } = ev.args
        items.push({ type: 'message', roomId: Number(r), author, cid, hash: contentHash, blockNumber: ev.blockNumber })
      }
      for (const ev of files) {
        const { roomId: r, author, cid, contentHash, filename, size } = ev.args
        items.push({ type: 'file', roomId: Number(r), author, cid, hash: contentHash, filename, size: Number(size), blockNumber: ev.blockNumber })
      }
      items.sort((a, b) => (b.blockNumber || 0) - (a.blockNumber || 0))
      setMessages(items)
      setStatus('Historial cargado')
    } catch (e) {
      console.error(e)
      setStatus('Error cargando historial')
    } finally {
      setLoadingFeed(false)
    }
  }

  useEffect(() => {
    // Cargar historial cuando cambia el Room ID
    if (chat && roomId) {
      loadHistory()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chat, roomId])

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text)
      addToast('Copiado al portapapeles', 'success')
    } catch (_) {
      addToast('No se pudo copiar', 'error')
    }
  }

  const downloadDecrypted = () => {
    if (!decryptedBytes) return
    const blob = new Blob([decryptedBytes], { type: 'application/octet-stream' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = decryptedFilename || 'contenido.bin'
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const donateETH = async () => {
    if (!signer) return setDonateStatus('Conecta la wallet')
    try {
      const value = ethers.parseEther(donateAmountEth || '0')
      if (value <= 0n) return setDonateStatus('Monto inválido')
      const tx = await signer.sendTransaction({ to: DONATIONS.ETH_ADDRESS, value })
      setDonateStatus(`Enviando donación ETH... ${tx.hash}`)
      await tx.wait()
      setDonateStatus('Donación ETH enviada. ¡Gracias!')
      addToast('Donación ETH enviada. ¡Gracias!', 'success')
      setDonateAmountEth('')
    } catch (e) {
      console.error(e)
      setDonateStatus('Error enviando donación ETH')
      addToast('Error enviando donación ETH', 'error')
    }
  }

  const donateAEGIS = async () => {
    if (!signer || !aegisToken) return setDonateStatus('Conecta la wallet')
    try {
      const amount = ethers.parseEther(donateAmountAegis || '0')
      if (amount <= 0n) return setDonateStatus('Monto AEGIS inválido')
      
      // Verificar balance
      const balance = await aegisToken.balanceOf(address)
      if (balance < amount) {
        setDonateStatus('Balance AEGIS insuficiente')
        addToast('Balance AEGIS insuficiente', 'error')
        return
      }
      
      const tx = await aegisToken.transfer(DONATIONS.ETH_ADDRESS, amount)
      setDonateStatus(`Enviando donación AEGIS... ${tx.hash}`)
      await tx.wait()
      
      // Actualizar balance
      const newBalance = await aegisToken.balanceOf(address)
      setAegisBalance(ethers.formatEther(newBalance))
      
      setDonateStatus('Donación AEGIS enviada. ¡Gracias!')
      addToast('Donación AEGIS enviada. ¡Gracias!', 'success')
      setDonateAmountAegis('')
    } catch (e) {
      console.error(e)
      setDonateStatus('Error enviando donación AEGIS')
      addToast('Error enviando donación AEGIS', 'error')
    }
  }

  const requestAEGISTokens = async () => {
    if (!signer || !aegisFaucet) return setFaucetStatus('Conecta la wallet')
    try {
      // Verificar cooldown
      const now = Math.floor(Date.now() / 1000)
      const cooldown = DONATIONS.AEGIS_FAUCET.cooldown
      if (lastRequestTime + cooldown > now) {
        const remaining = (lastRequestTime + cooldown - now) / 3600
        setFaucetStatus(`Espera ${remaining.toFixed(1)} horas`)
        addToast(`Espera ${remaining.toFixed(1)} horas para solicitar más tokens`, 'warning')
        return
      }
      
      const tx = await aegisFaucet.requestTokens()
      setFaucetStatus(`Solicitando tokens... ${tx.hash}`)
      await tx.wait()
      
      // Actualizar balance y tiempo
      if (aegisToken) {
        const newBalance = await aegisToken.balanceOf(address)
        setAegisBalance(ethers.formatEther(newBalance))
      }
      setLastRequestTime(now)
      
      setFaucetStatus('Tokens AEGIS recibidos!')
      addToast(`${DONATIONS.AEGIS_FAUCET.amount} AEGIS tokens recibidos!`, 'success')
    } catch (e) {
      console.error(e)
      setFaucetStatus('Error solicitando tokens')
      addToast('Error solicitando tokens del faucet', 'error')
    }
  }

  const addToast = (text, type = 'info') => {
    const id = Date.now() + Math.random()
    setToasts(prev => [...prev, { id, text, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 3000)
  }

  const setActionState = (itemId, action, state) => {
    setActionStates(prev => ({
      ...prev,
      [`${itemId}-${action}`]: state
    }))
  }

  const getActionState = (itemId, action) => {
    return actionStates[`${itemId}-${action}`] || 'idle'
  }

  const verifyIntegrity = async (cidValue, expectedHash, idx, itemId = null) => {
    if (itemId) setActionState(itemId, 'verify', 'loading')
    try {
      if (!cidValue || !expectedHash) {
        addToast('Faltan datos para verificar', 'warning')
        if (itemId) setActionState(itemId, 'verify', 'error')
        return
      }
      const bytes = await fetchBytesFromIPFS(cidValue)
      const actual = ethers.keccak256(bytes)
      const ok = actual.toLowerCase() === expectedHash.toLowerCase()
      addToast(ok ? '✓ Integridad verificada correctamente' : '⚠ Integridad comprometida - hash no coincide', ok ? 'success' : 'error')
      // Marcar en el feed
      setMessages(prev => prev.map((m, i) => i === idx ? { ...m, integrity: ok } : m))
      if (itemId) {
        setActionState(itemId, 'verify', ok ? 'success' : 'error')
        setTimeout(() => setActionState(itemId, 'verify', 'idle'), 3000)
      }
    } catch (e) {
      console.error(e)
      addToast('Error verificando integridad', 'error')
      if (itemId) {
        setActionState(itemId, 'verify', 'error')
        setTimeout(() => setActionState(itemId, 'verify', 'idle'), 3000)
      }
    }
  }

  const togglePin = async (cidValue, idx, itemId = null) => {
    try {
      if (!cidValue) return
      const pinned = await isPinned(cidValue)
      const action = pinned ? 'unpin' : 'pin'
      if (itemId) setActionState(itemId, action, 'loading')
      
      if (pinned) {
        await pinRm(cidValue)
        addToast(`Contenido desanclado de IPFS: ${cidValue}`, 'success')
        setMessages(prev => prev.map((m, i) => i === idx ? { ...m, pinned: false } : m))
      } else {
        await pinAdd(cidValue)
        addToast(`Contenido anclado en IPFS: ${cidValue}`, 'success')
        setMessages(prev => prev.map((m, i) => i === idx ? { ...m, pinned: true } : m))
      }
      
      if (itemId) {
        setActionState(itemId, action, 'success')
        setTimeout(() => setActionState(itemId, action, 'idle'), 2000)
      }
    } catch (e) {
      console.error(e)
      const pinned = await isPinned(cidValue).catch(() => false)
      const action = pinned ? 'unpin' : 'pin'
      addToast(`Error al ${pinned ? 'desanclar' : 'anclar'}: ${e.message}`, 'error')
      if (itemId) {
        setActionState(itemId, action, 'error')
        setTimeout(() => setActionState(itemId, action, 'idle'), 3000)
      }
    }
  }

  const renderActionButton = (itemId, action, onClick, children, className = '') => {
    const state = getActionState(itemId, action)
    const isLoading = state === 'loading'
    const isDisabled = isLoading
    
    let icon = null
    if (state === 'loading') {
      icon = <div className="spinner"></div>
    } else if (state === 'success') {
      icon = <span className="status-icon status-success">✓</span>
    } else if (state === 'error') {
      icon = <span className="status-icon status-error">✗</span>
    }
    
    return (
      <button
        onClick={onClick}
        disabled={isDisabled}
        className={`${className} ${isLoading ? 'loading' : ''}`}
      >
        {icon}
        {children}
      </button>
    )
  }

  return (
    <div className={`container ${darkMode ? 'dark' : ''}`} style={{ background: darkMode ? '#121212' : '#fff', color: darkMode ? '#eee' : '#000' }}>
      <h2>Secure Chat DApp (E2E + IPFS, metadatos on-chain)</h2>
      <div className="section">
        <button onClick={() => setDarkMode(v => !v)}>{darkMode ? 'Modo claro' : 'Modo oscuro'}</button>
      </div>
      <div className="section">
        <button onClick={connect}>Conectar wallet</button>
        <div>Estado: {status}</div>
        <div>Cuenta: {address || '-'}</div>
        <div>Red: {networkInfo ? `${networkInfo.name} (chainId ${networkInfo.chainId})` : '-'}</div>
        <div>IPFS: {ipfsStatus ? `OK (${ipfsStatus.agentVersion || ''})` : 'No conectado'}</div>
      </div>

      <div className="section">
        <h3>Publicar clave pública</h3>
        <input placeholder="Clave pública (base64)" value={publicKey || getPublicKeyBase64()} onChange={e => setPublicKey(e.target.value)} />
        <button onClick={publishKey}>Publicar</button>
      </div>

      <div className="section">
        <h3>Crear sala</h3>
        <input placeholder="Dirección a invitar (opcional)" value={inviteAddr} onChange={e => setInviteAddr(e.target.value)} />
        <button onClick={createRoom}>Crear</button>
      </div>

      <div className="section">
        <h3>Enviar mensaje (CID)</h3>
        <input placeholder="Room ID" value={roomId} onChange={e => setRoomId(e.target.value)} />
        <input placeholder="CID" value={cid} onChange={e => setCid(e.target.value)} />
        <input placeholder="Mensaje (texto que será cifrado)" value={contentHash} onChange={e => setContentHash(e.target.value)} />
        <input placeholder="Dirección destinatario" value={recipientAddr} onChange={e => setRecipientAddr(e.target.value)} />
        <input placeholder="Clave pública destinatario (base64)" value={recipientPubKey} onChange={e => setRecipientPubKey(e.target.value)} />
        <button onClick={postMsg}>Enviar</button>
      </div>

      <div className="section">
        <h3>Compartir archivo cifrado</h3>
        <input type="file" onChange={e => setFileToShare(e.target.files?.[0] || null)} />
        <button onClick={shareFile} disabled={!fileToShare}>Compartir archivo</button>
      </div>

      <div className="section">
        <h3>Feed</h3>
        <button onClick={loadHistory} disabled={!chat || !roomId || loadingFeed}>{loadingFeed ? 'Cargando...' : 'Refrescar'}</button>
        <div style={{ marginBottom: 10 }}>
          <strong>Vista previa descifrada:</strong>
          <div style={{ whiteSpace: 'pre-wrap' }}>{decryptedPreview || '(Vacío)'}</div>
          <div>
            <button onClick={downloadDecrypted} disabled={!decryptedBytes}>Descargar contenido</button>
          </div>
        </div>
        {loadingFeed && messages.length === 0 ? (
          <div className="cards-grid">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="skeleton-card" />
            ))}
          </div>
        ) : messages.length === 0 ? (
          <div className="meta">No hay elementos en el feed</div>
        ) : (
          <div className="cards-grid">
            {messages.map((m, idx) => (
              <div key={idx} className="card">
                <div className="card-header">
                  <span className={`badge ${m.type === 'file' ? 'file' : 'message'}`}>{m.type === 'file' ? 'Archivo' : 'Mensaje'}</span>
                  <span className="card-title">Room {m.roomId}</span>
                </div>
                <div className="card-body">
                  <div className="meta">Autor: {m.author}</div>
                  {m.filename ? <div className="meta">Nombre: {m.filename} {m.size ? `(${m.size} bytes)` : ''}</div> : null}
                  <div className="meta">CID: {m.cid || '-'}</div>
                  <div className="meta">Hash: {m.hash || '-'}</div>
                  {typeof m.integrity === 'boolean' ? (
                    <div className="meta">Integridad: {m.integrity ? 'OK' : 'NO coincide'}</div>
                  ) : null}
                </div>
                <div className="card-actions">
                  {m.cid ? (
                    <>
                      {renderActionButton(
                        `${idx}-${m.cid}`,
                        'decrypt',
                        () => decryptFromCID(m.cid, m.author, m.filename || '', `${idx}-${m.cid}`),
                        'Descifrar',
                        'btn-decrypt'
                      )}
                      {renderActionButton(
                        `${idx}-${m.cid}`,
                        'verify',
                        () => verifyIntegrity(m.cid, m.hash, idx, `${idx}-${m.cid}`),
                        'Verificar integridad',
                        'btn-verify'
                      )}
                      {renderActionButton(
                        `${idx}-${m.cid}`,
                        m.pinned ? 'unpin' : 'pin',
                        () => togglePin(m.cid, idx, `${idx}-${m.cid}`),
                        m.pinned ? 'Unpin' : 'Pin',
                        'btn-pin'
                      )}
                      <a href={`${IPFS_GATEWAY}${m.cid}`} target="_blank" rel="noreferrer">Abrir en gateway</a>
                    </>
                  ) : null}
                  <button onClick={() => copyToClipboard(m.cid || '')} disabled={!m.cid}>Copiar CID</button>
                  <button onClick={() => copyToClipboard(m.hash || '')} disabled={!m.hash}>Copiar hash</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Toasts */}
      <div style={{ position: 'fixed', top: 10, right: 10, zIndex: 9999 }}>
        {toasts.map(t => (
          <div key={t.id} style={{
            marginBottom: 8,
            padding: '8px 12px',
            borderRadius: 6,
            background: t.type === 'error' ? '#ffdddd' : t.type === 'success' ? '#ddffdd' : t.type === 'warning' ? '#fff6cc' : '#eeeeee',
            color: '#333',
            boxShadow: '0 2px 6px rgba(0,0,0,0.15)'
          }}>{t.text}</div>
        ))}
      </div>

      <div className="section">
        <h3>💰 Donaciones</h3>
        
        {/* Donaciones ETH */}
        <div className="donation-section">
          <h4>Donar ETH</h4>
          <div>Dirección: {DONATIONS.ETH_ADDRESS}</div>
          <input 
            placeholder="Monto en ETH" 
            value={donateAmountEth} 
            onChange={e => setDonateAmountEth(e.target.value)} 
          />
          <button onClick={donateETH} disabled={!signer}>Donar ETH</button>
        </div>

        {/* Donaciones AEGIS */}
        <div className="donation-section">
          <h4>🛡️ Donar AEGIS Token</h4>
          <div>Token: {DONATIONS.AEGIS_TOKEN.name} ({DONATIONS.AEGIS_TOKEN.symbol})</div>
          <div>Tu balance: {aegisBalance} AEGIS</div>
          <input 
            placeholder="Monto en AEGIS" 
            value={donateAmountAegis} 
            onChange={e => setDonateAmountAegis(e.target.value)} 
          />
          <button onClick={donateAEGIS} disabled={!signer || !aegisToken}>Donar AEGIS</button>
        </div>

        {/* Faucet AEGIS */}
        <div className="donation-section">
          <h4>🚰 Faucet AEGIS</h4>
          <div>Obtén {DONATIONS.AEGIS_FAUCET.amount} AEGIS tokens gratis cada {DONATIONS.AEGIS_FAUCET.cooldown / 3600} horas</div>
          <button onClick={requestAEGISTokens} disabled={!signer || !aegisFaucet}>
            Solicitar Tokens AEGIS
          </button>
          <div>{faucetStatus}</div>
        </div>

        <div className="donation-status">{donateStatus}</div>
      </div>

      <div className="section">
        <h3>Claves locales</h3>
        <div>Clave pública (base64): {getPublicKeyBase64() || '-'}</div>
        <button onClick={() => copyToClipboard(getPublicKeyBase64() || '')}>Copiar clave pública</button>
        <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>Nunca compartas tu clave secreta. Úsala solo para backup.</div>
        <button onClick={() => copyToClipboard(localStorage.getItem('securechat_sk') || '')}>Copiar clave secreta (base64)</button>
        <button onClick={() => {
          const sk = localStorage.getItem('securechat_sk') || ''
          const blob = new Blob([sk], { type: 'text/plain' })
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = 'securechat_secret_key.b64.txt'
          document.body.appendChild(a)
          a.click()
          a.remove()
          URL.revokeObjectURL(url)
        }}>Descargar clave secreta</button>
      </div>
    </div>
  )
}

export default App