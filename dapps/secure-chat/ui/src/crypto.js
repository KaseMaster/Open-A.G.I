import nacl from 'tweetnacl'

// Helpers base64
const u8ToBase64 = (u8) => {
  let binary = ''
  const len = u8.byteLength
  for (let i = 0; i < len; i++) binary += String.fromCharCode(u8[i])
  return btoa(binary)
}

const base64ToU8 = (b64) => {
  const binary = atob(b64)
  const len = binary.length
  const u8 = new Uint8Array(len)
  for (let i = 0; i < len; i++) u8[i] = binary.charCodeAt(i)
  return u8
}

export const ensureLocalKeypair = () => {
  const sk = localStorage.getItem('securechat_sk')
  const pk = localStorage.getItem('securechat_pk')
  if (!sk || !pk) {
    const kp = nacl.box.keyPair()
    localStorage.setItem('securechat_sk', u8ToBase64(kp.secretKey))
    localStorage.setItem('securechat_pk', u8ToBase64(kp.publicKey))
  }
}

export const getPublicKeyBase64 = () => localStorage.getItem('securechat_pk') || ''
export const getSecretKeyBase64 = () => localStorage.getItem('securechat_sk') || ''
export const getSecretKeyUint8 = () => base64ToU8(getSecretKeyBase64())

export const loadRecipientPublicKey = async (registryContract, address) => {
  try {
    const pub = await registryContract.getPublicKey(address)
    if (pub) return pub
    return null
  } catch (_) {
    return null
  }
}

export const encryptBytes = (plainU8, recipientPublicKeyBase64, mySecretKeyU8) => {
  const nonce = nacl.randomBytes(nacl.box.nonceLength)
  const recipientU8 = base64ToU8(recipientPublicKeyBase64)
  const cipher = nacl.box(plainU8, nonce, recipientU8, mySecretKeyU8)
  // Concatenate nonce + cipher for storage
  const out = new Uint8Array(nonce.length + cipher.length)
  out.set(nonce, 0)
  out.set(cipher, nonce.length)
  return out
}

export const decryptBytes = (cipherWithNonceU8, senderPublicKeyBase64, mySecretKeyU8) => {
  const nonceLen = nacl.box.nonceLength
  const nonce = cipherWithNonceU8.slice(0, nonceLen)
  const cipher = cipherWithNonceU8.slice(nonceLen)
  const senderU8 = base64ToU8(senderPublicKeyBase64)
  const plain = nacl.box.open(cipher, nonce, senderU8, mySecretKeyU8)
  return plain
}