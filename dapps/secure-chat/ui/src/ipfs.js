import { create } from 'ipfs-http-client'
import { IPFS } from './config'

let client = null
if (IPFS.API_URL) {
  const options = { url: IPFS.API_URL }
  if (IPFS.BASIC_AUTH) options.headers = { Authorization: IPFS.BASIC_AUTH }
  client = create(options)
}

export const uploadBytesToIPFS = async (u8) => {
  if (!client) throw new Error('IPFS client no configurado')
  const { cid } = await client.add(u8)
  return cid.toString()
}

export const fetchBytesFromIPFS = async (cid) => {
  if (!client) throw new Error('IPFS client no configurado')
  const chunks = []
  for await (const chunk of client.cat(cid)) {
    chunks.push(chunk)
  }
  // Concatena chunks en un solo Uint8Array
  let totalLength = 0
  for (const ch of chunks) totalLength += ch.length
  const out = new Uint8Array(totalLength)
  let offset = 0
  for (const ch of chunks) {
    out.set(ch, offset)
    offset += ch.length
  }
  return out
}

export const ipfsInfo = async () => {
  if (!client) return null
  try {
    const info = await client.id()
    return info
  } catch (e) {
    return null
  }
}

export const pinAdd = async (cid) => {
  if (!client) throw new Error('IPFS client no configurado')
  await client.pin.add(cid)
  return true
}

export const pinRm = async (cid) => {
  if (!client) throw new Error('IPFS client no configurado')
  await client.pin.rm(cid)
  return true
}

export const isPinned = async (cid) => {
  if (!client) throw new Error('IPFS client no configurado')
  try {
    for await (const p of client.pin.ls({ paths: cid })) {
      if (p.cid?.toString?.() === cid) return true
    }
  } catch (_) {}
  return false
}