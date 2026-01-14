const ipfsBaseUrl = (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:5173') + '/ipfs-api'

export const uploadBytesToIPFS = async (u8) => {
  const body = new FormData()
  body.append('file', new Blob([u8]), 'data.bin')

  const response = await fetch(`${ipfsBaseUrl}/api/v0/add?pin=true`, { method: 'POST', body })
  if (!response.ok) throw new Error(`IPFS add failed: ${response.status}`)

  const text = await response.text()
  const lines = text.trim().split('\n')
  const last = JSON.parse(lines[lines.length - 1])
  return last.Hash || last.Cid || last.cid || last.path
}

export const fetchBytesFromIPFS = async (cid) => {
  const response = await fetch(`${ipfsBaseUrl}/api/v0/cat?arg=${encodeURIComponent(cid)}`, { method: 'POST' })
  if (!response.ok) throw new Error(`IPFS cat failed: ${response.status}`)
  const buf = await response.arrayBuffer()
  return new Uint8Array(buf)
}

export const ipfsInfo = async () => {
  try {
    const response = await fetch(`${ipfsBaseUrl}/api/v0/id`, { method: 'POST' })
    if (!response.ok) return null
    return await response.json()
  } catch {
    return null
  }
}

export const pinAdd = async (cid) => {
  const response = await fetch(`${ipfsBaseUrl}/api/v0/pin/add?arg=${encodeURIComponent(cid)}`, { method: 'POST' })
  if (!response.ok) throw new Error(`IPFS pin add failed: ${response.status}`)
  return true
}

export const pinRm = async (cid) => {
  const response = await fetch(`${ipfsBaseUrl}/api/v0/pin/rm?arg=${encodeURIComponent(cid)}`, { method: 'POST' })
  if (!response.ok) throw new Error(`IPFS pin rm failed: ${response.status}`)
  return true
}

export const isPinned = async (cid) => {
  const response = await fetch(`${ipfsBaseUrl}/api/v0/pin/ls?arg=${encodeURIComponent(cid)}`, { method: 'POST' })
  if (!response.ok) return false
  const data = await response.json()
  return Boolean(data?.Keys?.[cid])
}
