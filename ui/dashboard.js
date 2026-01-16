let selectedId = null
let selectedProof = null

const el = (id) => document.getElementById(id)

async function fetchJson(url, opts) {
  const r = await fetch(url, opts)
  const t = await r.text()
  try { return JSON.parse(t) } catch { return { error: t, status: r.status } }
}

function shortHex(s) {
  if (!s) return ""
  if (s.length <= 16) return s
  return s.slice(0, 8) + "…" + s.slice(-8)
}

function setVerifyState(ok, reason) {
  const b = el('verifyBadge')
  b.className = 'badge ' + (ok ? 'ok' : 'bad')
  b.textContent = ok ? 'válido' : ('no válido' + (reason ? ` (${reason})` : ''))
}

async function loadStatus() {
  const s = await fetchJson('/api/audit/status')
  el('root').textContent = s.root || ''
  el('count').textContent = String(s.events ?? 0)
}

async function loadEvents() {
  const q = el('q').value || ''
  const res = await fetchJson('/api/audit/events?query=' + encodeURIComponent(q))
  const events = res.events || []
  const tbl = el('tbl')
  tbl.innerHTML = ''

  for (const ev of events) {
    const tr = document.createElement('tr')
    tr.style.cursor = 'pointer'
    tr.onclick = () => selectEvent(ev.id)
    tr.innerHTML = `
      <td class="mono">${shortHex(ev.id)}</td>
      <td>${ev.event_type}</td>
      <td><span class="badge ${ev.status === 'ok' ? 'ok' : 'bad'}">${ev.status}</span></td>
      <td class="mono">${shortHex(ev.leaf_hash)}</td>
    `
    tbl.appendChild(tr)
  }
}

async function selectEvent(eventId) {
  selectedId = eventId
  selectedProof = null
  el('btnProof').disabled = false
  el('btnVerify').disabled = true
  setVerifyState(true, '')

  const ev = await fetchJson('/api/audit/events/' + encodeURIComponent(eventId))
  el('detail').textContent = JSON.stringify(ev, null, 2)
}

async function loadProof() {
  if (!selectedId) return
  const p = await fetchJson('/api/audit/events/' + encodeURIComponent(selectedId) + '/proof')
  selectedProof = p
  el('btnVerify').disabled = false
  el('detail').textContent = JSON.stringify(p, null, 2)
}

async function verifyProof() {
  if (!selectedProof) return
  const req = {
    leaf_hash: selectedProof.leaf_hash,
    index: selectedProof.index,
    siblings: selectedProof.siblings,
    root: selectedProof.root,
  }
  const res = await fetchJson('/api/audit/verify', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req),
  })
  setVerifyState(!!res.valid, res.reason)
}

async function refreshAll() {
  await loadStatus()
  await loadEvents()
}

el('refresh').onclick = refreshAll
el('q').oninput = () => loadEvents()
el('btnProof').onclick = loadProof
el('btnVerify').onclick = verifyProof

refreshAll()

