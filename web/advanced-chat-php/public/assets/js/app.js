/**
 * OpenAGI Secure Chat+ - JavaScript Frontend
 * Programador Principal: Jose Gómez alias KaseMaster
 * Contacto: kasemaster@aegis-framework.com
 * Versión: 2.0.0
 * Licencia: MIT
 */

(() => {
  const roomsEl = document.getElementById('rooms');
  const messagesEl = document.getElementById('messages');
  const chatTitleEl = document.getElementById('chatTitle');
  const chatSubEl = document.getElementById('chatSub');
  const messageInputEl = document.getElementById('messageInput');
  const fileInputEl = document.getElementById('fileInput');
  const encryptToggleEl = document.getElementById('encryptToggle');
  const sendBtnEl = document.getElementById('sendBtn');
  const openagiStatusEl = document.getElementById('openagiStatus');
  const btnLoginEl = document.getElementById('btnLogin');
  const btnLogoutEl = document.getElementById('btnLogout');
  const userBadgeEl = document.getElementById('userBadge');
  const roomActionsEl = document.getElementById('roomActions');
  const toastContainerEl = document.getElementById('toastContainer');

  const btnOpenCreate = document.getElementById('btnOpenCreate');
  const createModal = document.getElementById('createModal');
  const createRoomBtn = document.getElementById('createRoomBtn');
  const cancelCreateBtn = document.getElementById('cancelCreateBtn');
  const newRoomNameEl = document.getElementById('newRoomName');
  const newRoomAccessFreeEl = document.getElementById('newRoomAccessFree');
  const membersModal = document.getElementById('membersModal');
  const membersListEl = document.getElementById('membersList');
  const closeMembersBtn = document.getElementById('closeMembersBtn');

  let currentRoomId = null;
  let currentRoomAccess = 'open';
  let isMember = true;
  let userRole = null; // 'owner' (creador), 'admin', 'member', 'viewer' o null
  let isMuted = false;
  let isBanned = false;
  let pollInterval = null;
  let ws = null;
  let wsHeartbeat = null;
  let wsRetries = 0;
  const wsBackoff = [1000, 2000, 5000, 10000];
  let sessionToken = localStorage.getItem('openagi_session_token') || null;
  let userAddress = localStorage.getItem('openagi_user_address') || null;
  // Intervalo global para refrescar el tiempo restante del ban en UI
  let banRemainInterval = null;

  const api = {
    async getRooms() {
      const res = await fetch('/api.php?action=rooms');
      return res.json();
    },
    async roomInfo(roomId) {
      const res = await fetch(`/api.php?action=room_info&room_id=${encodeURIComponent(roomId)}`, { headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async createRoom(name, access='open') {
      const form = new FormData();
      form.append('action', 'create_room');
      form.append('name', name);
      form.append('access', access);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async getMessages(roomId) {
      const res = await fetch(`/api.php?action=messages&room_id=${encodeURIComponent(roomId)}`);
      return res.json();
    },
    async sendMessage(roomId, text, author='usuario') {
      const form = new FormData();
      form.append('action', 'send_message');
      form.append('room_id', roomId);
      form.append('text', text);
      form.append('author', author);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async joinRoom(roomId) {
      const form = new FormData();
      form.append('action', 'join_room');
      form.append('room_id', roomId);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async leaveRoom(roomId) {
      const form = new FormData();
      form.append('action', 'leave_room');
      form.append('room_id', roomId);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async addMember(roomId, address, role='member') {
      const form = new FormData();
      form.append('action', 'add_member');
      form.append('room_id', roomId);
      form.append('address', address);
      form.append('role', role);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async removeMember(roomId, address) {
      const form = new FormData();
      form.append('action', 'remove_member');
      form.append('room_id', roomId);
      form.append('address', address);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async setRole(roomId, address, role) {
      const form = new FormData();
      form.append('action', 'set_role');
      form.append('room_id', roomId);
      form.append('address', address);
      form.append('role', role);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async muteMember(roomId, address) {
      const form = new FormData();
      form.append('action', 'mute_member');
      form.append('room_id', roomId);
      form.append('address', address);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async unmuteMember(roomId, address) {
      const form = new FormData();
      form.append('action', 'unmute_member');
      form.append('room_id', roomId);
      form.append('address', address);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async banMember(roomId, address) {
      const form = new FormData();
      form.append('action', 'ban_member');
      form.append('room_id', roomId);
      form.append('address', address);
      if (arguments.length >= 3 && arguments[2] !== undefined) {
        form.append('reason', arguments[2] || '');
      }
      if (arguments.length >= 4 && arguments[3] !== undefined) {
        form.append('expires_at', arguments[3] || '');
      }
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async unbanMember(roomId, address) {
      const form = new FormData();
      form.append('action', 'unban_member');
      form.append('room_id', roomId);
      form.append('address', address);
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      return res.json();
    },
    async openagiStatus() {
      // Primero intenta FastAPI en 8182; si falla, usa PHP
      try {
        const res = await fetch('http://localhost:8182/status');
        if (res.ok) return res.json();
      } catch {}
      const res2 = await fetch('/openagi.php?action=status');
      return res2.json();
    }
  };

  // Cliente hacia FastAPI para cifrado/descifrado
  const apiFast = {
    async encrypt(payload, room_id) {
      const res = await fetch('http://localhost:8182/crypto/encrypt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ payload, room_id })
      });
      return res.json();
    },
    async decrypt(payload, room_id) {
      const res = await fetch('http://localhost:8182/crypto/decrypt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ payload, room_id })
      });
      return res.json();
    },
    async ipfsUpload(content) {
      const res = await fetch('http://localhost:8182/ipfs/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content })
      });
      return res.json();
    },
    async authChallenge(address) {
      const res = await fetch(`http://localhost:8182/auth/challenge?address=${encodeURIComponent(address)}`);
      return res.json();
    },
    async authVerify(address, signature, message) {
      const res = await fetch('http://localhost:8182/auth/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address, signature, message })
      });
      return res.json();
    },
    async publishMessage(room_id, message) {
      const res = await fetch('http://localhost:8182/events/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ room_id, message })
      });
      return res.json();
    }
  };

  // Toast notifications (no intrusivos)
  function showToast(message, type = 'info', duration = 4000) {
    try {
      const container = toastContainerEl || document.getElementById('toastContainer');
      if (!container) {
        // Fallback suave si no hay contenedor
        console[type === 'error' ? 'error' : 'log'](message);
        return;
      }
      const toast = document.createElement('div');
      toast.className = `toast toast--${type}`;
      toast.textContent = message;
      // Mostrar y ocultar con animación
      container.appendChild(toast);
      requestAnimationFrame(() => {
        toast.classList.add('show');
      });
      const hide = () => {
        toast.classList.remove('show');
        toast.classList.add('hide');
        toast.addEventListener('transitionend', () => {
          try { container.removeChild(toast); } catch {}
        }, { once: true });
      };
      // Cerrar manualmente al hacer click
      toast.addEventListener('click', hide);
      // Ocultar tras duración
      setTimeout(hide, Math.max(1000, duration));
    } catch (e) {
      console.error('Toast error', e);
    }
  }

  // Helper: formatear fecha/hora desde timestamp (segundos o ms)
  function formatDateTime(v) {
    try {
      let n = (typeof v === 'string') ? parseInt(v, 10) : v;
      if (!n || isNaN(n)) return '';
      // si parece segundos, convertir a ms
      if (n < 1e12) n = n * 1000;
      const d = new Date(n);
      return d.toLocaleString();
    } catch { return ''; }
  }

  // Helper: parsear duración tipo "30m", "2h", "7d" a segundos
  function parseDurationToSeconds(s) {
    if (!s) return 0;
    const str = String(s).trim().toLowerCase();
    const m = str.match(/^([0-9]+)\s*(m|min|h|d)$/);
    if (!m) return 0;
    const val = parseInt(m[1], 10);
    const unit = m[2];
    if (unit === 'm' || unit === 'min') return val * 60;
    if (unit === 'h') return val * 3600;
    if (unit === 'd') return val * 86400;
    return 0;
  }

  async function logout() {
    try {
      if (sessionToken) {
        await fetch(`http://localhost:8182/auth/logout?token=${encodeURIComponent(sessionToken)}`, { method: 'POST' });
      }
    } catch {}
    sessionToken = null;
    userAddress = null;
    localStorage.removeItem('openagi_session_token');
    localStorage.removeItem('openagi_user_address');
    if (userBadgeEl) userBadgeEl.textContent = 'Invitado';
    if (btnLoginEl) btnLoginEl.style.display = '';
    if (btnLogoutEl) btnLogoutEl.style.display = 'none';
    try { if (ws) ws.close(); } catch {}
    // actualizar estado UI según acceso de sala actual
    isMember = (currentRoomAccess === 'open');
    if (sendBtnEl) sendBtnEl.disabled = (currentRoomAccess === 'restricted');
    if (currentRoomId) {
      updateRoomActions({ id: currentRoomId, access: currentRoomAccess }, []);
    }
  }

  function renderRooms(rooms) {
    roomsEl.innerHTML = '';
    rooms.forEach(r => {
      const item = document.createElement('div');
      item.className = 'room-item';
      item.innerHTML = `
        <div class="room-name">${r.name}</div>
        <div class="room-meta">ID: ${r.id} · Acceso: ${(r.access || 'open') === 'open' ? 'libre' : 'restringido'}</div>
      `;
      item.onclick = () => selectRoom(r);
      roomsEl.appendChild(item);
    });
  }

  function renderMessages(messages) {
    messagesEl.innerHTML = '';
    messages.forEach(m => {
      const div = document.createElement('div');
      div.className = 'message' + (m.author === 'usuario' ? ' message--self' : '');
      const type = m.type || 'text';
      if (type === 'attachment') {
        const preview = buildAttachmentPreview(m);
        div.appendChild(preview);
      } else if (m.enc) {
        (async () => {
          try {
            const r = await apiFast.decrypt(m.text, m.room_id);
            div.textContent = r.ok ? r.plaintext : m.text;
          } catch {
            div.textContent = m.text;
          }
        })();
      } else {
        div.textContent = m.text;
      }
      messagesEl.appendChild(div);
    });
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function selectRoom(room) {
    currentRoomId = room.id;
    chatTitleEl.textContent = room.name;
    currentRoomAccess = (room.access || 'open');
    chatSubEl.textContent = `Sala ID: ${room.id} · Acceso: ${currentRoomAccess === 'open' ? 'libre' : 'restringido'}`;
    // cargar info de sala y miembros si autenticado
    isMember = currentRoomAccess === 'open';
    userRole = null;
    if (sessionToken) {
      try {
        const info = await api.roomInfo(room.id);
        if (info.ok) {
          const members = info.members || [];
          const ua = (userAddress || '').toLowerCase();
          isMember = currentRoomAccess === 'open' ? true : (ua ? members.map(m => m.toLowerCase()).includes(ua) : false);
          const creator = (info.room && info.room.creator_address) ? info.room.creator_address : 'system';
          // rol del usuario
          userRole = null;
          const roles = Array.isArray(info.roles) ? info.roles : [];
          const myR = roles.find(r => (r.address || '').toLowerCase() === ua);
          if (myR && myR.role) userRole = myR.role.toLowerCase();
          if (ua && creator && ua === (creator || '').toLowerCase()) userRole = 'owner';
          // sanciones
          const sanctions = info.sanctions || { muted: [], banned: [] };
          const mutedList = (sanctions.muted || []).map(x => (x || '').toLowerCase());
          isMuted = ua ? mutedList.includes(ua) : false;
          const bannedList = (sanctions.banned || []).map(x => (x || '').toLowerCase());
          isBanned = ua ? bannedList.includes(ua) : false;
          const roleLabel = userRole ? ` · Rol: ${userRole}` : '';
          chatSubEl.textContent = `Sala ID: ${room.id} · Acceso: ${currentRoomAccess === 'open' ? 'libre' : 'restringido'} · Creador: ${creator}${roleLabel}`;
          updateRoomActions(info.room, members);
        } else {
          updateRoomActions(room, []);
        }
      } catch {
        updateRoomActions(room, []);
      }
    } else {
      updateRoomActions(room, []);
    }
    await loadMessages();
    startPolling();
    // abrir WS solo si acceso libre o ya es miembro
    if (currentRoomAccess === 'open' || isMember) {
      openWebSocket(room.id);
    }
    // deshabilitar envío si restringida y no miembro, rol viewer o muted
    sendBtnEl.disabled = (currentRoomAccess === 'restricted' && (!isMember || userRole === 'viewer' || isMuted));
  }

  async function loadMessages() {
    if (!currentRoomId) return;
    const res = await api.getMessages(currentRoomId);
    if (res.ok) renderMessages(res.messages);
  }

  function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(loadMessages, 2500);
  }

  function openWebSocket(roomId) {
    // Evitar conectar si no hay sesión: previene 403 del backend
    if (!sessionToken) {
      return;
    }
    try { if (ws) ws.close(); } catch {}
    const url = `ws://localhost:8182/ws/${encodeURIComponent(roomId)}${sessionToken ? `?token=${encodeURIComponent(sessionToken)}` : ''}`;
    ws = new WebSocket(url);
    ws.onopen = () => {
      wsRetries = 0;
      showToast('Conectado al chat', 'success');
      if (wsHeartbeat) clearInterval(wsHeartbeat);
      wsHeartbeat = setInterval(() => {
        if (ws && ws.readyState === 1) {
          try { ws.send('ping'); } catch {}
        }
      }, 15000);
    };
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data && data.type === 'message' && data.message) {
          appendMessage(data.message);
        }
      } catch {}
    };
    ws.onerror = () => {
      showToast('Error de conexión al chat', 'error');
      // sólo reconectar si seguimos con sesión
      if (sessionToken) scheduleReconnect(roomId);
    };
    ws.onclose = () => {
      showToast('Desconectado del chat', 'warning');
      if (sessionToken) scheduleReconnect(roomId);
    };
  }

  function updateRoomActions(room, members) {
    if (!roomActionsEl) return;
    roomActionsEl.innerHTML = '';
    const access = (room.access || 'open');
    // limpiar cualquier intervalo previo de contador de ban
    if (banRemainInterval) { try { clearInterval(banRemainInterval); } catch {} banRemainInterval = null; }
    // botón ver miembros (si autenticado)
    if (sessionToken) {
      const btnMembers = document.createElement('button');
      btnMembers.className = 'button';
      btnMembers.textContent = 'Ver miembros';
      btnMembers.onclick = async () => {
        membersModal.style.display = 'flex';
        await renderMembers(room.id);
      };
      roomActionsEl.appendChild(btnMembers);
    }
    // botón unirse si restringida y no miembro
    if (access === 'restricted' && sessionToken && !isMember) {
      const btnJoin = document.createElement('button');
      btnJoin.className = 'button button-primary';
      btnJoin.textContent = 'Unirse';
      if (isBanned) {
        btnJoin.disabled = true;
        btnJoin.title = 'Acceso bloqueado: estás baneado en esta sala';
        // Mostrar razón/expiración del ban si está disponible
        (async () => {
          try {
            const info = await api.roomInfo(room.id);
            if (info.ok && info.sanctions && info.sanctions.banned_meta) {
              const ua = (userAddress || '').toLowerCase();
              const meta = info.sanctions.banned_meta[ua];
              if (meta) {
                const hint = document.createElement('div');
                hint.style.marginTop = '6px';
                hint.style.fontSize = '12px';
                hint.style.color = '#888';
                const renderHint = () => {
                  const parts = [];
                  if (meta.reason) parts.push(`Razón: ${meta.reason}`);
                  if (meta.expires_at) {
                    const nowSec = Math.floor(Date.now() / 1000);
                    const remain = Math.max(0, Number(meta.expires_at) - nowSec);
                    const hours = Math.floor(remain / 3600);
                    const mins = Math.floor((remain % 3600) / 60);
                    const remainStr = hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
                    parts.push(`Expira: ${formatDateTime(meta.expires_at)}`);
                    parts.push(`Quedan: ${remainStr}`);
                    // si el ban ha expirado, refrescar acciones y limpiar intervalo
                    if (remain <= 0 && banRemainInterval) {
                      try { clearInterval(banRemainInterval); } catch {}
                      banRemainInterval = null;
                      (async () => {
                        try {
                          const info2 = await api.roomInfo(room.id);
                          if (info2.ok) {
                            updateRoomActions(info2.room, info2.members || []);
                          }
                        } catch {}
                      })();
                    }
                  }
                  hint.textContent = parts.length ? parts.join(' · ') : 'Baneo activo';
                };
                renderHint();
                roomActionsEl.appendChild(hint);
                if (meta.expires_at) {
                  banRemainInterval = setInterval(renderHint, 60000);
                }
              }
            }
          } catch {}
        })();
      }
      btnJoin.onclick = async () => {
        if (btnJoin.disabled) return;
        const res = await api.joinRoom(room.id);
        if (res.ok) {
          showToast('Te has unido a la sala', 'success');
          // refrescar info
          try {
            const info = await api.roomInfo(room.id);
            if (info.ok) {
              const ua = (userAddress || '').toLowerCase();
              isMember = ua ? (info.members || []).map(m => m.toLowerCase()).includes(ua) : false;
              sendBtnEl.disabled = (access === 'restricted' && !isMember);
              updateRoomActions(info.room, info.members || []);
              // abrir WS ahora que somos miembros
              openWebSocket(room.id);
              // limpiar contador de ban si ya somos miembros
              if (banRemainInterval) { try { clearInterval(banRemainInterval); } catch {} banRemainInterval = null; }
            }
          } catch {}
        } else {
          if (res.error === 'banned') {
            showToast('Acceso bloqueado: has sido baneado de esta sala', 'error');
          } else {
            showToast(res.error || 'No se pudo unir a la sala', 'error');
          }
        }
      };
      roomActionsEl.appendChild(btnJoin);
    }
    // botón salir si restringida y es miembro
    if (access === 'restricted' && sessionToken && isMember) {
      const btnLeave = document.createElement('button');
      btnLeave.className = 'button';
      btnLeave.textContent = 'Salir';
      btnLeave.onclick = async () => {
        const res = await api.leaveRoom(room.id);
        if (res.ok) {
          // actualizar estado
          showToast('Has salido de la sala', 'success');
          isMember = false;
          sendBtnEl.disabled = true;
          try { if (ws) ws.close(); } catch {}
          updateRoomActions(room, res.members || []);
          // limpiar contador de ban al salir
          if (banRemainInterval) { try { clearInterval(banRemainInterval); } catch {} banRemainInterval = null; }
        }
      };
      roomActionsEl.appendChild(btnLeave);
    }
  }

  async function renderMembers(roomId) {
    if (!membersListEl) return;
    
    // Limpiar intervalos previos de contadores de ban
    if (window.memberBanIntervals) {
      window.memberBanIntervals.forEach(id => clearInterval(id));
      window.memberBanIntervals = [];
    }
    
    membersListEl.innerHTML = '';
    try {
      const info = await api.roomInfo(roomId);
      if (info.ok) {
        const members = info.members || [];
        const roles = Array.isArray(info.roles) ? info.roles : [];
        const sanctions = info.sanctions || { muted: [], banned: [] };
        const mutedList = (sanctions.muted || []).map(x => (x || '').toLowerCase());
        const bannedList = (sanctions.banned || []).map(x => (x || '').toLowerCase());
        const bannedMeta = (sanctions.banned_meta || {});
        const creator = (info.room && info.room.creator_address) ? (info.room.creator_address || '').toLowerCase() : 'system';
        const ua = (userAddress || '').toLowerCase();
        const isOwner = ua && ua === creator;
        const canManage = isOwner || (roles.find(r => (r.address || '').toLowerCase() === ua)?.role === 'admin');
        // UI: añadir miembro
        if (canManage) {
          const formRow = document.createElement('div');
          formRow.style.display = 'flex';
          formRow.style.gap = '8px';
          formRow.style.alignItems = 'center';
          const inputAddr = document.createElement('input');
          inputAddr.type = 'text';
          inputAddr.placeholder = 'Dirección (0x...)';
          inputAddr.style.flex = '1';
          const selectRole = document.createElement('select');
          ['member','viewer','admin'].forEach(opt => { const o = document.createElement('option'); o.value = opt; o.textContent = opt; selectRole.appendChild(o); });
          const btnAdd = document.createElement('button');
          btnAdd.className = 'button button-primary';
          btnAdd.textContent = 'Añadir';
          btnAdd.onclick = async () => {
            const addr = (inputAddr.value || '').trim();
            const role = selectRole.value;
            if (!addr) return;
            const res = await api.addMember(roomId, addr, role);
            if (res.ok) { await renderMembers(roomId); } else { showToast(res.error || 'Error al añadir', 'error'); }
          };
          formRow.appendChild(inputAddr);
          formRow.appendChild(selectRole);
          formRow.appendChild(btnAdd);
          membersListEl.appendChild(formRow);
        }
        if (members.length === 0) {
          const div = document.createElement('div');
          div.textContent = 'No hay miembros todavía';
          membersListEl.appendChild(div);
        } else {
          members.forEach(m => {
            const row = document.createElement('div');
            row.className = 'badge';
            const addrL = (m || '').toLowerCase();
            const r = roles.find(rr => (rr.address || '').toLowerCase() === addrL);
            const role = r ? r.role : (addrL === creator ? 'owner' : 'member');
            const isM = mutedList.includes(addrL);
            row.textContent = `${m} · rol: ${role}${isM ? ' · muted' : ''}`;
            // Controles de gestión
            if (canManage && addrL !== creator) {
              const controls = document.createElement('span');
              controls.style.marginLeft = '8px';
              const sel = document.createElement('select');
              ['member','viewer','admin'].forEach(opt => { const o = document.createElement('option'); o.value = opt; o.textContent = opt; sel.appendChild(o); });
              sel.value = role === 'owner' ? 'admin' : role;
              const btnSet = document.createElement('button');
              btnSet.className = 'button';
              btnSet.textContent = 'Cambiar';
              btnSet.onclick = async () => {
                const newRole = sel.value;
                const r = await api.setRole(roomId, addrL, newRole);
                if (r.ok) { showToast('Rol actualizado', 'success'); await renderMembers(roomId); } else { showToast(r.error || 'Error al cambiar rol', 'error'); }
              };
              const btnMute = document.createElement('button');
              btnMute.className = 'button';
              btnMute.textContent = isM ? 'Desmutear' : 'Mutear';
              btnMute.style.marginLeft = '6px';
              btnMute.onclick = async () => {
                const r = isM ? await api.unmuteMember(roomId, addrL) : await api.muteMember(roomId, addrL);
                if (r.ok) { showToast(isM ? 'Usuario desmuteado' : 'Usuario muteado', 'success'); await renderMembers(roomId); } else { showToast(r.error || 'Error de sanción', 'error'); }
              };
              const btnBan = document.createElement('button');
              btnBan.className = 'button';
              btnBan.textContent = 'Banear';
              btnBan.style.marginLeft = '6px';
              btnBan.onclick = async () => {
                // Crear modal para ban con UI mejorada
                const banModal = document.createElement('div');
                banModal.style.cssText = `
                  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                  background: rgba(0,0,0,0.5); display: flex; align-items: center;
                  justify-content: center; z-index: 1000;
                `;
                
                const banForm = document.createElement('div');
                banForm.style.cssText = `
                  background: white; padding: 20px; border-radius: 8px;
                  min-width: 300px; max-width: 400px;
                `;
                
                banForm.innerHTML = `
                  <h3 style="margin-top: 0;">Banear Usuario</h3>
                  <p><strong>Usuario:</strong> ${addrL}</p>
                  <div style="margin: 12px 0;">
                    <label>Razón (opcional):</label><br>
                    <input type="text" id="banReason" style="width: 100%; padding: 6px; margin-top: 4px;" placeholder="Motivo del ban">
                  </div>
                  <div style="margin: 12px 0;">
                    <label>Duración:</label><br>
                    <select id="banDuration" style="width: 100%; padding: 6px; margin-top: 4px;">
                      <option value="">Permanente</option>
                      <option value="30m">30 minutos</option>
                      <option value="1h">1 hora</option>
                      <option value="2h">2 horas</option>
                      <option value="6h">6 horas</option>
                      <option value="12h">12 horas</option>
                      <option value="1d">1 día</option>
                      <option value="3d">3 días</option>
                      <option value="7d">7 días</option>
                      <option value="30d">30 días</option>
                    </select>
                  </div>
                  <div style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px;">
                    <button id="banCancel" style="padding: 8px 16px;">Cancelar</button>
                    <button id="banConfirm" style="padding: 8px 16px; background: #ff6b6b; color: white; border: none; border-radius: 4px;">Banear</button>
                  </div>
                `;
                
                banModal.appendChild(banForm);
                document.body.appendChild(banModal);
                
                const reasonInput = banForm.querySelector('#banReason');
                const durationSelect = banForm.querySelector('#banDuration');
                const cancelBtn = banForm.querySelector('#banCancel');
                const confirmBtn = banForm.querySelector('#banConfirm');
                
                reasonInput.focus();
                
                cancelBtn.onclick = () => document.body.removeChild(banModal);
                
                confirmBtn.onclick = async () => {
                  const reason = reasonInput.value.trim();
                  const duration = durationSelect.value;
                  let expiresAt = '';
                  
                  if (duration) {
                    const secs = parseDurationToSeconds(duration);
                    if (secs > 0) {
                      expiresAt = Math.floor(Date.now() / 1000) + secs;
                    }
                  }
                  
                  document.body.removeChild(banModal);
                  const r = await api.banMember(roomId, addrL, reason || '', expiresAt || '');
                  if (r.ok) { 
                    showToast(`Usuario baneado${duration ? ` por ${duration}` : ' permanentemente'}`, 'success'); 
                    await renderMembers(roomId); 
                  } else { 
                    showToast(r.error || 'Error al banear', 'error'); 
                  }
                };
                
                // Cerrar con ESC
                const handleEsc = (e) => {
                  if (e.key === 'Escape') {
                    document.body.removeChild(banModal);
                    document.removeEventListener('keydown', handleEsc);
                  }
                };
                document.addEventListener('keydown', handleEsc);
              };
              const btnDel = document.createElement('button');
              btnDel.className = 'button';
              btnDel.textContent = 'Eliminar';
              btnDel.style.marginLeft = '6px';
              btnDel.onclick = async () => {
                const r = await api.removeMember(roomId, addrL);
                if (r.ok) { showToast('Miembro eliminado', 'success'); await renderMembers(roomId); } else { showToast(r.error || 'Error al eliminar', 'error'); }
              };
              controls.appendChild(sel);
              controls.appendChild(btnSet);
              controls.appendChild(btnMute);
              controls.appendChild(btnBan);
              controls.appendChild(btnDel);
              row.appendChild(controls);
            }
            membersListEl.appendChild(row);
          });
          // Sección de usuarios baneados (solo visible para gestión)
          if (canManage) {
            const title = document.createElement('div');
            title.style.marginTop = '12px';
            title.textContent = 'Baneados';
            membersListEl.appendChild(title);
            if (bannedList.length === 0) {
              const none = document.createElement('div');
              none.textContent = 'No hay usuarios baneados';
              membersListEl.appendChild(none);
            } else {
              bannedList.forEach(addrL => {
                const row = document.createElement('div');
                row.className = 'badge';
                row.textContent = `${addrL}`;
                const meta = bannedMeta[addrL];
                if (meta) {
                  const infoSpan = document.createElement('span');
                  infoSpan.style.marginLeft = '6px';
                  infoSpan.style.color = '#888';
                  const parts = [];
                  if (meta.reason) parts.push(`razón: ${meta.reason}`);
                  if (meta.expires_at) {
                    parts.push(`expira: ${formatDateTime(meta.expires_at)}`);
                    // Agregar tiempo restante dinámico
                    const remainingSpan = document.createElement('span');
                    remainingSpan.style.marginLeft = '4px';
                    remainingSpan.style.fontWeight = 'bold';
                    remainingSpan.style.color = '#ff6b6b';
                    
                    const updateRemaining = () => {
                      const now = Math.floor(Date.now() / 1000);
                      const remaining = meta.expires_at - now;
                      if (remaining > 0) {
                        remainingSpan.textContent = `(${formatDuration(remaining)})`;
                      } else {
                        remainingSpan.textContent = '(expirado)';
                        remainingSpan.style.color = '#51cf66';
                        // Auto-refresh cuando expire
                        setTimeout(() => renderMembers(roomId), 1000);
                      }
                    };
                    
                    updateRemaining();
                    // Actualizar cada minuto
                    const intervalId = setInterval(updateRemaining, 60000);
                    
                    // Limpiar interval cuando se refresque la lista
                    if (!window.memberBanIntervals) window.memberBanIntervals = [];
                    window.memberBanIntervals.push(intervalId);
                    
                    infoSpan.appendChild(remainingSpan);
                  }
                  infoSpan.textContent = parts.join(' · ');
                  row.appendChild(infoSpan);
                }
                const btnUnban = document.createElement('button');
                btnUnban.className = 'button';
                btnUnban.style.marginLeft = '6px';
                btnUnban.textContent = 'Desbanear';
                btnUnban.onclick = async () => {
                  const r = await api.unbanMember(roomId, addrL);
                  if (r.ok) { showToast('Usuario desbaneado', 'success'); await renderMembers(roomId); } else { showToast(r.error || 'Error al desbanear', 'error'); }
                };
                row.appendChild(btnUnban);
                membersListEl.appendChild(row);
              });
            }
          }
        }
      } else {
        const div = document.createElement('div');
        div.textContent = 'Necesitas iniciar sesión para ver miembros';
        membersListEl.appendChild(div);
      }
    } catch {
      const div = document.createElement('div');
      div.textContent = 'Error al cargar miembros';
      membersListEl.appendChild(div);
    }
  }

  function scheduleReconnect(roomId) {
    if (wsHeartbeat) { clearInterval(wsHeartbeat); wsHeartbeat = null; }
    const delay = wsBackoff[Math.min(wsRetries, wsBackoff.length - 1)];
    wsRetries++;
    setTimeout(() => {
      openWebSocket(roomId);
    }, delay);
  }

  function appendMessage(m) {
    const div = document.createElement('div');
    div.className = 'message' + (m.author === 'usuario' ? ' message--self' : '');
    const type = m.type || 'text';
    if (type === 'attachment') {
      const preview = buildAttachmentPreview(m);
      div.appendChild(preview);
    } else if (m.enc) {
      (async () => {
        try {
          const r = await apiFast.decrypt(m.text, m.room_id);
          div.textContent = r.ok ? r.plaintext : m.text;
        } catch {
          div.textContent = m.text;
        }
      })();
    } else {
      div.textContent = m.text;
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function buildAttachmentPreview(m) {
    const mime = m.mime || '';
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '6px';
    const title = document.createElement('div');
    title.textContent = (m.filename || 'Adjunto') + (mime ? ` (${mime})` : '');
    title.style.fontSize = '12px';
    title.style.color = '#7b8aa3';
    const showDataUrl = (url) => {
      if (mime.startsWith('image/')) {
        const img = document.createElement('img');
        img.src = url;
        img.style.maxWidth = '420px';
        img.style.maxHeight = '320px';
        img.style.borderRadius = '8px';
        container.appendChild(img);
      } else if (mime.startsWith('video/')) {
        const vid = document.createElement('video');
        vid.controls = true;
        vid.src = url;
        vid.style.maxWidth = '420px';
        container.appendChild(vid);
      } else if (mime.startsWith('audio/')) {
        const aud = document.createElement('audio');
        aud.controls = true;
        aud.src = url;
        container.appendChild(aud);
      } else if (mime === 'application/pdf') {
        const frame = document.createElement('iframe');
        frame.src = url;
        frame.style.width = '420px';
        frame.style.height = '320px';
        frame.style.border = 'none';
        container.appendChild(frame);
      } else {
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.textContent = 'Abrir adjunto';
        container.appendChild(a);
      }
    };
    const showRemote = (url) => {
      showDataUrl(url);
    };
    container.appendChild(title);
    if (m.enc) {
      (async () => {
        try {
          const res = await fetch(m.ipfs_uri);
          if (!res.ok) throw new Error('ipfs get failed');
          const j = await res.json();
          if (!j.ok || !j.content) throw new Error('no content');
          const dec = await apiFast.decrypt(j.content, m.room_id);
          if (dec.ok && dec.plaintext) {
            const dataUrl = `data:${mime || 'application/octet-stream'};base64,${dec.plaintext}`;
            showDataUrl(dataUrl);
          } else {
            showRemote(m.ipfs_uri || '#');
          }
        } catch (e) {
          console.error('No se pudo descifrar adjunto', e);
          showRemote(m.ipfs_uri || '#');
        }
      })();
    } else {
      showRemote(m.ipfs_uri || '#');
    }
    return container;
  }

  async function connectWalletLogin() {
    try {
      if (!window.ethereum) {
        showToast('MetaMask no encontrado', 'error');
        return;
      }
      await window.ethereum.request({ method: 'eth_requestAccounts' });
      const provider = new ethers.BrowserProvider(window.ethereum);
      const signer = await provider.getSigner();
      const address = await signer.getAddress();
      const chal = await apiFast.authChallenge(address);
      if (!chal.ok || !chal.message) {
        showToast('No se pudo obtener challenge', 'error');
        return;
      }
      const signature = await signer.signMessage(chal.message);
      const ver = await apiFast.authVerify(address, signature, chal.message);
      if (ver.ok && ver.token) {
        sessionToken = ver.token;
        userAddress = ver.address || address;
        localStorage.setItem('openagi_session_token', sessionToken);
        localStorage.setItem('openagi_user_address', userAddress);
        if (userBadgeEl) userBadgeEl.textContent = `Usuario: ${userAddress}`;
        if (btnLoginEl) btnLoginEl.style.display = 'none';
        if (btnLogoutEl) btnLogoutEl.style.display = '';
        // si hay sala seleccionada restringida, refrescar acciones
        if (currentRoomId) {
          try {
            const info = await api.roomInfo(currentRoomId);
            if (info.ok) {
              const ua = (userAddress || '').toLowerCase();
              isMember = (currentRoomAccess === 'open') ? true : (ua ? (info.members || []).map(m => m.toLowerCase()).includes(ua) : false);
              updateRoomActions(info.room, info.members || []);
              if (currentRoomAccess === 'open' || isMember) {
                openWebSocket(currentRoomId);
              }
            }
          } catch {}
        }
      } else {
        showToast('Verificación fallida', 'error');
      }
    } catch (e) {
      console.error('Login wallet error', e);
      showToast('Error de inicio de sesión', 'error');
    }
  }

  // Modal handlers
  btnOpenCreate.onclick = () => {
    createModal.style.display = 'flex';
    newRoomNameEl.value = '';
    newRoomNameEl.focus();
  };
  cancelCreateBtn.onclick = () => {
    createModal.style.display = 'none';
  };
  createRoomBtn.onclick = async () => {
    const name = newRoomNameEl.value.trim();
    if (!name) return;
    const access = newRoomAccessFreeEl && newRoomAccessFreeEl.checked ? 'open' : 'restricted';
    const res = await api.createRoom(name, access);
    if (res.ok) {
      const accessText = access === 'open' ? 'pública' : 'privada';
      showToast(`Sala "${name}" creada (${accessText})`, 'success');
      createModal.style.display = 'none';
      await initRooms();
      // auto-seleccionar la nueva sala
      selectRoom(res.room);
    }
  };

  if (closeMembersBtn) closeMembersBtn.onclick = () => { membersModal.style.display = 'none'; };

  // helper: leer archivo como base64
  function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const res = reader.result;
        // si es data URL, extraer base64; si es ArrayBuffer, convertir
        if (typeof res === 'string' && res.startsWith('data:')) {
          const b64 = res.split(',')[1];
          resolve(b64);
        } else if (res instanceof ArrayBuffer) {
          const bytes = new Uint8Array(res);
          let binary = '';
          for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
          resolve(btoa(binary));
        } else {
          resolve('');
        }
      };
      reader.onerror = reject;
      try {
        reader.readAsDataURL(file);
      } catch (e) {
        reject(e);
      }
    });
  }

  // Send message (con cifrado opcional vía FastAPI y adjuntos vía IPFS)
  sendBtnEl.onclick = async () => {
    if (!currentRoomId) return;
    const files = fileInputEl.files;
    const text = messageInputEl.value.trim();

    // Si hay archivo, subir a IPFS primero y enviar mensaje de adjunto
    if (files && files.length > 0) {
      const file = files[0];
      try {
        let base64 = await readFileAsBase64(file);
        // si toggle cifrado activo, cifrar base64 antes de subir
        if (encryptToggleEl && encryptToggleEl.checked) {
          try {
            const encRes = await apiFast.encrypt(base64, currentRoomId);
            if (encRes.ok && encRes.enc && encRes.ciphertext) {
              base64 = encRes.ciphertext;
            }
          } catch {}
        }
        const up = await apiFast.ipfsUpload(base64);
        if (up.ok && up.cid) {
          const ipfs_uri = `http://localhost:8182/ipfs/get?cid=${up.cid}`;
          const form = new FormData();
          form.append('action', 'send_message');
          form.append('room_id', currentRoomId);
          form.append('type', 'attachment');
          form.append('ipfs_uri', ipfs_uri);
          form.append('filename', file.name);
          form.append('mime', file.type || 'application/octet-stream');
          form.append('author', 'usuario');
          if (encryptToggleEl && encryptToggleEl.checked) form.append('enc', 'true');
          const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
          const json = await res.json();
          if (json.ok) {
            fileInputEl.value = '';
            // La publicación en tiempo real la realiza el servidor PHP (publishToFastApi)
          } else {
            const err = json.error || 'error';
            if (err === 'muted') showToast('No puedes enviar: estás muteado en esta sala', 'error');
            else if (err === 'role_denied') showToast('Tu rol (viewer) no permite enviar', 'error');
            else if (err === 'no_member') showToast('Debes ser miembro para enviar en esta sala', 'error');
            else showToast('No se pudo enviar el adjunto', 'error');
          }
        }
      } catch (e) {
        console.error('Error subiendo adjunto', e);
      }
    }

    // Si hay texto, enviar mensaje de texto (cifrado si toggle activo)
    if (text) {
      let payload = text;
      let enc = false;
      if (encryptToggleEl && encryptToggleEl.checked) {
        try {
          const e = await apiFast.encrypt(text, currentRoomId);
          if (e.ok && e.enc && e.ciphertext) {
            payload = e.ciphertext;
            enc = true;
          }
        } catch {}
      }
      const form = new FormData();
      form.append('action', 'send_message');
      form.append('room_id', currentRoomId);
      form.append('text', payload);
      form.append('author', 'usuario');
      if (enc) form.append('enc', 'true');
      const res = await fetch('/api.php', { method: 'POST', body: form, headers: sessionToken ? { 'X-Session-Token': sessionToken } : {} });
      const json = await res.json();
      if (json.ok) {
        messageInputEl.value = '';
        // La publicación en tiempo real la realiza el servidor PHP (publishToFastApi)
      } else {
        const err = json.error || 'error';
        if (err === 'muted') showToast('No puedes enviar: estás muteado en esta sala', 'error');
        else if (err === 'role_denied') showToast('Tu rol (viewer) no permite enviar', 'error');
        else if (err === 'no_member') showToast('Debes ser miembro para enviar en esta sala', 'error');
        else showToast('No se pudo enviar el mensaje', 'error');
      }
    }
    await loadMessages();
  };
  messageInputEl.addEventListener('keydown', async (e) => {
    // Enviar con Enter o Ctrl+Enter; permitir Shift+Enter para nueva línea
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendBtnEl.click();
    }
  });

  async function initRooms() {
    const res = await api.getRooms();
    if (res.ok) renderRooms(res.rooms);
  }

  async function initStatus() {
    try {
      const res = await api.openagiStatus();
      if (res.ok) {
        const st = res.status;
        openagiStatusEl.textContent = `Estado: ${st.state ?? 'N/A'}`;
        // métricas básicas (si FastAPI activo)
        try {
          const mres = await fetch('http://localhost:8182/stats/chat');
          if (mres.ok) {
            const mjson = await mres.json();
            if (mjson.ok && mjson.stats) {
              const s = mjson.stats;
              openagiStatusEl.textContent += ` | Salas: ${s.rooms} | Msg: ${s.messages} | Enc: ${s.encrypted_messages} | Adj: ${s.attachments}`;
            }
          }
        } catch {}
      } else {
        openagiStatusEl.textContent = 'Estado: N/A';
      }
    } catch (e) {
      openagiStatusEl.textContent = 'Estado: N/A';
    }
  }

  // Init
  initRooms();
  initStatus();
  // refrescar estado y métricas periódicamente
  setInterval(initStatus, 10000);
  if (userAddress && userBadgeEl) userBadgeEl.textContent = `Usuario: ${userAddress}`;
  if (btnLoginEl) btnLoginEl.onclick = connectWalletLogin;
  if (btnLogoutEl) btnLogoutEl.onclick = logout;
  // visibilidad botones login/logout
  if (sessionToken) {
    if (btnLoginEl) btnLoginEl.style.display = 'none';
    if (btnLogoutEl) btnLogoutEl.style.display = '';
  } else {
    if (btnLogoutEl) btnLogoutEl.style.display = 'none';
  }
})();