import React from 'react';

export default function StatusBar({ ipfsOk, chainOk, account, aegisBalance }) {
  return (
    <div className="status-bar">
      <div className="status-item">
        <span className={`dot ${ipfsOk ? 'ok' : 'bad'}`}></span>
        <span>IPFS {ipfsOk ? 'conectado' : 'sin conexión'}</span>
      </div>
      <div className="status-item">
        <span className={`dot ${chainOk ? 'ok' : 'bad'}`}></span>
        <span>Red {chainOk ? 'localhost:8545' : 'no conectada'}</span>
      </div>
      <div className="status-spacer" />
      <div className="status-item">
        <span className="label">Cuenta:</span>
        <span className="mono">{account ? account : '—'}</span>
      </div>
      <div className="status-item">
        <span className="label">AEGIS:</span>
        <span className="mono">{aegisBalance}</span>
      </div>
    </div>
  );
}