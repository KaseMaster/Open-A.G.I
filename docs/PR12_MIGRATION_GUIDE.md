# Migration Guide: PR #12 Security Enhancements

**Date**: December 15, 2025
**Affected Version**: 2.0.0+

This guide details the changes introduced to restore and enforce strict security policies in the Open-A.G.I framework.

## Breaking Changes

### 1. Mandatory CryptoEngine

The `ConnectionManager` now **requires** a valid `CryptoEngine` instance during initialization. Initialization with `crypto_engine=None` will raise a `ValueError`.

**Before:**

```python
manager = ConnectionManager(node_id="node1", port=8080, crypto_engine=None) # Allowed (Insecure)
```

**After:**

```python
# Must provide engine or initialization fails
manager = ConnectionManager(node_id="node1", port=8080, crypto_engine=my_engine)
```

### 2. Fail-Closed Message Sending

The `send_message` method no longer supports plaintext fallback. All messages sent to peers MUST be encrypted using an established secure channel.

- If the peer does not have an active Ratchet state: `RuntimeError`.
- If encryption fails: `RuntimeError`.

**Action Required**: Ensure all peers complete the secure handshake before attempting to send `data` or `consensus` messages.

### 3. Identity Verification

Peer connections are now gated by:

- **Cryptographic Signature**: mDNS and Bootstrap announcements must be signed.
- **Reputation**: Peers with low reputation scores are automatically rejected during handshake.

## Configuration Updates

Update your node configuration to ensure `security_level` is set to `high` (default).

```python
crypto_config = CryptoConfig(
    security_level=SecurityLevel.HIGH, # Enforced
    key_rotation_interval=3600
)
```

## Troubleshooting

### "ValueError: crypto_engine is mandatory"

Ensure you are initializing `P2PNetworkManager` or `ConnectionManager` with a valid `CryptoEngine`. Check if `crypto_framework` dependencies are installed.

### "RuntimeError: Insecure channel rejected"

This occurs if you attempt to send a message before the secure handshake completes. Wait for the `Handshake completado` log or check connection status.
