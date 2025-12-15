# Open-A.G.I Security Architecture

## Overview

This document outlines the security architecture of the Open-A.G.I distributed AI framework. The system is designed to provide secure, resilient, and private peer-to-peer communication for distributed learning and consensus.

## Threat Model

We assume a hostile network environment where:

- **Eavesdropping**: Attackers can monitor network traffic.
- **Tampering**: Message contents can be modified in transit.
- **Impersonation**: Rogue nodes may attempt to join the network as legitimate peers.
- **Sybil Attacks**: Attackers may flood the network with fake identities.

## Cryptographic Guarantees

### Post-Quantum Cryptography (PQC)

Open-A.G.I utilizes a hybrid cryptographic approach to ensure long-term security against quantum adversaries.

- **Identity**: Ed25519 for digital signatures (fast, secure, widely supported).
- **Key Exchange**: X25519 + Kyber (planned) for key encapsulation.
- **Encryption**: ChaCha20-Poly1305 for authenticated encryption.

### Double Ratchet Algorithm

All secure channels employ the Double Ratchet Algorithm to provide:

- **Perfect Forward Secrecy (PFS)**: Compromise of current keys does not compromise past messages.
- **Future Secrecy**: Compromise of current keys does not compromise future messages (self-healing).
- **Authentication**: Usage of shared secrets derived from authenticated key exchange.

## Authentication Flow

### 1. Discovery (mDNS/Bootstrap)

Peers announce their presence via mDNS (local) or Bootstrap nodes (WAN).

- **Identity Verification**: Discovery announcements include a cryptographic signature of the `node_id` to prove ownership.
- **Reputation Check**: `PeerReputationManager` validates discovered peers against local history before connection.

### 2. Connection Handshake

Upon TCP connection establishment:

1. **Identity Exchange**: Peers exchange public keys and identity proofs.
2. **Reputation Enforcement**: Connection is rejected if peer reputation is below threshold.
3. **Secure Channel Establishment**: A Double Ratchet session is initialized using the exchanged keys.

## Network Topology Security

- **Fail-Closed Design**: Messages MUST be encrypted. If encryption fails or the channel is not secure, transmission is aborted.
- **Isolation**: Suspicious nodes are isolated and added to a blocklist after a threshold of incidents (e.g., invalid signatures, spam).
- **Heartbeats**: Secure heartbeats prevent replay attacks and ensure liveliness.

## Implementation Details

- `CryptoEngine`: Handles all cryptographic primitives and ratchet state.
- `ConnectionManager`: Enforces security policies (mandatory crypto, fail-closed).
- `PeerReputationManager`: Tracks peer behavior and gates access.
