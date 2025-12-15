# Changelog

## [Unreleased] - 2025-12-15

### Security

- **P2P Network**: Enforced `reputation_manager` check in `ConnectionManager` handshake.
- **P2P Network**: Mandated `crypto_engine` in `ConnectionManager` initialization to prevent plaintext messaging.
- **Crypto Framework**: Refactored `encrypt_message` and `decrypt_message` for better auditability and forward secrecy.
- **CI/CD**: Pinned all GitHub Actions to specific commit SHAs to prevent supply chain attacks.

### Fixed

- **Heartbeat**: Fixed duplicate recovery tasks by merging nested conditions and adding `_recover_node` method.
- **P2P**: Fixed missing reputation validation during peer auto-connection and handshake.
- **Crypto**: Removed redundant f-strings and cleaned up code.

### Documentation

- Consolidated multiple summary files into this Changelog.

---
*Consolidated from:*

- COSMIC_ASCENSION_VERIFICATION_SUMMARY.md
- QUANTUM_CURRENCY_V0.3.0_INTEGRATION_SUMMARY.md
- V0.4.0_DOCUMENTATION_ORGANIZATION_SUMMARY.md
- QUANTUM_CURRENCY_TESTING_SUMMARY.md
- QUANTUM_CURRENCY_IMPLEMENTATION_SUMMARY.md
- SUMMARY_OF_CHANGES.md
- QUANTUM_CURRENCY_SUMMARY.md
- PHASE_4_COMPLETION_SUMMARY.md
- PHASES_COMPLETED_SUMMARY.md
- TOKEN_INTEGRATION_SUMMARY.md
- DEPLOYMENT_SUMMARY.md
- HMN_ENHANCEMENT_SUMMARY.md
- HMN_PRODUCTION_DEPLOYMENT_SUMMARY.md
- QUANTUM_CURRENCY_INTEGRATION_SUMMARY.md
- GLOBAL_CURVATURE_RESONANCE_DEPLOYMENT_SUMMARY.md
- FOUR_KEY_AREAS_IMPLEMENTATION_SUMMARY.md
- FINAL_HMN_ENHANCEMENT_SUMMARY.md
- FIELD_GRAVITATION_IMPLEMENTATION_SUMMARY.md
- HARMONIC_MESH_NETWORK_SUMMARY.md
- MASS_EMERGENCE_IMPLEMENTATION_SUMMARY.md
- PROJECT_COMPLETION_SUMMARY.md
- PHASE_6_7_IMPLEMENTATION_SUMMARY.md
- FINAL_HMN_VERIFICATION_SUMMARY.md
- TOKEN_COHERENCE_INTEGRATION_SUMMARY.md
