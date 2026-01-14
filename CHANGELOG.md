# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-01-14

### Added
- **Structure**: Reorganized project into a `src/aegis_core` package structure for better modularity.
- **Node.js Support**: Added `node-test` job to GitHub Actions CI pipeline.
- **Security**: Added `.github/dependabot.yml` for automated dependency updates.
- **Python**: Added `pyproject.toml` for modern dependency management (PEP 621).

### Changed
- **Dependencies**: Updated `package.json` dependencies for all DApps to stable versions.
- **Scripts**: Standardized npm scripts (`test`, `lint`, `build`) across all DApp subprojects.
- **CI/CD**: Expanded `ci.yml` to include Node.js validation alongside Python/Docker.
- **Refactor**: Moved core Python modules (`p2p_network`, `crypto_framework`, etc.) to `src/aegis_core`.
- **Refactor**: Centralized deployment scripts in `scripts/deployment/`.

### Fixed
- **Cleanup**: Removed redundant debug files and legacy artifacts from root.
- **Configuration**: Standardized `main.py` entry point to use new package structure.

## [2.0.0] - 2025-10-15

### Added
- Initial release of AEGIS Framework v2.
- Integration with TOR network for anonymous communication.
- P2P network layer with custom discovery protocol.
- PBFT + Proof of Computation consensus algorithm.
- Basic monitoring dashboard.
