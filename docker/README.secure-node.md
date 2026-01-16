# Secure Storage Node

## Build

`docker build -f docker/Dockerfile.secure-node -t aegis-storage-node:secure .`

## Run

`docker run --rm -p 8088:8088 -v /path/to/encrypted-volume:/data --security-opt no-new-privileges --cap-drop ALL aegis-storage-node:secure`

## AppArmor/Seccomp

- Seccomp: `--security-opt seccomp=docker/seccomp/secure-node.json`
- AppArmor: `--security-opt apparmor=secure-node`

