// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title UserRegistry - Registro de claves públicas de cifrado para chat E2E
contract UserRegistry {
    event UserKeyUpdated(address indexed user, string publicKey);

    mapping(address => string) public userPublicKey;

    /// @notice Registra o actualiza la clave pública de cifrado del usuario
    /// @param publicKey Clave pública (p.ej., base64 de X25519)
    function setPublicKey(string calldata publicKey) external {
        userPublicKey[msg.sender] = publicKey;
        emit UserKeyUpdated(msg.sender, publicKey);
    }

    /// @notice Obtiene la clave pública de un usuario
    function getPublicKey(address user) external view returns (string memory) {
        return userPublicKey[user];
    }
}