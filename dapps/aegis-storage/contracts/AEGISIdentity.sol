// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract AEGISIdentity {
    mapping(bytes32 => address) public aegisIdToEvm;
    mapping(address => bytes32) public evmToAegisId;

    event IdentityLinked(bytes32 indexed aegisId, address indexed evmAddress);

    function linkIdentity(bytes32 aegisId) external {
        require(aegisId != bytes32(0), "AEGIS: invalid id");
        require(evmToAegisId[msg.sender] == bytes32(0), "AEGIS: already linked");
        require(aegisIdToEvm[aegisId] == address(0), "AEGIS: id already linked");
        evmToAegisId[msg.sender] = aegisId;
        aegisIdToEvm[aegisId] = msg.sender;
        emit IdentityLinked(aegisId, msg.sender);
    }
}

