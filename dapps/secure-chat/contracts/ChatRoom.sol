// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ChatRoom - Metadatos de salas y eventos on-chain para mensajes/archivos cifrados en IPFS
contract ChatRoom {
    struct Room {
        address admin;
        uint256 id;
        bool exists;
        mapping(address => bool) participants;
    }

    uint256 public nextRoomId;
    mapping(uint256 => Room) private rooms;

    event RoomCreated(uint256 indexed roomId, address indexed admin, address[] participants);
    event MessagePosted(uint256 indexed roomId, address indexed sender, string cid, bytes32 contentHash);
    event FileShared(
        uint256 indexed roomId,
        address indexed sender,
        string cid,
        bytes32 contentHash,
        string filename,
        uint256 size
    );

    function createRoom(address[] calldata participants) external returns (uint256) {
        uint256 id = ++nextRoomId;
        Room storage r = rooms[id];
        r.admin = msg.sender;
        r.id = id;
        r.exists = true;
        r.participants[msg.sender] = true;
        for (uint256 i = 0; i < participants.length; i++) {
            r.participants[participants[i]] = true;
        }
        emit RoomCreated(id, msg.sender, participants);
        return id;
    }

    modifier onlyParticipant(uint256 roomId) {
        require(rooms[roomId].exists, "room-not-found");
        require(rooms[roomId].participants[msg.sender], "not-participant");
        _;
    }

    function addParticipant(uint256 roomId, address user) external {
        Room storage r = rooms[roomId];
        require(r.exists, "room-not-found");
        require(r.admin == msg.sender, "not-admin");
        r.participants[user] = true;
    }

    function postMessage(uint256 roomId, string calldata cid, bytes32 contentHash) external onlyParticipant(roomId) {
        emit MessagePosted(roomId, msg.sender, cid, contentHash);
    }

    function shareFile(
        uint256 roomId,
        string calldata cid,
        bytes32 contentHash,
        string calldata filename,
        uint256 size
    ) external onlyParticipant(roomId) {
        emit FileShared(roomId, msg.sender, cid, contentHash, filename, size);
    }

    function isParticipant(uint256 roomId, address user) external view returns (bool) {
        Room storage r = rooms[roomId];
        if (!r.exists) return false;
        return r.participants[user];
    }
}