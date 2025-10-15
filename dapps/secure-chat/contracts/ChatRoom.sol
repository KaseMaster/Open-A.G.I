// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ChatRoom {
    struct Room {
        uint256 id;
        string name;
        string ipfsHash;
        address creator;
        uint256 memberCount;
    }

    struct Message {
        address sender;
        string ipfsHash;
        uint256 timestamp;
    }

    event RoomCreated(uint256 indexed roomId, string name, address creator);
    event MessageSent(uint256 indexed roomId, address sender, string ipfsHash);
    event MemberJoined(uint256 indexed roomId, address member);

    Room[] private rooms;
    mapping(uint256 => address[]) private roomMembers;
    mapping(uint256 => Message[]) private roomMessages;
    mapping(uint256 => mapping(address => bool)) private isMember;

    function createRoom(string memory name, string memory ipfsHash) public {
        uint256 id = rooms.length;
        rooms.push(Room({
            id: id,
            name: name,
            ipfsHash: ipfsHash,
            creator: msg.sender,
            memberCount: 1
        }));

        roomMembers[id].push(msg.sender);
        isMember[id][msg.sender] = true;

        emit RoomCreated(id, name, msg.sender);
    }

    function joinRoom(uint256 roomId) public {
        require(roomId < rooms.length, "Invalid room");
        if (!isMember[roomId][msg.sender]) {
            roomMembers[roomId].push(msg.sender);
            isMember[roomId][msg.sender] = true;
            rooms[roomId].memberCount += 1;
            emit MemberJoined(roomId, msg.sender);
        }
    }

    function sendMessage(uint256 roomId, string memory ipfsHash) public {
        require(roomId < rooms.length, "Invalid room");
        // Allow non-members to send; or enforce membership:
        if (!isMember[roomId][msg.sender]) {
            joinRoom(roomId);
        }
        roomMessages[roomId].push(Message({
            sender: msg.sender,
            ipfsHash: ipfsHash,
            timestamp: block.timestamp
        }));
        emit MessageSent(roomId, msg.sender, ipfsHash);
    }

    function getRooms() public view returns (Room[] memory) {
        return rooms;
    }

    function getRoomMessages(uint256 roomId) public view returns (Message[] memory) {
        require(roomId < rooms.length, "Invalid room");
        return roomMessages[roomId];
    }

    function getRoomMembers(uint256 roomId) public view returns (address[] memory) {
        require(roomId < rooms.length, "Invalid room");
        return roomMembers[roomId];
    }
}