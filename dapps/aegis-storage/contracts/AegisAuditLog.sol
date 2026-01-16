// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract AegisAuditLog {
    address public pbftExecutor;

    struct AuditEntry {
        bytes32 leafHash;
        bytes32 actorAegisId;
        bytes32 eventType;
        bytes32 status;
        uint64 timestamp;
        bytes32 immutableHash;
    }

    mapping(uint256 => AuditEntry) public entries;
    uint256 public entryCount;

    mapping(bytes32 => bool) public finalizedOps;

    event ConsensusRequired(bytes32 indexed operationHash, bytes32 indexed operationType, uint256 indexed index);
    event OperationFinalized(bytes32 indexed operationHash, bytes32 indexed operationType);
    event AuditEntryCommitted(uint256 indexed index, bytes32 leafHash);

    bytes32 public constant OP_COMMIT_AUDIT_ENTRY = keccak256("COMMIT_AUDIT_ENTRY");

    modifier onlyExecutor() {
        require(msg.sender == pbftExecutor, "AEGIS: executor only");
        _;
    }

    constructor(address executor) {
        pbftExecutor = executor;
    }

    function setExecutor(address executor) external onlyExecutor {
        pbftExecutor = executor;
    }

    function _opHash(bytes32 opType, bytes memory opData) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(opType, keccak256(opData)));
    }

    function requestCommitAuditEntry(
        uint256 index,
        bytes32 leafHash,
        bytes32 actorAegisId,
        bytes32 eventType,
        bytes32 status,
        uint64 timestamp
    ) external returns (bytes32 operationHash) {
        bytes memory opData = abi.encode(index, leafHash, actorAegisId, eventType, status, timestamp);
        operationHash = _opHash(OP_COMMIT_AUDIT_ENTRY, opData);
        emit ConsensusRequired(operationHash, OP_COMMIT_AUDIT_ENTRY, index);
    }

    function finalizeCommitAuditEntry(
        bytes32 operationHash,
        uint256 index,
        bytes32 leafHash,
        bytes32 actorAegisId,
        bytes32 eventType,
        bytes32 status,
        uint64 timestamp
    ) external onlyExecutor {
        require(!finalizedOps[operationHash], "AEGIS: op finalized");
        require(index == entryCount, "AEGIS: bad index");
        bytes memory opData = abi.encode(index, leafHash, actorAegisId, eventType, status, timestamp);
        require(operationHash == _opHash(OP_COMMIT_AUDIT_ENTRY, opData), "AEGIS: op hash mismatch");

        entries[index] = AuditEntry({
            leafHash: leafHash,
            actorAegisId: actorAegisId,
            eventType: eventType,
            status: status,
            timestamp: timestamp,
            immutableHash: keccak256(opData)
        });
        entryCount += 1;
        finalizedOps[operationHash] = true;
        emit OperationFinalized(operationHash, OP_COMMIT_AUDIT_ENTRY);
        emit AuditEntryCommitted(index, leafHash);
    }
}

