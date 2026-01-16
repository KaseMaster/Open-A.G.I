// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IStorageIdentity {
    function isAuditor(bytes32 aegisId) external view returns (bool);
}

contract AegisStorageLedger {
    bytes32 public constant OP_FILE_UPLOAD = keccak256("FILE_UPLOAD");
    bytes32 public constant OP_FRAGMENT_LOCATION = keccak256("FRAGMENT_LOCATION");
    bytes32 public constant OP_ACCESS_GRANT = keccak256("ACCESS_GRANT");
    bytes32 public constant OP_INTEGRITY_CHALLENGE = keccak256("INTEGRITY_CHALLENGE");
    bytes32 public constant OP_INTEGRITY_RESULT = keccak256("INTEGRITY_RESULT");

    address public pbftExecutor;
    IStorageIdentity public identity;

    struct FileUpload {
        bytes32 fileId;
        bytes32 ownerAegisId;
        bytes32 fileRootHash;
        uint64 fileSize;
        uint64 fragmentCount;
        bytes32 metadataHash;
        uint64 createdAt;
        bytes32 immutableHash;
    }

    struct FileFragmentLocation {
        bytes32 fileId;
        bytes32 fragmentHash;
        bytes32 storageNodeAegisId;
        uint64 recordedAt;
        bytes32 immutableHash;
    }

    struct AccessGrant {
        bytes32 fileId;
        bytes32 ownerAegisId;
        bytes32 granteeAegisId;
        uint32 permissionsMask;
        uint64 expiresAt;
        uint64 grantedAt;
        bytes32 immutableHash;
    }

    enum ChallengeStatus {
        OPEN,
        RESOLVED
    }

    struct IntegrityChallenge {
        bytes32 challengeId;
        bytes32 fileId;
        bytes32 fragmentHash;
        bytes32 auditorAegisId;
        bytes32 storageNodeAegisId;
        bytes32 nonce;
        uint64 createdAt;
        ChallengeStatus status;
        bool success;
        bytes32 responseHash;
        uint64 resolvedAt;
        bytes32 immutableHash;
    }

    mapping(bytes32 => bool) public finalizedOps;
    mapping(bytes32 => FileUpload) public files;
    mapping(bytes32 => bytes32[]) public fileFragmentHashes;

    mapping(bytes32 => mapping(bytes32 => bool)) public fragmentHasLocation;
    mapping(bytes32 => bytes32[]) public fragmentLocations;

    mapping(bytes32 => mapping(bytes32 => AccessGrant)) public accessGrants;
    mapping(bytes32 => mapping(bytes32 => bool)) public hasAccessGrant;

    mapping(bytes32 => IntegrityChallenge) public challenges;

    event ConsensusRequired(bytes32 indexed operationHash, bytes32 indexed operationType, bytes32 indexed subject);
    event OperationFinalized(bytes32 indexed operationHash, bytes32 indexed operationType);

    event FileRecorded(bytes32 indexed fileId, bytes32 indexed ownerAegisId, bytes32 fileRootHash, uint64 fragmentCount);
    event FragmentLocationRecorded(bytes32 indexed fileId, bytes32 indexed fragmentHash, bytes32 indexed storageNodeAegisId);
    event AccessGranted(bytes32 indexed fileId, bytes32 indexed granteeAegisId, uint32 permissionsMask, uint64 expiresAt);
    event IntegrityChallengeRecorded(bytes32 indexed challengeId, bytes32 indexed fragmentHash, bytes32 indexed storageNodeAegisId);
    event IntegrityResultRecorded(bytes32 indexed challengeId, bool success, bytes32 responseHash);

    modifier onlyExecutor() {
        require(msg.sender == pbftExecutor, "AEGIS: executor only");
        _;
    }

    constructor(address identityContract, address executor) {
        identity = IStorageIdentity(identityContract);
        pbftExecutor = executor;
    }

    function setExecutor(address executor) external onlyExecutor {
        pbftExecutor = executor;
    }

    function _opHash(bytes32 opType, bytes memory opData) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(opType, keccak256(opData)));
    }

    function requestFileUpload(
        bytes32 fileId,
        bytes32 ownerAegisId,
        bytes32 fileRootHash,
        uint64 fileSize,
        bytes32 metadataHash,
        bytes32[] calldata fragmentHashes
    ) external returns (bytes32 operationHash) {
        bytes memory opData = abi.encode(fileId, ownerAegisId, fileRootHash, fileSize, metadataHash, fragmentHashes);
        operationHash = _opHash(OP_FILE_UPLOAD, opData);
        emit ConsensusRequired(operationHash, OP_FILE_UPLOAD, fileId);
    }

    function finalizeFileUpload(
        bytes32 operationHash,
        bytes32 fileId,
        bytes32 ownerAegisId,
        bytes32 fileRootHash,
        uint64 fileSize,
        bytes32 metadataHash,
        bytes32[] calldata fragmentHashes
    ) external onlyExecutor {
        require(!finalizedOps[operationHash], "AEGIS: op finalized");
        require(files[fileId].fileId == bytes32(0), "AEGIS: file exists");

        bytes memory opData = abi.encode(fileId, ownerAegisId, fileRootHash, fileSize, metadataHash, fragmentHashes);
        require(operationHash == _opHash(OP_FILE_UPLOAD, opData), "AEGIS: op hash mismatch");

        bytes32 immutableHash = keccak256(opData);
        files[fileId] = FileUpload({
            fileId: fileId,
            ownerAegisId: ownerAegisId,
            fileRootHash: fileRootHash,
            fileSize: fileSize,
            fragmentCount: uint64(fragmentHashes.length),
            metadataHash: metadataHash,
            createdAt: uint64(block.timestamp),
            immutableHash: immutableHash
        });

        for (uint256 i = 0; i < fragmentHashes.length; i++) {
            fileFragmentHashes[fileId].push(fragmentHashes[i]);
        }

        finalizedOps[operationHash] = true;
        emit OperationFinalized(operationHash, OP_FILE_UPLOAD);
        emit FileRecorded(fileId, ownerAegisId, fileRootHash, uint64(fragmentHashes.length));
    }

    function requestFragmentLocation(
        bytes32 fileId,
        bytes32 fragmentHash,
        bytes32 storageNodeAegisId
    ) external returns (bytes32 operationHash) {
        bytes memory opData = abi.encode(fileId, fragmentHash, storageNodeAegisId);
        operationHash = _opHash(OP_FRAGMENT_LOCATION, opData);
        emit ConsensusRequired(operationHash, OP_FRAGMENT_LOCATION, fragmentHash);
    }

    function finalizeFragmentLocation(
        bytes32 operationHash,
        bytes32 fileId,
        bytes32 fragmentHash,
        bytes32 storageNodeAegisId
    ) external onlyExecutor {
        require(!finalizedOps[operationHash], "AEGIS: op finalized");
        require(files[fileId].fileId != bytes32(0), "AEGIS: file missing");

        bytes memory opData = abi.encode(fileId, fragmentHash, storageNodeAegisId);
        require(operationHash == _opHash(OP_FRAGMENT_LOCATION, opData), "AEGIS: op hash mismatch");

        if (!fragmentHasLocation[fragmentHash][storageNodeAegisId]) {
            fragmentHasLocation[fragmentHash][storageNodeAegisId] = true;
            fragmentLocations[fragmentHash].push(storageNodeAegisId);
        }

        finalizedOps[operationHash] = true;
        emit OperationFinalized(operationHash, OP_FRAGMENT_LOCATION);
        emit FragmentLocationRecorded(fileId, fragmentHash, storageNodeAegisId);
    }

    function requestAccessGrant(
        bytes32 fileId,
        bytes32 ownerAegisId,
        bytes32 granteeAegisId,
        uint32 permissionsMask,
        uint64 expiresAt
    ) external returns (bytes32 operationHash) {
        bytes memory opData = abi.encode(fileId, ownerAegisId, granteeAegisId, permissionsMask, expiresAt);
        operationHash = _opHash(OP_ACCESS_GRANT, opData);
        emit ConsensusRequired(operationHash, OP_ACCESS_GRANT, fileId);
    }

    function finalizeAccessGrant(
        bytes32 operationHash,
        bytes32 fileId,
        bytes32 ownerAegisId,
        bytes32 granteeAegisId,
        uint32 permissionsMask,
        uint64 expiresAt
    ) external onlyExecutor {
        require(!finalizedOps[operationHash], "AEGIS: op finalized");
        require(files[fileId].fileId != bytes32(0), "AEGIS: file missing");
        require(!hasAccessGrant[fileId][granteeAegisId], "AEGIS: grant immutable");

        bytes memory opData = abi.encode(fileId, ownerAegisId, granteeAegisId, permissionsMask, expiresAt);
        require(operationHash == _opHash(OP_ACCESS_GRANT, opData), "AEGIS: op hash mismatch");

        bytes32 immutableHash = keccak256(opData);
        accessGrants[fileId][granteeAegisId] = AccessGrant({
            fileId: fileId,
            ownerAegisId: ownerAegisId,
            granteeAegisId: granteeAegisId,
            permissionsMask: permissionsMask,
            expiresAt: expiresAt,
            grantedAt: uint64(block.timestamp),
            immutableHash: immutableHash
        });
        hasAccessGrant[fileId][granteeAegisId] = true;

        finalizedOps[operationHash] = true;
        emit OperationFinalized(operationHash, OP_ACCESS_GRANT);
        emit AccessGranted(fileId, granteeAegisId, permissionsMask, expiresAt);
    }

    function requestIntegrityChallenge(
        bytes32 fileId,
        bytes32 fragmentHash,
        bytes32 auditorAegisId,
        bytes32 storageNodeAegisId,
        bytes32 nonce
    ) external returns (bytes32 operationHash, bytes32 challengeId) {
        if (address(identity) != address(0)) {
            require(identity.isAuditor(auditorAegisId), "AEGIS: auditor only");
        }
        challengeId = keccak256(abi.encodePacked(fileId, fragmentHash, auditorAegisId, storageNodeAegisId, nonce));
        bytes memory opData = abi.encode(challengeId, fileId, fragmentHash, auditorAegisId, storageNodeAegisId, nonce);
        operationHash = _opHash(OP_INTEGRITY_CHALLENGE, opData);
        emit ConsensusRequired(operationHash, OP_INTEGRITY_CHALLENGE, fragmentHash);
    }

    function finalizeIntegrityChallenge(
        bytes32 operationHash,
        bytes32 challengeId,
        bytes32 fileId,
        bytes32 fragmentHash,
        bytes32 auditorAegisId,
        bytes32 storageNodeAegisId,
        bytes32 nonce
    ) external onlyExecutor {
        require(!finalizedOps[operationHash], "AEGIS: op finalized");
        require(challenges[challengeId].challengeId == bytes32(0), "AEGIS: challenge exists");

        bytes memory opData = abi.encode(challengeId, fileId, fragmentHash, auditorAegisId, storageNodeAegisId, nonce);
        require(operationHash == _opHash(OP_INTEGRITY_CHALLENGE, opData), "AEGIS: op hash mismatch");

        challenges[challengeId] = IntegrityChallenge({
            challengeId: challengeId,
            fileId: fileId,
            fragmentHash: fragmentHash,
            auditorAegisId: auditorAegisId,
            storageNodeAegisId: storageNodeAegisId,
            nonce: nonce,
            createdAt: uint64(block.timestamp),
            status: ChallengeStatus.OPEN,
            success: false,
            responseHash: bytes32(0),
            resolvedAt: 0,
            immutableHash: keccak256(opData)
        });

        finalizedOps[operationHash] = true;
        emit OperationFinalized(operationHash, OP_INTEGRITY_CHALLENGE);
        emit IntegrityChallengeRecorded(challengeId, fragmentHash, storageNodeAegisId);
    }

    function requestIntegrityResult(
        bytes32 challengeId,
        bool success,
        bytes32 responseHash
    ) external returns (bytes32 operationHash) {
        require(challenges[challengeId].challengeId != bytes32(0), "AEGIS: missing challenge");
        bytes memory opData = abi.encode(challengeId, success, responseHash);
        operationHash = _opHash(OP_INTEGRITY_RESULT, opData);
        emit ConsensusRequired(operationHash, OP_INTEGRITY_RESULT, challengeId);
    }

    function finalizeIntegrityResult(
        bytes32 operationHash,
        bytes32 challengeId,
        bool success,
        bytes32 responseHash
    ) external onlyExecutor {
        require(!finalizedOps[operationHash], "AEGIS: op finalized");
        IntegrityChallenge storage c = challenges[challengeId];
        require(c.challengeId != bytes32(0), "AEGIS: missing challenge");
        require(c.status == ChallengeStatus.OPEN, "AEGIS: already resolved");

        bytes memory opData = abi.encode(challengeId, success, responseHash);
        require(operationHash == _opHash(OP_INTEGRITY_RESULT, opData), "AEGIS: op hash mismatch");

        c.status = ChallengeStatus.RESOLVED;
        c.success = success;
        c.responseHash = responseHash;
        c.resolvedAt = uint64(block.timestamp);

        finalizedOps[operationHash] = true;
        emit OperationFinalized(operationHash, OP_INTEGRITY_RESULT);
        emit IntegrityResultRecorded(challengeId, success, responseHash);
    }
}

