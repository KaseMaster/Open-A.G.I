// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./AEGISIdentity.sol";

contract StorageIdentity is AEGISIdentity {
    enum Role {
        NONE,
        DATA_OWNER,
        DATA_AUDITOR,
        STORAGE_NODE
    }

    struct DataOwner {
        bytes32 aegisId;
        uint256 reputationScore;
    }

    struct DataAuditor {
        bytes32 aegisId;
        uint256 reputationScore;
    }

    struct StorageNode {
        bytes32 aegisId;
        uint256 reputationScore;
    }

    struct IdentityInfo {
        Role role;
        uint256 reputationScore;
        bool exists;
    }

    struct AttributePolicy {
        uint256 thresholdM;
        bool exists;
    }

    struct ApprovalState {
        uint256 approvals;
        bool executed;
    }

    mapping(bytes32 => IdentityInfo) public identities;
    mapping(bytes32 => bool) public isAuditor;
    uint256 public auditorCount;

    mapping(bytes32 => mapping(bytes32 => bool)) public attributes;
    mapping(bytes32 => AttributePolicy) public attributePolicies;

    mapping(bytes32 => mapping(bytes32 => bool)) public opApprovedBy;
    mapping(bytes32 => ApprovalState) public opState;

    event RoleGranted(bytes32 indexed aegisId, Role role);
    event AttributePolicySet(bytes32 indexed attributeKey, uint256 thresholdM);
    event AttributeChanged(bytes32 indexed targetAegisId, bytes32 indexed attributeKey, bool value, bytes32 opId);

    function _grantRole(bytes32 aegisId, Role role) internal {
        require(aegisId != bytes32(0), "AEGIS: invalid id");
        IdentityInfo storage info = identities[aegisId];
        info.role = role;
        info.exists = true;
        if (info.reputationScore == 0) {
            info.reputationScore = 50;
        }
        if (role == Role.DATA_AUDITOR) {
            if (!isAuditor[aegisId]) {
                isAuditor[aegisId] = true;
                auditorCount += 1;
            }
        }
        emit RoleGranted(aegisId, role);
    }

    function grantRoleToSelf(Role role) external {
        bytes32 aegisId = evmToAegisId[msg.sender];
        require(aegisId != bytes32(0), "AEGIS: not linked");
        _grantRole(aegisId, role);
    }

    function setAttributePolicy(bytes32 attributeKey, uint256 thresholdM) external {
        bytes32 callerId = evmToAegisId[msg.sender];
        require(isAuditor[callerId], "AEGIS: auditor only");
        require(attributeKey != bytes32(0), "AEGIS: invalid attribute");
        require(thresholdM > 0, "AEGIS: invalid threshold");
        attributePolicies[attributeKey] = AttributePolicy({thresholdM: thresholdM, exists: true});
        emit AttributePolicySet(attributeKey, thresholdM);
    }

    function approveAttributeChange(bytes32 targetAegisId, bytes32 attributeKey, bool value) external {
        bytes32 auditorId = evmToAegisId[msg.sender];
        require(isAuditor[auditorId], "AEGIS: auditor only");
        require(attributePolicies[attributeKey].exists, "AEGIS: no policy");
        require(targetAegisId != bytes32(0), "AEGIS: invalid target");

        bytes32 opId = keccak256(abi.encodePacked(targetAegisId, attributeKey, value));
        require(!opApprovedBy[opId][auditorId], "AEGIS: already approved");

        opApprovedBy[opId][auditorId] = true;
        ApprovalState storage s = opState[opId];
        require(!s.executed, "AEGIS: executed");
        s.approvals += 1;

        uint256 requiredM = attributePolicies[attributeKey].thresholdM;
        if (s.approvals >= requiredM) {
            attributes[targetAegisId][attributeKey] = value;
            s.executed = true;
            emit AttributeChanged(targetAegisId, attributeKey, value, opId);
        }
    }
}

