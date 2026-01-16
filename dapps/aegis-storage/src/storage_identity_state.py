import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from aegis_core.crypto_framework import PublicNodeIdentity


@dataclass
class RoleInfo:
    role: str
    reputation_score: float = 0.5
    attributes: Dict[str, bool] = field(default_factory=dict)


@dataclass
class MultisigPolicy:
    policy_id: str
    operation_type: str
    required_signers: List[str]
    threshold_m: int


@dataclass
class PendingMultisigOp:
    op_id: str
    op_payload: Dict[str, Any]
    policy_id: str
    approvals: Dict[str, str] = field(default_factory=dict)
    executed: bool = False


class StorageIdentityState:
    def __init__(self):
        self.roles: Dict[str, RoleInfo] = {}
        self.auditors: Set[str] = set()
        self.policies: Dict[str, MultisigPolicy] = {}
        self.pending_ops: Dict[str, PendingMultisigOp] = {}

    def apply_change(self, change_data: Dict[str, Any]) -> bool:
        change_type = change_data.get("type")
        if change_type != "aegis_storage_identity":
            return False

        action = change_data.get("action")
        if action == "grant_role":
            return self._apply_grant_role(change_data)
        if action == "create_multisig_policy":
            return self._apply_create_policy(change_data)
        if action == "multisig_bundle":
            return self._apply_multisig_bundle(change_data)
        return False

    def _apply_grant_role(self, change_data: Dict[str, Any]) -> bool:
        node_id = change_data.get("node_id")
        role = change_data.get("role")
        if not node_id or not role:
            return False

        info = self.roles.get(node_id) or RoleInfo(role=role)
        info.role = role
        self.roles[node_id] = info
        if role == "DataAuditor":
            self.auditors.add(node_id)
        return True

    def _apply_create_policy(self, change_data: Dict[str, Any]) -> bool:
        policy_id = change_data.get("policy_id")
        operation_type = change_data.get("operation_type")
        required_signers = change_data.get("required_signers") or []
        threshold_m = int(change_data.get("threshold_m") or max(1, len(required_signers)))
        if not policy_id or not operation_type or not isinstance(required_signers, list):
            return False

        self.policies[policy_id] = MultisigPolicy(
            policy_id=policy_id,
            operation_type=operation_type,
            required_signers=required_signers,
            threshold_m=threshold_m,
        )
        return True

    def _apply_multisig_bundle(self, change_data: Dict[str, Any]) -> bool:
        op = change_data.get("op")
        approvals = change_data.get("approvals") or []
        if not isinstance(op, dict) or not isinstance(approvals, list):
            return False

        op_id = op.get("op_id")
        policy_id = op.get("policy_id")
        if not op_id or not policy_id:
            return False

        policy = self.policies.get(policy_id)
        if policy is None:
            return False

        pending = self.pending_ops.get(op_id) or PendingMultisigOp(
            op_id=op_id,
            op_payload=op,
            policy_id=policy_id,
        )
        if pending.executed:
            return True

        for a in approvals:
            signer = a.get("signer_node_id")
            sig = a.get("signature_b64")
            if signer in policy.required_signers and signer not in pending.approvals:
                pending.approvals[signer] = sig

        self.pending_ops[op_id] = pending

        if len(pending.approvals) >= policy.threshold_m:
            return self._execute_op(pending)
        return True

    def _execute_op(self, pending: PendingMultisigOp) -> bool:
        action = pending.op_payload.get("action")
        if action != "attribute_change":
            return False

        target_node_id = pending.op_payload.get("target_node_id")
        attribute = pending.op_payload.get("attribute")
        value = bool(pending.op_payload.get("value"))
        if not target_node_id or not attribute:
            return False

        info = self.roles.get(target_node_id) or RoleInfo(role="DataOwner")
        info.attributes[attribute] = value
        self.roles[target_node_id] = info
        pending.executed = True
        return True

