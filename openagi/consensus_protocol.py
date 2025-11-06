#!/usr/bin/env python3
"""
Consensus Protocol Wrapper for Harmonic Validation
Integrates harmonic validation into the consensus flow
"""

import logging
from openagi.harmonic_validation import compute_coherence_score
from openagi.token_rules import validate_harmonic_tx

log = logging.getLogger("consensus")


def pre_prepare_block(block, config):
    """
    Runs before block is proposed to validators.
    Adds harmonic proof if needed.
    
    Args:
        block: Block object to be validated
        config: Configuration dictionary with thresholds
        
    Returns:
        block: Validated block with harmonic proofs
    """
    if not getattr(block, "transactions", None):
        return block

    harmonic_txs = [tx for tx in block.transactions if tx.get("type") == "harmonic"]
    if not harmonic_txs:
        return block

    for tx in harmonic_txs:
        local_snapshot = tx.get("local_snapshot")
        bundle = tx.get("snapshot_bundle", [])
        cs = compute_coherence_score(local_snapshot, bundle)
        tx["aggregated_cs"] = cs
        
        if not validate_harmonic_tx(tx, config):
            raise Exception(f"Rejected harmonic tx {tx['id']} – coherence {cs:.3f} below threshold")

        log.info(f"Harmonic tx {tx['id']} coherence OK: {cs:.3f}")
        
    return block