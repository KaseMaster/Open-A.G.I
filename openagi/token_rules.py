#!/usr/bin/env python3
"""
Token Rules Engine for FLX and CHR Tokens
Defines mint/burn/transfer rules based on harmonic coherence score (CS) and CHR reputation
"""

def validate_harmonic_tx(tx, config):
    """
    Validates harmonic transaction based on coherence score and CHR reputation
    
    Args:
        tx: dict containing {"aggregated_cs": float, "sender_chr": float, "type": "harmonic", ...}
        config: dict with thresholds, e.g. {"mint_threshold": 0.75, "min_chr": 0.6}
        
    Returns:
        bool: True if transaction is valid, False otherwise
    """
    cs = tx.get("aggregated_cs", 0)
    chr_score = tx.get("sender_chr", 0)
    mint_th = config.get("mint_threshold", 0.75)
    chr_th = config.get("min_chr", 0.6)

    # Both coherence and reputation must be high enough
    if cs >= mint_th and chr_score >= chr_th:
        return True
    return False


def apply_token_effects(state, tx):
    """
    Updates ledger state after successful validation
    
    Args:
        state: dict {"balances": {...}, "chr": {...}}
        tx: harmonic transaction
        
    Returns:
        dict: Updated ledger state
    """
    sender = tx["sender"]
    receiver = tx["receiver"]
    amount = tx.get("amount", 0)
    token = tx.get("token", "FLX")

    if tx.get("action") == "mint" and token == "FLX":
        state["balances"].setdefault(receiver, 0)
        state["balances"][receiver] += amount
    elif tx.get("action") == "transfer" and token == "FLX":
        if state["balances"].get(sender, 0) >= amount:
            state["balances"][sender] -= amount
            state["balances"].setdefault(receiver, 0)
            state["balances"][receiver] += amount
    elif token == "CHR" and tx.get("action") == "reward":
        # Non-transferable: just increment CHR reputation
        state["chr"].setdefault(receiver, 0)
        state["chr"][receiver] += amount
    return state