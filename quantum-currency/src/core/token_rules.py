#!/usr/bin/env python3
"""
Token Rules Engine for Multi-Token System
Defines mint/burn/transfer rules based on harmonic coherence score (CS) and CHR reputation

Enhanced for v0.2.0 with Harmonic Gating mechanism that uses the φ-lattice memory structure:
- CHR (Coheron): Macro Memory Gating (L_Φ) - Only validator proposals weighted by CHR can trigger Macro Memory writes
- FLX (Φlux): Micro/Phase Retrieval Cost (L_μ, L_φ) - FLX consumed for high-bandwidth memory retrieval
- PSY (ΨSync): Synchronization Signal - ΨScore determines PSY distribution with behavioral balancing
- ATR (Attractor): Long-Term Anchor - Staked ATR determines Ω_target for L_Φ memory store
- RES (Resonance): Network Expansion Multiplier - Multiplies available Ω bandwidth for coherence-aligned growth
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
    Updates ledger state after successful validation with enhanced harmonic gating
    
    Args:
        state: dict {"balances": {...}, "chr": {...}}
        tx: harmonic transaction
        
    Returns:
        dict: Updated ledger state
    """
    sender = tx["sender"]
    receiver = tx.get("receiver", sender)  # Default to sender for self-transactions
    amount = tx.get("amount", 0)
    token = tx.get("token", "FLX")
    action = tx.get("action", "transfer")
    
    # Initialize balances if they don't exist
    if sender not in state["balances"]:
        state["balances"][sender] = {"CHR": 0, "FLX": 0, "PSY": 0, "ATR": 0, "RES": 0}
    if receiver not in state["balances"]:
        state["balances"][receiver] = {"CHR": 0, "FLX": 0, "PSY": 0, "ATR": 0, "RES": 0}
    
    # Handle different token types based on their properties and harmonic gating
    if token == "CHR":
        # CHR (Coheron) - Non-transferable, represents ethical alignment
        # Harmonic Gating: Macro Memory Gating (L_Φ) - Only validator proposals weighted by CHR can trigger Macro Memory writes
        if action == "reward":
            state["balances"][receiver]["CHR"] += amount
            # Update CHR reputation score
            state["chr"][receiver] = state["chr"].get(receiver, 0) + (amount / 1000)  # Scale down for reputation
        elif action == "stake":
            # Staking CHR for ATR (Stability)
            if state["balances"][sender]["CHR"] >= amount:
                state["balances"][sender]["CHR"] -= amount
                # Convert to ATR at a rate based on coherence
                cs = tx.get("aggregated_cs", 0)
                atr_amount = amount * (0.5 + cs * 0.5)  # 50-100% conversion based on coherence
                state["balances"][receiver]["ATR"] += atr_amount
        elif action == "convert_to_res":
            # Converting CHR to RES (Expansion) for network bridging
            if state["balances"][sender]["CHR"] >= amount:
                state["balances"][sender]["CHR"] -= amount
                # Convert to RES at a rate based on network health
                res_amount = amount * 0.1  # 10% conversion to RES
                state["balances"][receiver]["RES"] += res_amount
        elif action == "macro_write_gate":
            # Special action: Only high-CHR validators can trigger macro memory writes
            chr_score = state["chr"].get(sender, 0)
            psi_threshold = tx.get("psi_threshold", 0.85)  # Ψ threshold for macro writes
            if chr_score >= 0.7 and tx.get("psi_score", 0) >= psi_threshold:
                # Allow macro memory write - this would integrate with CAL's L_Φ memory store
                state["balances"][receiver]["CHR"] += amount * 0.01  # Small reward for ethical anchoring
                print(f"✅ Macro Memory Write authorized by {sender} (CHR: {chr_score:.4f})")
            else:
                print(f"❌ Macro Memory Write rejected for {sender} (CHR: {chr_score:.4f}, Ψ: {tx.get('psi_score', 0):.4f})")
    
    elif token == "FLX":
        # FLX (Φlux) - Transferable, represents usable field energy
        # Harmonic Gating: Micro/Phase Retrieval Cost (L_μ, L_φ) - FLX consumed for high-bandwidth memory retrieval
        if action == "mint":
            state["balances"][receiver]["FLX"] += amount
        elif action == "transfer":
            if state["balances"][sender]["FLX"] >= amount:
                state["balances"][sender]["FLX"] -= amount
                state["balances"][receiver]["FLX"] += amount
        elif action == "convert_to_psy":
            # Converting FLX to PSY (ΨSync) for synchronization
            if state["balances"][sender]["FLX"] >= amount:
                state["balances"][sender]["FLX"] -= amount
                # Convert to PSY at a rate based on network synchronization
                cs = tx.get("aggregated_cs", 0)
                psy_amount = amount * (0.3 + cs * 0.4)  # 30-70% conversion based on coherence
                state["balances"][receiver]["PSY"] += psy_amount
        elif action == "memory_retrieval":
            # Special action: FLX consumed for high-bandwidth memory retrieval
            retrieval_cost = tx.get("retrieval_cost", amount * 0.1)  # 10% cost by default
            if state["balances"][sender]["FLX"] >= retrieval_cost:
                state["balances"][sender]["FLX"] -= retrieval_cost
                # This would integrate with CAL's memory retrieval mechanisms
                print(f"💾 Memory retrieval completed for {sender} (cost: {retrieval_cost} FLX)")
            else:
                print(f"❌ Insufficient FLX for memory retrieval by {sender}")
    
    elif token == "PSY":
        # PSY (ΨSync) - Semi-transferable, represents synchronization between nodes
        # Harmonic Gating: Synchronization Signal - ΨScore determines PSY distribution with behavioral balancing
        if action == "mint":
            state["balances"][receiver]["PSY"] += amount
        elif action == "transfer":
            # PSY can only be transferred with a fee for semi-transferability
            fee = amount * 0.1  # 10% fee
            transfer_amount = amount - fee
            if state["balances"][sender]["PSY"] >= amount:
                state["balances"][sender]["PSY"] -= amount
                state["balances"][receiver]["PSY"] += transfer_amount
                # Fee goes to network stability (ATR)
                state["balances"]["network"]["ATR"] = state["balances"].get("network", {}).get("ATR", 0) + fee
        elif action == "convert_to_atr":
            # Converting PSY to ATR (Stability) for anchoring stability
            if state["balances"][sender]["PSY"] >= amount:
                state["balances"][sender]["PSY"] -= amount
                # Convert to ATR at a rate based on stability needs
                atr_amount = amount * 0.8  # 80% conversion to ATR
                state["balances"][receiver]["ATR"] += atr_amount
        elif action == "behavioral_balance":
            # Special action: Behavioral Balancer based on ΨScore
            psi_score = tx.get("psi_score", 0)
            if psi_score < 0.5:
                # Low ΨScore - apply penalty
                penalty = min(amount * 0.5, state["balances"][sender]["PSY"] * 0.1)  # Max 10% penalty
                state["balances"][sender]["PSY"] -= penalty
                print(f"⚠️ PSY penalty applied to {sender} (Ψ: {psi_score:.4f}, penalty: {penalty:.4f})")
            elif psi_score > 0.9:
                # High ΨScore - apply reward
                reward = amount * 0.2  # 20% reward
                state["balances"][receiver]["PSY"] += reward
                print(f"🌟 PSY reward applied to {sender} (Ψ: {psi_score:.4f}, reward: {reward:.4f})")
    
    elif token == "ATR":
        # ATR (Attractor) - Stakable, anchors stability during transitions
        # Harmonic Gating: Long-Term Anchor - Staked ATR determines Ω_target for L_Φ memory store
        if action == "mint":
            state["balances"][receiver]["ATR"] += amount
        elif action == "stake":
            # ATR can be staked for network validation
            if state["balances"][sender]["ATR"] >= amount:
                state["balances"][sender]["ATR"] -= amount
                # Staked ATR contributes to validator power
                validator_id = tx.get("validator_id", "default")
                state["staking"] = state.get("staking", {})
                state["staking"][validator_id] = state["staking"].get(validator_id, 0) + amount
        elif action == "unstake":
            # Unstake ATR
            validator_id = tx.get("validator_id", "default")
            if state["staking"].get(validator_id, 0) >= amount:
                state["staking"][validator_id] -= amount
                state["balances"][receiver]["ATR"] += amount
        elif action == "convert_to_res":
            # Converting ATR to RES (Expansion) for network expansion
            if state["balances"][sender]["ATR"] >= amount:
                state["balances"][sender]["ATR"] -= amount
                # Convert to RES at a rate based on expansion needs
                res_amount = amount * 0.5  # 50% conversion to RES
                state["balances"][receiver]["RES"] += res_amount
        elif action == "set_omega_target":
            # Special action: Staked ATR determines Ω_target for L_Φ memory store
            staked_atr = state["staking"].get(sender, 0)
            if staked_atr >= amount:
                # This would integrate with CAL's Ω_target mechanism
                omega_target = tx.get("omega_target", 0.8)  # Default Ω target
                print(f"🎯 Ω_target set by {sender} (staked ATR: {staked_atr}, target: {omega_target:.4f})")
    
    elif token == "RES":
        # RES (Resonance) - Multiplicative, rewards bridging of new networks
        # Harmonic Gating: Network Expansion Multiplier - Multiplies available Ω bandwidth for coherence-aligned growth
        if action == "mint":
            state["balances"][receiver]["RES"] += amount
        elif action == "transfer":
            if state["balances"][sender]["RES"] >= amount:
                state["balances"][sender]["RES"] -= amount
                state["balances"][receiver]["RES"] += amount
        elif action == "multiply":
            # RES has multiplicative properties
            multiplier = tx.get("multiplier", 1.1)  # Default 10% increase
            current_res = state["balances"][receiver]["RES"]
            multiplied_amount = current_res * (multiplier - 1)
            state["balances"][receiver]["RES"] *= multiplier
            # Distribute some to network for expansion incentives
            network_share = multiplied_amount * 0.2  # 20% to network
            state["balances"]["network"]["RES"] = state["balances"].get("network", {}).get("RES", 0) + network_share
        elif action == "expand_bandwidth":
            # Special action: Multiply available Ω bandwidth for coherence-aligned growth
            res_balance = state["balances"][sender]["RES"]
            omega_bandwidth_multiplier = 1.0 + (res_balance / 1000)  # Scale RES to bandwidth multiplier
            max_multiplier = tx.get("max_multiplier", 2.0)  # Cap at 2x by default
            actual_multiplier = min(omega_bandwidth_multiplier, max_multiplier)
            print(f"🚀 Ω bandwidth expanded by {sender} (RES: {res_balance}, multiplier: {actual_multiplier:.4f})")
            # This would integrate with CAL's Ω bandwidth management

    return state

def get_token_properties(token_type):
    """
    Get properties of a specific token type with harmonic gating information
    
    Args:
        token_type: Token type (CHR, FLX, PSY, ATR, RES)
        
    Returns:
        dict: Token properties
    """
    token_properties = {
        "CHR": {
            "name": "Coheron",
            "description": "Measures conscious and ethical coherence",
            "transferable": False,
            "stakable": False,
            "convertible": True,
            "utility": "ethical_alignment",
            "phi_lattice_mapping": "Macro Memory Gating (L_Φ)",
            "harmonic_gate": "Only validator proposals weighted by CHR can trigger Macro Memory writes"
        },
        "FLX": {
            "name": "Φlux",
            "description": "Represents usable field energy",
            "transferable": True,
            "stakable": False,
            "convertible": True,
            "utility": "energy",
            "phi_lattice_mapping": "Micro/Phase Retrieval Cost (L_μ, L_φ)",
            "harmonic_gate": "FLX consumed for high-bandwidth memory retrieval"
        },
        "PSY": {
            "name": "ΨSync",
            "description": "Represents synchronization between nodes",
            "transferable": "semi",  # Semi-transferable with fees
            "stakable": False,
            "convertible": True,
            "utility": "coherence",
            "phi_lattice_mapping": "Synchronization Signal",
            "harmonic_gate": "ΨScore determines PSY distribution with behavioral balancing"
        },
        "ATR": {
            "name": "Attractor",
            "description": "Anchors stability during transitions",
            "transferable": True,
            "stakable": True,
            "convertible": True,
            "utility": "stability",
            "phi_lattice_mapping": "Long-Term Anchor",
            "harmonic_gate": "Staked ATR determines Ω_target for L_Φ memory store"
        },
        "RES": {
            "name": "Resonance",
            "description": "Rewards bridging of new networks",
            "transferable": True,
            "stakable": False,
            "convertible": True,
            "utility": "expansion",
            "multiplicative": True,
            "phi_lattice_mapping": "Network Expansion Multiplier",
            "harmonic_gate": "Multiplies available Ω bandwidth for coherence-aligned growth"
        }
    }
    
    return token_properties.get(token_type, {})