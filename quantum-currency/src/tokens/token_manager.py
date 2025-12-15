#!/usr/bin/env python3
"""
Token Manager for Quantum Currency Coherence System
Implements the TokenLedger class to track balances of T1-T5 tokens
"""

import time
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class TokenTransaction:
    """Represents a token transaction"""
    transaction_id: str
    from_address: str
    to_address: str
    token_type: str  # T1, T2, T3, T4, T5
    amount: float
    timestamp: float
    transaction_type: str  # transfer, stake, slash, reward
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenLedger:
    """
    Smart Token Ledger to track balances of T1-T5 tokens and maintain audit log
    """
    
    def __init__(self, network_id: str = "quantum-currency-tokens-001"):
        self.network_id = network_id
        self.balances: Dict[str, Dict[str, float]] = {}  # {address: {token_type: amount}}
        self.transactions: List[TokenTransaction] = []
        self.audit_log: List[Dict[str, Any]] = []
        
        # Initialize token types
        self.token_types = ["T1", "T2", "T3", "T4", "T5"]
        
    def _get_address_balance(self, address: str, token_type: str) -> float:
        """Get balance for a specific address and token type"""
        return self.balances.get(address, {}).get(token_type, 0.0)
    
    def _set_address_balance(self, address: str, token_type: str, amount: float):
        """Set balance for a specific address and token type"""
        if address not in self.balances:
            self.balances[address] = {}
        self.balances[address][token_type] = max(0.0, amount)
    
    def transfer(self, from_addr: str, to_addr: str, token_type: str, amount: float) -> bool:
        """
        Transfer tokens between addresses
        
        Args:
            from_addr: Source address
            to_addr: Destination address
            token_type: Type of token (T1-T5)
            amount: Amount to transfer
            
        Returns:
            bool: True if transfer successful, False otherwise
        """
        # Validate token type
        if token_type not in self.token_types:
            self._log_audit("transfer", "error", f"Invalid token type: {token_type}")
            return False
            
        # Check sufficient balance
        from_balance = self._get_address_balance(from_addr, token_type)
        if from_balance < amount:
            self._log_audit("transfer", "error", f"Insufficient balance: {from_balance} < {amount}")
            return False
            
        # Perform transfer
        self._set_address_balance(from_addr, token_type, from_balance - amount)
        to_balance = self._get_address_balance(to_addr, token_type)
        self._set_address_balance(to_addr, token_type, to_balance + amount)
        
        # Log transaction
        transaction_id = self._generate_transaction_id(from_addr, to_addr, token_type, amount)
        transaction = TokenTransaction(
            transaction_id=transaction_id,
            from_address=from_addr,
            to_address=to_addr,
            token_type=token_type,
            amount=amount,
            timestamp=time.time(),
            transaction_type="transfer"
        )
        self.transactions.append(transaction)
        
        # Log audit
        self._log_audit("transfer", "success", f"Transferred {amount} {token_type} from {from_addr} to {to_addr}")
        
        return True
    
    def stake(self, validator_id: str, token_type: str, amount: float) -> bool:
        """
        Stake tokens for a validator
        
        Args:
            validator_id: Validator ID
            token_type: Type of token (T1 for staking)
            amount: Amount to stake
            
        Returns:
            bool: True if staking successful, False otherwise
        """
        # Validate token type (T1 is for staking)
        if token_type != "T1":
            self._log_audit("stake", "error", f"Invalid token type for staking: {token_type}")
            return False
            
        # Check sufficient balance
        balance = self._get_address_balance(validator_id, token_type)
        if balance < amount:
            self._log_audit("stake", "error", f"Insufficient balance for staking: {balance} < {amount}")
            return False
            
        # Perform staking (move to staked balance)
        self._set_address_balance(validator_id, token_type, balance - amount)
        staked_balance = self._get_address_balance(validator_id, f"{token_type}_staked")
        self._set_address_balance(validator_id, f"{token_type}_staked", staked_balance + amount)
        
        # Log transaction
        transaction_id = self._generate_transaction_id(validator_id, "staking", token_type, amount)
        transaction = TokenTransaction(
            transaction_id=transaction_id,
            from_address=validator_id,
            to_address="staking",
            token_type=token_type,
            amount=amount,
            timestamp=time.time(),
            transaction_type="stake",
            metadata={"validator_id": validator_id}
        )
        self.transactions.append(transaction)
        
        # Log audit
        self._log_audit("stake", "success", f"Staked {amount} {token_type} for validator {validator_id}")
        
        return True
    
    def slash(self, validator_id: str, token_type: str, fraction: float) -> float:
        """
        Slash tokens from a validator
        
        Args:
            validator_id: Validator ID
            token_type: Type of token (T1 for staking, T4 for boosts)
            fraction: Fraction to slash (0.0 to 1.0)
            
        Returns:
            float: Amount slashed
        """
        # Validate fraction
        if not (0.0 <= fraction <= 1.0):
            self._log_audit("slash", "error", f"Invalid fraction: {fraction}")
            return 0.0
            
        # Validate token type (T1 and T4 can be slashed)
        if token_type not in ["T1", "T4"]:
            self._log_audit("slash", "error", f"Invalid token type for slashing: {token_type}")
            return 0.0
            
        # Get staked balance
        staked_balance_key = f"{token_type}_staked" if token_type == "T1" else token_type
        staked_balance = self._get_address_balance(validator_id, staked_balance_key)
        
        # Calculate amount to slash
        amount_slashed = staked_balance * fraction
        
        # Perform slashing
        if token_type == "T1":
            # Slash staked T1
            self._set_address_balance(validator_id, staked_balance_key, staked_balance - amount_slashed)
        else:  # T4
            # Slash T4 tokens
            t4_balance = self._get_address_balance(validator_id, token_type)
            self._set_address_balance(validator_id, token_type, t4_balance - amount_slashed)
        
        # Log transaction
        transaction_id = self._generate_transaction_id(validator_id, "slashing", token_type, amount_slashed)
        transaction = TokenTransaction(
            transaction_id=transaction_id,
            from_address=validator_id,
            to_address="slashing",
            token_type=token_type,
            amount=amount_slashed,
            timestamp=time.time(),
            transaction_type="slash",
            metadata={"validator_id": validator_id, "fraction": fraction}
        )
        self.transactions.append(transaction)
        
        # Log audit
        self._log_audit("slash", "success", f"Slashed {amount_slashed} {token_type} from validator {validator_id} ({fraction*100:.2f}%)")
        
        return amount_slashed
    
    def reward(self, validator_id: str, token_type: str, amount: float) -> bool:
        """
        Reward tokens to a validator
        
        Args:
            validator_id: Validator ID
            token_type: Type of token (T2 for rewards, T5 for memory incentives)
            amount: Amount to reward
            
        Returns:
            bool: True if rewarding successful, False otherwise
        """
        # Validate token type (T2 and T5 can be rewarded)
        if token_type not in ["T2", "T5"]:
            self._log_audit("reward", "error", f"Invalid token type for rewarding: {token_type}")
            return False
            
        # Add reward to validator's balance
        current_balance = self._get_address_balance(validator_id, token_type)
        self._set_address_balance(validator_id, token_type, current_balance + amount)
        
        # Log transaction
        transaction_id = self._generate_transaction_id("reward_pool", validator_id, token_type, amount)
        transaction = TokenTransaction(
            transaction_id=transaction_id,
            from_address="reward_pool",
            to_address=validator_id,
            token_type=token_type,
            amount=amount,
            timestamp=time.time(),
            transaction_type="reward",
            metadata={"validator_id": validator_id}
        )
        self.transactions.append(transaction)
        
        # Log audit
        self._log_audit("reward", "success", f"Rewarded {amount} {token_type} to validator {validator_id}")
        
        return True
    
    def get_balance(self, address: str, token_type: str) -> float:
        """
        Get token balance for an address
        
        Args:
            address: Address to check
            token_type: Type of token (T1-T5)
            
        Returns:
            float: Token balance
        """
        return self._get_address_balance(address, token_type)
    
    def get_total_supply(self, token_type: str) -> float:
        """
        Get total supply of a token type
        
        Args:
            token_type: Type of token (T1-T5)
            
        Returns:
            float: Total supply
        """
        total = 0.0
        for address_balances in self.balances.values():
            total += address_balances.get(token_type, 0.0)
        return total
    
    def _generate_transaction_id(self, from_addr: str, to_addr: str, token_type: str, amount: float) -> str:
        """Generate a unique transaction ID"""
        data = f"{from_addr}{to_addr}{token_type}{amount}{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def _log_audit(self, action: str, status: str, message: str):
        """Log an audit entry"""
        audit_entry = {
            "timestamp": time.time(),
            "action": action,
            "status": status,
            "message": message
        }
        self.audit_log.append(audit_entry)
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log"""
        return self.audit_log.copy()
    
    def get_transaction_history(self, address: Optional[str] = None) -> List[TokenTransaction]:
        """
        Get transaction history, optionally filtered by address
        
        Args:
            address: Optional address to filter by
            
        Returns:
            List of TokenTransaction objects
        """
        if address is None:
            return self.transactions.copy()
        
        # Filter by address
        filtered_transactions = [
            tx for tx in self.transactions 
            if tx.from_address == address or tx.to_address == address
        ]
        return filtered_transactions


# Example usage and testing
if __name__ == "__main__":
    # Create token ledger
    ledger = TokenLedger()
    
    # Test transfers
    print("Testing token transfers...")
    success = ledger.transfer("genesis_pool", "validator-001", "T1", 1000.0)
    print(f"Transfer T1: {success}")
    
    success = ledger.transfer("genesis_pool", "validator-001", "T2", 500.0)
    print(f"Transfer T2: {success}")
    
    success = ledger.transfer("genesis_pool", "validator-001", "T4", 100.0)
    print(f"Transfer T4: {success}")
    
    # Test staking
    print("\nTesting staking...")
    success = ledger.stake("validator-001", "T1", 500.0)
    print(f"Stake T1: {success}")
    
    # Test rewards
    print("\nTesting rewards...")
    success = ledger.reward("validator-001", "T2", 50.0)
    print(f"Reward T2: {success}")
    
    success = ledger.reward("validator-001", "T5", 25.0)
    print(f"Reward T5: {success}")
    
    # Check balances
    print("\nChecking balances...")
    print(f"Validator T1 balance: {ledger.get_balance('validator-001', 'T1')}")
    print(f"Validator T1 staked: {ledger.get_balance('validator-001', 'T1_staked')}")
    print(f"Validator T2 balance: {ledger.get_balance('validator-001', 'T2')}")
    print(f"Validator T4 balance: {ledger.get_balance('validator-001', 'T4')}")
    print(f"Validator T5 balance: {ledger.get_balance('validator-001', 'T5')}")
    
    # Test slashing
    print("\nTesting slashing...")
    slashed_amount = ledger.slash("validator-001", "T1", 0.1)  # Slash 10%
    print(f"Slashed T1: {slashed_amount}")
    
    slashed_amount = ledger.slash("validator-001", "T4", 0.2)  # Slash 20%
    print(f"Slashed T4: {slashed_amount}")
    
    # Check final balances
    print("\nFinal balances...")
    print(f"Validator T1 balance: {ledger.get_balance('validator-001', 'T1')}")
    print(f"Validator T1 staked: {ledger.get_balance('validator-001', 'T1_staked')}")
    print(f"Validator T4 balance: {ledger.get_balance('validator-001', 'T4')}")
    
    print("\n✅ Token ledger demo completed!")