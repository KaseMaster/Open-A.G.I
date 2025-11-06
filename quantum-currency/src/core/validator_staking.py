#!/usr/bin/env python3
"""
Validator Staking System for Quantum Currency
Implements validator staking, delegation, and liquidity incentives with multi-token support
"""

import sys
import os
import json
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class StakingPosition:
    """Represents a staking position"""
    position_id: str
    staker_address: str
    validator_id: str
    token_type: str  # "FLX", "ATR", "CHR" (CHR staking converts to ATR)
    amount: float
    staked_at: float
    lockup_period: float  # in days
    reward_rate: float  # annual percentage rate
    claimed_rewards: float = 0.0
    is_active: bool = True

@dataclass
class Delegation:
    """Represents a delegation from a token holder to a validator"""
    delegation_id: str
    delegator_address: str
    validator_id: str
    token_type: str  # "FLX", "ATR"
    amount: float
    delegated_at: float
    reward_share: float  # percentage of validator rewards to share with delegator (0.0 to 1.0)
    claimed_rewards: float = 0.0
    is_active: bool = True

@dataclass
class Validator:
    """Represents a network validator"""
    validator_id: str
    operator_address: str
    chr_score: float  # Reputation score
    total_staked: Dict[str, float]  # {token_type: amount}
    total_delegated: Dict[str, float]  # {token_type: amount}
    commission_rate: float = 0.1  # 10% commission on rewards
    uptime: float = 1.0  # 0.0 to 1.0
    is_active: bool = True
    last_rewarded: float = 0.0

@dataclass
class LiquidityPool:
    """Represents a liquidity pool for token pairs"""
    pool_id: str
    token_pair: Tuple[str, str]  # e.g., ("FLX", "CHR")
    reserves: Dict[str, float]  # {token_type: amount}
    total_liquidity: float
    reward_rate: float  # APR for liquidity providers
    active_providers: int = 0

class ValidatorStakingSystem:
    """
    Implements validator staking, delegation, and liquidity incentives with multi-token support
    """
    
    def __init__(self):
        self.staking_positions: Dict[str, StakingPosition] = {}
        self.delegations: Dict[str, Delegation] = {}
        self.validators: Dict[str, Validator] = {}
        self.liquidity_pools: Dict[str, LiquidityPool] = {}
        self.pending_rewards: Dict[str, Dict[str, float]] = {}  # {address: {token_type: amount}}
        self.staking_config = {
            "min_stake_amount": {
                "FLX": 1000.0,
                "ATR": 500.0,
                "CHR": 2000.0  # CHR staking converts to ATR
            },
            "max_lockup_period": 365.0,  # days
            "base_reward_rates": {
                "FLX": 0.12,  # 12% APR
                "ATR": 0.15,  # 15% APR (higher for stability token)
                "CHR": 0.08   # 8% APR (lower for reputation token)
            },
            "commission_rate": 0.1,  # 10% validator commission
            "unbonding_period": 7.0  # days
        }
        self._initialize_validators()
        self._initialize_liquidity_pools()
    
    def _initialize_validators(self):
        """Initialize network validators"""
        validators_data = [
            ("validator-001", "valoper1xyz...", 0.95),
            ("validator-002", "valoper2abc...", 0.87),
            ("validator-003", "valoper3def...", 0.92),
            ("validator-004", "valoper4ghi...", 0.78),
            ("validator-005", "valoper5jkl...", 0.89)
        ]
        
        for validator_id, operator_address, chr_score in validators_data:
            validator = Validator(
                validator_id=validator_id,
                operator_address=operator_address,
                chr_score=chr_score,
                total_staked={"FLX": 0.0, "ATR": 0.0},
                total_delegated={"FLX": 0.0, "ATR": 0.0},
                commission_rate=self.staking_config["commission_rate"]
            )
            self.validators[validator_id] = validator
    
    def _initialize_liquidity_pools(self):
        """Initialize liquidity pools"""
        pools_data = [
            ("pool-flx-chr", ("FLX", "CHR"), {"FLX": 100000.0, "CHR": 50000.0}),
            ("pool-flx-atr", ("FLX", "ATR"), {"FLX": 75000.0, "ATR": 37500.0}),
            ("pool-atr-res", ("ATR", "RES"), {"ATR": 25000.0, "RES": 12500.0}),
            ("pool-psych-atr", ("PSY", "ATR"), {"PSY": 50000.0, "ATR": 25000.0})
        ]
        
        for pool_id, token_pair, reserves in pools_data:
            total_liquidity = sum(reserves.values())
            pool = LiquidityPool(
                pool_id=pool_id,
                token_pair=token_pair,
                reserves=reserves,
                total_liquidity=total_liquidity,
                reward_rate=0.15  # 15% APR
            )
            self.liquidity_pools[pool_id] = pool
    
    def create_staking_position(self, staker_address: str, validator_id: str, 
                              token_type: str, amount: float, lockup_period: float) -> Optional[str]:
        """
        Create a new staking position
        
        Args:
            staker_address: Address of the staker
            validator_id: ID of the validator to stake with
            token_type: Type of token to stake ("FLX", "ATR", or "CHR")
            amount: Amount to stake
            lockup_period: Lockup period in days
            
        Returns:
            Position ID if successful, None otherwise
        """
        # Validate inputs
        min_stake = self.staking_config["min_stake_amount"].get(token_type, 1000.0)
        if amount < min_stake:
            print(f"Staking amount {amount} is below minimum {min_stake} for {token_type}")
            return None
        
        if lockup_period > self.staking_config["max_lockup_period"]:
            print(f"Lockup period {lockup_period} exceeds maximum {self.staking_config['max_lockup_period']}")
            return None
        
        if validator_id not in self.validators:
            print(f"Validator {validator_id} not found")
            return None
        
        validator = self.validators[validator_id]
        if not validator.is_active:
            print(f"Validator {validator_id} is not active")
            return None
        
        # Handle CHR staking (converts to ATR)
        effective_token_type = token_type
        effective_amount = amount
        if token_type == "CHR":
            # CHR staking converts to ATR at a rate based on validator CHR score
            conversion_rate = 0.5 + validator.chr_score * 0.3  # 50-80% conversion
            effective_token_type = "ATR"
            effective_amount = amount * conversion_rate
            print(f"CHR staking converted {amount} CHR to {effective_amount:.2f} ATR at {conversion_rate:.2%} rate")
        
        # Create staking position
        position_id = f"stake-{int(time.time())}-{hashlib.md5(staker_address.encode()).hexdigest()[:8]}"
        
        # Calculate reward rate based on validator CHR score, lockup period, and token type
        base_rate = self.staking_config["base_reward_rates"].get(effective_token_type, 0.12)
        chr_bonus = validator.chr_score * 0.05  # Up to 5% bonus for high CHR score
        lockup_bonus = min(lockup_period / 365.0 * 0.03, 0.03)  # Up to 3% bonus for long lockup
        reward_rate = base_rate + chr_bonus + lockup_bonus
        
        position = StakingPosition(
            position_id=position_id,
            staker_address=staker_address,
            validator_id=validator_id,
            token_type=token_type,  # Store original token type
            amount=amount,
            staked_at=time.time(),
            lockup_period=lockup_period,
            reward_rate=reward_rate
        )
        
        self.staking_positions[position_id] = position
        
        # Update validator's total staked for the effective token type
        validator.total_staked[effective_token_type] = validator.total_staked.get(effective_token_type, 0.0) + effective_amount
        
        return position_id
    
    def delegate_tokens(self, delegator_address: str, validator_id: str, 
                       token_type: str, amount: float, reward_share: float = 0.5) -> Optional[str]:
        """
        Delegate tokens to a validator
        
        Args:
            delegator_address: Address of the delegator
            validator_id: ID of the validator to delegate to
            token_type: Type of token to delegate ("FLX" or "ATR")
            amount: Amount to delegate
            reward_share: Percentage of rewards to share with delegator (0.0 to 1.0)
            
        Returns:
            Delegation ID if successful, None otherwise
        """
        # Validate inputs
        min_delegation = self.staking_config["min_stake_amount"].get(token_type, 1000.0) / 10  # Lower minimum for delegation
        if amount < min_delegation:
            print(f"Delegation amount {amount} is below minimum {min_delegation} for {token_type}")
            return None
        
        if not (0.0 <= reward_share <= 1.0):
            print("Reward share must be between 0.0 and 1.0")
            return None
        
        if token_type not in ["FLX", "ATR"]:
            print(f"Token type {token_type} not supported for delegation")
            return None
        
        if validator_id not in self.validators:
            print(f"Validator {validator_id} not found")
            return None
        
        validator = self.validators[validator_id]
        if not validator.is_active:
            print(f"Validator {validator_id} is not active")
            return None
        
        # Create delegation
        delegation_id = f"delegate-{int(time.time())}-{hashlib.md5(delegator_address.encode()).hexdigest()[:8]}"
        
        delegation = Delegation(
            delegation_id=delegation_id,
            delegator_address=delegator_address,
            validator_id=validator_id,
            token_type=token_type,
            amount=amount,
            delegated_at=time.time(),
            reward_share=reward_share
        )
        
        self.delegations[delegation_id] = delegation
        
        # Update validator's total delegated
        validator.total_delegated[token_type] = validator.total_delegated.get(token_type, 0.0) + amount
        
        return delegation_id
    
    def add_liquidity(self, provider_address: str, pool_id: str, 
                     token_amounts: Dict[str, float]) -> bool:
        """
        Add liquidity to a pool
        
        Args:
            provider_address: Address of the liquidity provider
            pool_id: ID of the liquidity pool
            token_amounts: Dictionary of {token_type: amount} to add
            
        Returns:
            True if successful, False otherwise
        """
        if pool_id not in self.liquidity_pools:
            print(f"Liquidity pool {pool_id} not found")
            return False
        
        pool = self.liquidity_pools[pool_id]
        
        # Validate token amounts match pool tokens
        for token_type in token_amounts:
            if token_type not in pool.token_pair:
                print(f"Token {token_type} not supported in pool {pool_id}")
                return False
        
        # Add liquidity to pool reserves
        for token_type, amount in token_amounts.items():
            pool.reserves[token_type] = pool.reserves.get(token_type, 0.0) + amount
        
        # Update total liquidity
        pool.total_liquidity = sum(pool.reserves.values())
        pool.active_providers += 1
        
        # Add to pending rewards for the provider
        # Initialize pending rewards for this address if not exists
        if provider_address not in self.pending_rewards:
            self.pending_rewards[provider_address] = {}
        
        # Reward based on contribution
        reward_amount = sum(token_amounts.values()) * 0.001  # 0.1% incentive
        for token_type in token_amounts:
            self.pending_rewards[provider_address][token_type] = self.pending_rewards[provider_address].get(token_type, 0.0) + (reward_amount / len(token_amounts))
        
        return True
    
    def calculate_rewards(self, time_period_hours: float = 24.0) -> Dict[str, Dict[str, float]]:
        """
        Calculate rewards for stakers, delegators, and liquidity providers
        
        Args:
            time_period_hours: Time period to calculate rewards for (default 24 hours)
            
        Returns:
            Dictionary of {address: {token_type: reward_amount}}
        """
        rewards = {}
        
        # Calculate time factor (convert hours to years for APR calculation)
        time_factor = time_period_hours / (365.0 * 24.0)
        
        # Calculate staking rewards
        for position_id, position in self.staking_positions.items():
            if not position.is_active:
                continue
            
            # Calculate reward amount
            reward_amount = position.amount * position.reward_rate * time_factor
            
            # Handle CHR staking (rewards in ATR)
            reward_token_type = "ATR" if position.token_type == "CHR" else position.token_type
            
            # Add to validator's pending rewards
            validator = self.validators[position.validator_id]
            validator_reward = reward_amount * validator.commission_rate
            staker_reward = reward_amount * (1 - validator.commission_rate)
            
            # Add rewards to respective addresses
            # Validator rewards
            if validator.operator_address not in rewards:
                rewards[validator.operator_address] = {}
            rewards[validator.operator_address][reward_token_type] = rewards[validator.operator_address].get(reward_token_type, 0.0) + validator_reward
            
            # Staker rewards
            if position.staker_address not in rewards:
                rewards[position.staker_address] = {}
            rewards[position.staker_address][reward_token_type] = rewards[position.staker_address].get(reward_token_type, 0.0) + staker_reward
            
            # Update claimed rewards
            position.claimed_rewards += staker_reward
        
        # Calculate delegation rewards
        for delegation_id, delegation in self.delegations.items():
            if not delegation.is_active:
                continue
            
            # Get validator
            validator = self.validators[delegation.validator_id]
            
            # Calculate base reward (based on validator's total stake)
            base_reward_rate = self.staking_config["base_reward_rates"].get(delegation.token_type, 0.12)
            base_reward = validator.total_staked.get(delegation.token_type, 0.0) * base_reward_rate * time_factor
            
            # Calculate delegator's share
            delegation_share = base_reward * delegation.reward_share
            
            # Add to delegator's rewards
            if delegation.delegator_address not in rewards:
                rewards[delegation.delegator_address] = {}
            rewards[delegation.delegator_address][delegation.token_type] = rewards[delegation.delegator_address].get(delegation.token_type, 0.0) + delegation_share
            
            # Update claimed rewards
            delegation.claimed_rewards += delegation_share
        
        # Calculate liquidity provider rewards
        for pool_id, pool in self.liquidity_pools.items():
            # Calculate pool reward
            pool_reward = pool.total_liquidity * pool.reward_rate * time_factor
            
            # Distribute to active providers (simplified distribution)
            if pool.active_providers > 0:
                provider_reward = pool_reward / pool.active_providers
                
                # In a real implementation, this would be distributed based on actual liquidity provision
                # For now, we'll just add to a generic provider reward pool
                if "liquidity_providers" not in rewards:
                    rewards["liquidity_providers"] = {}
                
                # Distribute rewards in both token types of the pool
                for token_type in pool.token_pair:
                    token_reward = provider_reward / len(pool.token_pair)
                    rewards["liquidity_providers"][token_type] = rewards["liquidity_providers"].get(token_type, 0.0) + token_reward
        
        return rewards
    
    def claim_rewards(self, address: str) -> Dict[str, float]:
        """
        Claim pending rewards for an address
        
        Args:
            address: Address to claim rewards for
            
        Returns:
            Dictionary of {token_type: amount_claimed}
        """
        if address not in self.pending_rewards:
            return {}
        
        claimed_rewards = self.pending_rewards[address].copy()
        self.pending_rewards[address] = {}
        
        return claimed_rewards
    
    def unstake_tokens(self, position_id: str) -> bool:
        """
        Unstake tokens (subject to unbonding period)
        
        Args:
            position_id: ID of the staking position
            
        Returns:
            True if successful, False otherwise
        """
        if position_id not in self.staking_positions:
            print(f"Staking position {position_id} not found")
            return False
        
        position = self.staking_positions[position_id]
        
        # Check if lockup period has passed
        current_time = time.time()
        lockup_end = position.staked_at + (position.lockup_period * 24 * 60 * 60)  # Convert days to seconds
        
        if current_time < lockup_end:
            remaining_time = (lockup_end - current_time) / (24 * 60 * 60)  # Convert to days
            print(f"Lockup period not yet ended. {remaining_time:.2f} days remaining")
            return False
        
        # Handle CHR staking (converts to ATR)
        effective_token_type = position.token_type
        effective_amount = position.amount
        if position.token_type == "CHR":
            # Get validator to determine conversion rate
            validator = self.validators.get(position.validator_id)
            if validator:
                conversion_rate = 0.5 + validator.chr_score * 0.3  # 50-80% conversion
                effective_token_type = "ATR"
                effective_amount = position.amount * conversion_rate
        
        # Update validator's total staked
        validator = self.validators[position.validator_id]
        validator.total_staked[effective_token_type] = max(0.0, validator.total_staked.get(effective_token_type, 0.0) - effective_amount)
        
        # Mark position as inactive
        position.is_active = False
        
        return True
    
    def get_validator_info(self, validator_id: str) -> Optional[Dict]:
        """
        Get information about a validator
        
        Args:
            validator_id: ID of the validator
            
        Returns:
            Dictionary with validator information or None if not found
        """
        if validator_id not in self.validators:
            return None
        
        validator = self.validators[validator_id]
        
        # Count active staking positions and delegations
        active_stakes = [pos for pos in self.staking_positions.values() 
                        if pos.validator_id == validator_id and pos.is_active]
        active_delegations = [delg for delg in self.delegations.values() 
                             if delg.validator_id == validator_id and delg.is_active]
        
        return {
            "validator_id": validator.validator_id,
            "operator_address": validator.operator_address,
            "chr_score": validator.chr_score,
            "total_staked": validator.total_staked,
            "total_delegated": validator.total_delegated,
            "total_staked_and_delegated": {token: validator.total_staked.get(token, 0.0) + validator.total_delegated.get(token, 0.0) 
                                         for token in set(list(validator.total_staked.keys()) + list(validator.total_delegated.keys()))},
            "commission_rate": validator.commission_rate,
            "uptime": validator.uptime,
            "is_active": validator.is_active,
            "active_stakes": len(active_stakes),
            "active_delegations": len(active_delegations)
        }
    
    def get_staking_apr(self, validator_id: str, token_type: str = "FLX") -> Optional[float]:
        """
        Get the APR for staking with a validator
        
        Args:
            validator_id: ID of the validator
            token_type: Type of token to stake
            
        Returns:
            APR as a decimal (0.12 = 12%) or None if validator not found
        """
        if validator_id not in self.validators:
            return None
        
        validator = self.validators[validator_id]
        
        # Base rate plus CHR score bonus
        base_rate = self.staking_config["base_reward_rates"].get(token_type, 0.12)
        chr_bonus = validator.chr_score * 0.05  # Up to 5% bonus for high CHR score
        apr = base_rate + chr_bonus
        
        return apr
    
    def get_system_metrics(self) -> Dict:
        """
        Get overall system metrics
        
        Returns:
            Dictionary with system metrics
        """
        # Count active positions
        active_stakes = [pos for pos in self.staking_positions.values() if pos.is_active]
        active_delegations = [delg for delg in self.delegations.values() if delg.is_active]
        
        # Calculate total staked value by token type
        total_staked = {}
        total_delegated = {}
        
        for validator in self.validators.values():
            for token_type, amount in validator.total_staked.items():
                total_staked[token_type] = total_staked.get(token_type, 0.0) + amount
            for token_type, amount in validator.total_delegated.items():
                total_delegated[token_type] = total_delegated.get(token_type, 0.0) + amount
        
        # Calculate total liquidity
        total_liquidity = sum(pool.total_liquidity for pool in self.liquidity_pools.values())
        
        # Calculate total pending rewards
        total_pending_rewards = {}
        for address_rewards in self.pending_rewards.values():
            for token_type, amount in address_rewards.items():
                total_pending_rewards[token_type] = total_pending_rewards.get(token_type, 0.0) + amount
        
        return {
            "total_validators": len(self.validators),
            "active_validators": len([v for v in self.validators.values() if v.is_active]),
            "total_staking_positions": len(self.staking_positions),
            "active_staking_positions": len(active_stakes),
            "total_delegations": len(self.delegations),
            "active_delegations": len(active_delegations),
            "total_staked": total_staked,
            "total_delegated": total_delegated,
            "total_staked_and_delegated": {token: total_staked.get(token, 0.0) + total_delegated.get(token, 0.0) 
                                         for token in set(list(total_staked.keys()) + list(total_delegated.keys()))},
            "total_liquidity_pools": len(self.liquidity_pools),
            "total_liquidity": total_liquidity,
            "pending_rewards": total_pending_rewards
        }

def demo_validator_staking():
    """Demonstrate validator staking system capabilities"""
    print("🏦 Validator Staking System Demo")
    print("=" * 35)
    
    # Create staking system instance
    staking_system = ValidatorStakingSystem()
    
    # Show initial system metrics
    print("\n📊 Initial System Metrics:")
    metrics = staking_system.get_system_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    print(f"      {sub_key}: {sub_value:,.2f}")
                else:
                    print(f"      {sub_key}: {sub_value}")
        elif isinstance(value, float):
            print(f"   {key}: {value:,.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Show validators
    print("\n🏛️  Network Validators:")
    for validator_id, validator in staking_system.validators.items():
        print(f"   {validator_id}: CHR={validator.chr_score:.2f}, Commission={validator.commission_rate:.1%}")
    
    # Show liquidity pools
    print("\n💧 Liquidity Pools:")
    for pool_id, pool in staking_system.liquidity_pools.items():
        print(f"   {pool_id}: {pool.token_pair[0]}/{pool.token_pair[1]}, Liquidity=${pool.total_liquidity:,.2f}")
    
    # Create staking positions with different token types
    print("\n🔒 Creating Staking Positions:")
    staker_addresses = ["staker-001", "staker-002", "staker-003"]
    validator_ids = list(staking_system.validators.keys())
    token_types = ["FLX", "ATR", "CHR"]
    
    for i, staker in enumerate(staker_addresses):
        validator_id = validator_ids[i % len(validator_ids)]
        token_type = token_types[i % len(token_types)]
        amount = 5000.0 + i * 1000.0
        lockup_period = 90.0 + i * 30.0  # 90, 120, 150 days
        
        position_id = staking_system.create_staking_position(
            staker_address=staker,
            validator_id=validator_id,
            token_type=token_type,
            amount=amount,
            lockup_period=lockup_period
        )
        
        if position_id:
            apr = staking_system.get_staking_apr(validator_id, token_type)
            print(f"   {staker} staked {amount:.2f} {token_type} with {validator_id} (APR: {apr:.1%})")
        else:
            print(f"   Failed to create staking position for {staker}")
    
    # Create delegations
    print("\n🤝 Creating Delegations:")
    delegator_addresses = ["delegator-001", "delegator-002", "delegator-003"]
    
    for i, delegator in enumerate(delegator_addresses):
        validator_id = validator_ids[i % len(validator_ids)]
        token_type = "FLX" if i % 2 == 0 else "ATR"
        amount = 2000.0 + i * 500.0
        reward_share = 0.6 + i * 0.1  # 60%, 70%, 80%
        
        delegation_id = staking_system.delegate_tokens(
            delegator_address=delegator,
            validator_id=validator_id,
            token_type=token_type,
            amount=amount,
            reward_share=reward_share
        )
        
        if delegation_id:
            print(f"   {delegator} delegated {amount:.2f} {token_type} to {validator_id} ({reward_share:.0%} reward share)")
        else:
            print(f"   Failed to create delegation for {delegator}")
    
    # Add liquidity to pools
    print("\n💧 Adding Liquidity:")
    liquidity_providers = ["lp-001", "lp-002"]
    pool_ids = list(staking_system.liquidity_pools.keys())
    
    for i, provider in enumerate(liquidity_providers):
        pool_id = pool_ids[i % len(pool_ids)]
        token_amounts = {staking_system.liquidity_pools[pool_id].token_pair[0]: 10000.0 + i * 5000.0,
                        staking_system.liquidity_pools[pool_id].token_pair[1]: 5000.0 + i * 2500.0}
        
        success = staking_system.add_liquidity(provider, pool_id, token_amounts)
        if success:
            print(f"   {provider} added liquidity to {pool_id}: {token_amounts}")
        else:
            print(f"   Failed to add liquidity for {provider}")
    
    # Show updated system metrics
    print("\n📊 Updated System Metrics:")
    metrics = staking_system.get_system_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    print(f"      {sub_key}: {sub_value:,.2f}")
                else:
                    print(f"      {sub_key}: {sub_value}")
        elif isinstance(value, float):
            print(f"   {key}: {value:,.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Calculate rewards
    print("\n💰 Calculating Rewards:")
    rewards = staking_system.calculate_rewards(time_period_hours=24*7)  # 1 week
    total_reward_value = sum(sum(token_rewards.values()) for token_rewards in rewards.values())
    print(f"   Total rewards calculated: {total_reward_value:,.2f} tokens")
    
    # Show top reward recipients
    print("   Top reward recipients:")
    sorted_recipients = sorted(rewards.items(), key=lambda x: sum(x[1].values()), reverse=True)
    for address, token_rewards in sorted_recipients[:5]:
        total_rewards = sum(token_rewards.values())
        print(f"      {address}: {total_rewards:.2f} tokens")
        for token_type, amount in token_rewards.items():
            print(f"         {token_type}: {amount:.2f}")
    
    # Claim rewards
    print("\n💳 Claiming Rewards:")
    for address in staker_addresses[:2]:
        claimed = staking_system.claim_rewards(address)
        if claimed:
            total_claimed = sum(claimed.values())
            print(f"   {address} claimed {total_claimed:.2f} tokens:")
            for token_type, amount in claimed.items():
                print(f"      {token_type}: {amount:.2f}")
        else:
            print(f"   {address} had no rewards to claim")
    
    # Show validator information
    print("\n🏛️  Validator Information:")
    for validator_id in validator_ids[:2]:
        info = staking_system.get_validator_info(validator_id)
        if info:
            print(f"   {validator_id}:")
            print(f"      CHR Score: {info['chr_score']:.2f}")
            print(f"      Total Staked: {info['total_staked']}")
            print(f"      Total Delegated: {info['total_delegated']}")
            print(f"      Active Stakes: {info['active_stakes']}")
            print(f"      Active Delegations: {info['active_delegations']}")
    
    print("\n✅ Validator staking system demo completed!")

if __name__ == "__main__":
    demo_validator_staking()