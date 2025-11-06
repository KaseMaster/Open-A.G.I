#!/usr/bin/env python3
"""
Validator Staking System for Quantum Currency
Implements validator staking, delegation, and liquidity incentives
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
    token_type: str  # "FLX" or "CHR"
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
    total_staked: float = 0.0
    total_delegated: float = 0.0
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
    Implements validator staking, delegation, and liquidity incentives
    """
    
    def __init__(self):
        self.staking_positions: Dict[str, StakingPosition] = {}
        self.delegations: Dict[str, Delegation] = {}
        self.validators: Dict[str, Validator] = {}
        self.liquidity_pools: Dict[str, LiquidityPool] = {}
        self.pending_rewards: Dict[str, float] = {}  # {address: amount}
        self.staking_config = {
            "min_stake_amount": 1000.0,
            "max_lockup_period": 365.0,  # days
            "base_reward_rate": 0.12,  # 12% APR
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
                commission_rate=self.staking_config["commission_rate"]
            )
            self.validators[validator_id] = validator
    
    def _initialize_liquidity_pools(self):
        """Initialize liquidity pools"""
        pools_data = [
            ("pool-flx-chr", ("FLX", "CHR"), {"FLX": 100000.0, "CHR": 50000.0}),
            ("pool-flx-psy", ("FLX", "PSY"), {"FLX": 75000.0, "PSY": 37500.0}),
            ("pool-atr-res", ("ATR", "RES"), {"ATR": 25000.0, "RES": 12500.0})
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
            token_type: Type of token to stake ("FLX" or "CHR")
            amount: Amount to stake
            lockup_period: Lockup period in days
            
        Returns:
            Position ID if successful, None otherwise
        """
        # Validate inputs
        if amount < self.staking_config["min_stake_amount"]:
            print(f"Staking amount {amount} is below minimum {self.staking_config['min_stake_amount']}")
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
        
        # Create staking position
        position_id = f"stake-{int(time.time())}-{hashlib.md5(staker_address.encode()).hexdigest()[:8]}"
        
        # Calculate reward rate based on validator CHR score and lockup period
        base_rate = self.staking_config["base_reward_rate"]
        chr_bonus = validator.chr_score * 0.05  # Up to 5% bonus for high CHR score
        lockup_bonus = min(lockup_period / 365.0 * 0.03, 0.03)  # Up to 3% bonus for long lockup
        reward_rate = base_rate + chr_bonus + lockup_bonus
        
        position = StakingPosition(
            position_id=position_id,
            staker_address=staker_address,
            validator_id=validator_id,
            token_type=token_type,
            amount=amount,
            staked_at=time.time(),
            lockup_period=lockup_period,
            reward_rate=reward_rate
        )
        
        self.staking_positions[position_id] = position
        
        # Update validator's total staked
        validator.total_staked += amount
        
        return position_id
    
    def delegate_tokens(self, delegator_address: str, validator_id: str, 
                       amount: float, reward_share: float = 0.5) -> Optional[str]:
        """
        Delegate tokens to a validator
        
        Args:
            delegator_address: Address of the delegator
            validator_id: ID of the validator to delegate to
            amount: Amount to delegate
            reward_share: Percentage of rewards to share with delegator (0.0 to 1.0)
            
        Returns:
            Delegation ID if successful, None otherwise
        """
        # Validate inputs
        if amount < self.staking_config["min_stake_amount"] / 10:  # Lower minimum for delegation
            print(f"Delegation amount {amount} is below minimum {self.staking_config['min_stake_amount']/10}")
            return None
        
        if not (0.0 <= reward_share <= 1.0):
            print("Reward share must be between 0.0 and 1.0")
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
            amount=amount,
            delegated_at=time.time(),
            reward_share=reward_share
        )
        
        self.delegations[delegation_id] = delegation
        
        # Update validator's total delegated
        validator.total_delegated += amount
        
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
        # In a real implementation, this would be more sophisticated
        reward_amount = sum(token_amounts.values()) * 0.001  # 0.1% incentive
        self.pending_rewards[provider_address] = self.pending_rewards.get(provider_address, 0.0) + reward_amount
        
        return True
    
    def calculate_rewards(self, time_period_hours: float = 24.0) -> Dict[str, float]:
        """
        Calculate rewards for stakers, delegators, and liquidity providers
        
        Args:
            time_period_hours: Time period to calculate rewards for (default 24 hours)
            
        Returns:
            Dictionary of {address: reward_amount}
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
            
            # Add to validator's pending rewards
            validator = self.validators[position.validator_id]
            validator_reward = reward_amount * validator.commission_rate
            staker_reward = reward_amount * (1 - validator.commission_rate)
            
            # Add rewards to respective addresses
            rewards[validator.operator_address] = rewards.get(validator.operator_address, 0.0) + validator_reward
            rewards[position.staker_address] = rewards.get(position.staker_address, 0.0) + staker_reward
            
            # Update claimed rewards
            position.claimed_rewards += staker_reward
        
        # Calculate delegation rewards
        for delegation_id, delegation in self.delegations.items():
            if not delegation.is_active:
                continue
            
            # Get validator
            validator = self.validators[delegation.validator_id]
            
            # Calculate base reward (based on validator's total stake)
            base_reward = validator.total_staked * self.staking_config["base_reward_rate"] * time_factor
            
            # Calculate delegator's share
            delegation_share = base_reward * delegation.reward_share
            
            # Add to delegator's rewards
            rewards[delegation.delegator_address] = rewards.get(delegation.delegator_address, 0.0) + delegation_share
            
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
                rewards["liquidity_providers"] = rewards.get("liquidity_providers", 0.0) + pool_reward
        
        return rewards
    
    def claim_rewards(self, address: str) -> float:
        """
        Claim pending rewards for an address
        
        Args:
            address: Address to claim rewards for
            
        Returns:
            Amount of rewards claimed
        """
        if address not in self.pending_rewards:
            return 0.0
        
        reward_amount = self.pending_rewards[address]
        self.pending_rewards[address] = 0.0
        
        return reward_amount
    
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
        
        # Update validator's total staked
        validator = self.validators[position.validator_id]
        validator.total_staked -= position.amount
        
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
            "total_staked_and_delegated": validator.total_staked + validator.total_delegated,
            "commission_rate": validator.commission_rate,
            "uptime": validator.uptime,
            "is_active": validator.is_active,
            "active_stakes": len(active_stakes),
            "active_delegations": len(active_delegations)
        }
    
    def get_staking_apr(self, validator_id: str) -> Optional[float]:
        """
        Get the APR for staking with a validator
        
        Args:
            validator_id: ID of the validator
            
        Returns:
            APR as a decimal (0.12 = 12%) or None if validator not found
        """
        if validator_id not in self.validators:
            return None
        
        validator = self.validators[validator_id]
        
        # Base rate plus CHR score bonus
        base_rate = self.staking_config["base_reward_rate"]
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
        
        # Calculate total staked value
        total_staked = sum(pos.amount for pos in active_stakes)
        total_delegated = sum(delg.amount for delg in active_delegations)
        
        # Calculate total liquidity
        total_liquidity = sum(pool.total_liquidity for pool in self.liquidity_pools.values())
        
        return {
            "total_validators": len(self.validators),
            "active_validators": len([v for v in self.validators.values() if v.is_active]),
            "total_staking_positions": len(self.staking_positions),
            "active_staking_positions": len(active_stakes),
            "total_delegations": len(self.delegations),
            "active_delegations": len(active_delegations),
            "total_staked": total_staked,
            "total_delegated": total_delegated,
            "total_staked_and_delegated": total_staked + total_delegated,
            "total_liquidity_pools": len(self.liquidity_pools),
            "total_liquidity": total_liquidity,
            "pending_rewards": sum(self.pending_rewards.values())
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
        if isinstance(value, float):
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
    
    # Create staking positions
    print("\n🔒 Creating Staking Positions:")
    staker_addresses = ["staker-001", "staker-002", "staker-003"]
    validator_ids = list(staking_system.validators.keys())
    
    for i, staker in enumerate(staker_addresses):
        validator_id = validator_ids[i % len(validator_ids)]
        amount = 5000.0 + i * 1000.0
        lockup_period = 90.0 + i * 30.0  # 90, 120, 150 days
        
        position_id = staking_system.create_staking_position(
            staker_address=staker,
            validator_id=validator_id,
            token_type="FLX",
            amount=amount,
            lockup_period=lockup_period
        )
        
        if position_id:
            apr = staking_system.get_staking_apr(validator_id)
            print(f"   {staker} staked {amount:.2f} FLX with {validator_id} (APR: {apr:.1%})")
        else:
            print(f"   Failed to create staking position for {staker}")
    
    # Create delegations
    print("\n🤝 Creating Delegations:")
    delegator_addresses = ["delegator-001", "delegator-002", "delegator-003"]
    
    for i, delegator in enumerate(delegator_addresses):
        validator_id = validator_ids[i % len(validator_ids)]
        amount = 2000.0 + i * 500.0
        reward_share = 0.6 + i * 0.1  # 60%, 70%, 80%
        
        delegation_id = staking_system.delegate_tokens(
            delegator_address=delegator,
            validator_id=validator_id,
            amount=amount,
            reward_share=reward_share
        )
        
        if delegation_id:
            print(f"   {delegator} delegated {amount:.2f} tokens to {validator_id} ({reward_share:.0%} reward share)")
        else:
            print(f"   Failed to create delegation for {delegator}")
    
    # Add liquidity to pools
    print("\n💧 Adding Liquidity:")
    liquidity_providers = ["lp-001", "lp-002"]
    pool_ids = list(staking_system.liquidity_pools.keys())
    
    for i, provider in enumerate(liquidity_providers):
        pool_id = pool_ids[i % len(pool_ids)]
        token_amounts = {"FLX": 10000.0 + i * 5000.0, "CHR": 5000.0 + i * 2500.0}
        
        success = staking_system.add_liquidity(provider, pool_id, token_amounts)
        if success:
            print(f"   {provider} added liquidity to {pool_id}: {token_amounts}")
        else:
            print(f"   Failed to add liquidity for {provider}")
    
    # Show updated system metrics
    print("\n📊 Updated System Metrics:")
    metrics = staking_system.get_system_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:,.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Calculate rewards
    print("\n💰 Calculating Rewards:")
    rewards = staking_system.calculate_rewards(time_period_hours=24*7)  # 1 week
    total_rewards = sum(rewards.values())
    print(f"   Total rewards calculated: {total_rewards:,.2f} tokens")
    
    # Show top reward recipients
    sorted_rewards = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
    print("   Top reward recipients:")
    for address, reward in sorted_rewards[:5]:
        print(f"      {address}: {reward:.2f} tokens")
    
    # Claim rewards
    print("\n💳 Claiming Rewards:")
    for address in staker_addresses[:2]:
        claimed = staking_system.claim_rewards(address)
        if claimed > 0:
            print(f"   {address} claimed {claimed:.2f} tokens")
        else:
            print(f"   {address} had no rewards to claim")
    
    # Show validator information
    print("\n🏛️  Validator Information:")
    for validator_id in validator_ids[:2]:
        info = staking_system.get_validator_info(validator_id)
        if info:
            print(f"   {validator_id}:")
            print(f"      Total Staked: {info['total_staked']:,.2f}")
            print(f"      Total Delegated: {info['total_delegated']:,.2f}")
            print(f"      Active Stakes: {info['active_stakes']}")
            print(f"      Active Delegations: {info['active_delegations']}")
    
    print("\n✅ Validator staking system demo completed!")

if __name__ == "__main__":
    demo_validator_staking()