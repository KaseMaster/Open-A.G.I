#!/usr/bin/env python3
"""
Harmonic Wallet Application for Quantum Currency
Implements wallet with harmonic-validated keypair generation
"""

import sys
import os
import json
import time
import hashlib
import hmac
import secrets
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our existing modules
from openagi.harmonic_validation import compute_spectrum, compute_coherence_score, HarmonicSnapshot
from openagi.hardware_security import HardwareSecurityModule, ValidatorKey

@dataclass
class WalletKey:
    """Represents a wallet keypair"""
    wallet_id: str
    public_key: str
    private_key_handle: str  # Reference to secure storage
    created_at: float
    key_type: str = "Ed25519"  # or "Falcon-1024" for post-quantum
    # Harmonic validation data
    harmonic_signature: Optional[str] = None
    coherence_score: Optional[float] = None
    validation_timestamp: Optional[float] = None

@dataclass
class WalletAccount:
    """Represents a wallet account"""
    address: str
    chr_balance: float = 0.0
    flx_balance: float = 0.0
    psy_balance: float = 0.0
    atr_balance: float = 0.0
    res_balance: float = 0.0
    chr_score: float = 0.0  # Reputation score
    created_at: float = 0.0
    last_activity: float = 0.0

@dataclass
class Transaction:
    """Represents a wallet transaction"""
    tx_id: str
    sender: str
    receiver: str
    token_type: str
    amount: float
    timestamp: float
    signature: str
    fee: float = 0.0
    memo: Optional[str] = None
    status: str = "pending"  # pending, confirmed, failed

class HarmonicWallet:
    """
    Implements wallet application with harmonic-validated keypair generation
    """
    
    def __init__(self, wallet_name: str = "default-wallet"):
        self.wallet_name = wallet_name
        self.wallet_id = f"wallet-{int(time.time())}-{hashlib.md5(wallet_name.encode()).hexdigest()[:8]}"
        self.accounts: Dict[str, WalletAccount] = {}
        self.keys: Dict[str, WalletKey] = {}
        self.transactions: List[Transaction] = []
        self.hsm = HardwareSecurityModule(f"hsm-{self.wallet_id}")
        self.wallet_config = {
            "default_key_type": "Ed25519",
            "min_coherence_score": 0.1,  # Lowered for demo purposes
            "transaction_fee": 0.001,
            "max_pending_transactions": 100
        }
    
    def generate_harmonic_keypair(self, account_name: str = "default", 
                                 key_type: str = "Ed25519") -> Optional[str]:
        """
        Generate a new keypair with harmonic validation
        
        Args:
            account_name: Name for the account
            key_type: Type of cryptographic key to generate
            
        Returns:
            Address of the new account if successful, None otherwise
        """
        # Generate time series data for harmonic validation
        # In a real implementation, this would come from quantum sensors
        duration = 0.5  # 500ms
        sample_rate = 2048  # 2048 Hz
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Generate two highly coherent signals for validation
        # In a real implementation, these would come from quantum sensors
        freq = np.random.uniform(40, 60)  # Random frequency between 40-60 Hz
        
        # Generate identical signals for maximum coherence
        signal1 = np.sin(2 * np.pi * freq * t)
        signal2 = np.sin(2 * np.pi * freq * t)  # Identical signal
        
        # Add small noise to make it more realistic
        noise_level = 0.01
        signal1 += np.random.normal(0, noise_level, len(signal1))
        signal2 += np.random.normal(0, noise_level, len(signal2))
        
        # Compute spectra for the snapshots
        spectrum1 = compute_spectrum(t, signal1)
        spectrum2 = compute_spectrum(t, signal2)
        
        # Create spectrum hashes
        spectrum_hash1 = hashlib.sha256(str(spectrum1).encode()).hexdigest()[:32]
        spectrum_hash2 = hashlib.sha256(str(spectrum2).encode()).hexdigest()[:32]
        
        # Create phi parameters
        phi_params = {"phi": 1.618033988749895, "lambda": 0.618033988749895}
        
        # Create HarmonicSnapshot objects for coherence validation
        snapshot1 = HarmonicSnapshot(
            node_id=f"{self.wallet_id}-{account_name}-1",
            timestamp=time.time(),
            times=t.tolist(),
            values=signal1.tolist(),
            spectrum=spectrum1,
            spectrum_hash=spectrum_hash1,
            CS=0.0,  # Will be computed later
            phi_params=phi_params
        )
        
        snapshot2 = HarmonicSnapshot(
            node_id=f"{self.wallet_id}-{account_name}-2",
            timestamp=time.time(),
            times=t.tolist(),
            values=signal2.tolist(),
            spectrum=spectrum2,
            spectrum_hash=spectrum_hash2,
            CS=0.0,  # Will be computed later
            phi_params=phi_params
        )
        
        # Compute coherence score between the two snapshots
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        
        # Check if coherence score meets minimum requirement
        if coherence_score < self.wallet_config["min_coherence_score"]:
            print(f"Coherence score {coherence_score:.3f} below minimum {self.wallet_config['min_coherence_score']}")
            return None
        
        # Generate keypair using HSM
        validator_key = self.hsm.generate_validator_key(f"{self.wallet_id}-{account_name}", key_type)
        
        # Create wallet key with harmonic validation data
        wallet_key = WalletKey(
            wallet_id=validator_key.validator_id,
            public_key=validator_key.public_key,
            private_key_handle=validator_key.private_key_handle,
            created_at=time.time(),
            key_type=key_type,
            harmonic_signature=validator_key.attestation_cert,  # Reuse attestation as signature
            coherence_score=coherence_score,
            validation_timestamp=time.time()
        )
        
        self.keys[validator_key.validator_id] = wallet_key
        
        # Create account
        account = WalletAccount(
            address=validator_key.validator_id,
            chr_score=coherence_score,  # Use coherence score as initial CHR score
            created_at=time.time(),
            last_activity=time.time()
        )
        
        self.accounts[validator_key.validator_id] = account
        
        return validator_key.validator_id
    
    def get_account(self, address: str) -> Optional[WalletAccount]:
        """
        Get account information
        
        Args:
            address: Account address
            
        Returns:
            WalletAccount if found, None otherwise
        """
        return self.accounts.get(address)
    
    def get_balance(self, address: str, token_type: str = "FLX") -> float:
        """
        Get token balance for an account
        
        Args:
            address: Account address
            token_type: Type of token
            
        Returns:
            Balance amount
        """
        if address not in self.accounts:
            return 0.0
        
        account = self.accounts[address]
        balance_attr = f"{token_type.lower()}_balance"
        
        if hasattr(account, balance_attr):
            return getattr(account, balance_attr)
        
        return 0.0
    
    def create_transaction(self, sender: str, receiver: str, token_type: str, 
                          amount: float, memo: Optional[str] = None) -> Optional[str]:
        """
        Create a new transaction
        
        Args:
            sender: Sender address
            receiver: Receiver address
            token_type: Type of token to transfer
            amount: Amount to transfer
            memo: Optional memo for the transaction
            
        Returns:
            Transaction ID if successful, None otherwise
        """
        # Validate sender
        if sender not in self.accounts:
            print("Sender account not found")
            return None
        
        # Validate receiver
        if receiver not in self.accounts:
            print("Receiver account not found")
            return None
        
        # Validate sender has sufficient balance
        sender_account = self.accounts[sender]
        balance = self.get_balance(sender, token_type)
        
        if balance < amount:
            print(f"Insufficient {token_type} balance: {balance} < {amount}")
            return None
        
        # Check pending transactions limit
        pending_txs = [tx for tx in self.transactions if tx.status == "pending"]
        if len(pending_txs) >= self.wallet_config["max_pending_transactions"]:
            print("Too many pending transactions")
            return None
        
        # Calculate fee
        fee = amount * self.wallet_config["transaction_fee"]
        total_amount = amount + fee
        
        # Check if sender has sufficient balance for amount + fee
        if balance < total_amount:
            print(f"Insufficient balance for amount + fee: {balance} < {total_amount}")
            return None
        
        # Create transaction ID
        tx_data = f"{sender}{receiver}{token_type}{amount}{time.time()}"
        tx_id = hashlib.sha256(tx_data.encode()).hexdigest()[:32]
        
        # Sign transaction using HSM
        signature = self.hsm.sign_data(sender, tx_data)
        if not signature:
            print("Failed to sign transaction")
            return None
        
        # Create transaction
        transaction = Transaction(
            tx_id=tx_id,
            sender=sender,
            receiver=receiver,
            token_type=token_type,
            amount=amount,
            timestamp=time.time(),
            signature=signature,
            fee=fee,
            memo=memo
        )
        
        self.transactions.append(transaction)
        
        # Update account balances
        sender_balance_attr = f"{token_type.lower()}_balance"
        receiver_balance_attr = f"{token_type.lower()}_balance"
        
        if hasattr(sender_account, sender_balance_attr):
            current_balance = getattr(sender_account, sender_balance_attr)
            setattr(sender_account, sender_balance_attr, current_balance - total_amount)
        
        receiver_account = self.accounts[receiver]
        if hasattr(receiver_account, receiver_balance_attr):
            current_balance = getattr(receiver_account, receiver_balance_attr)
            setattr(receiver_account, receiver_balance_attr, current_balance + amount)
        
        # Update last activity
        sender_account.last_activity = time.time()
        receiver_account.last_activity = time.time()
        
        return tx_id
    
    def sign_message(self, address: str, message: str) -> Optional[str]:
        """
        Sign a message with a wallet key
        
        Args:
            address: Wallet address
            message: Message to sign
            
        Returns:
            Signature if successful, None otherwise
        """
        if address not in self.keys:
            print("Address not found in wallet")
            return None
        
        # Sign message using HSM
        signature = self.hsm.sign_data(address, message)
        return signature
    
    def verify_signature(self, address: str, message: str, signature: str) -> bool:
        """
        Verify a signature
        
        Args:
            address: Wallet address
            message: Message that was signed
            signature: Signature to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        if address not in self.keys:
            print("Address not found in wallet")
            return False
        
        wallet_key = self.keys[address]
        return self.hsm.verify_signature(wallet_key.public_key, message, signature)
    
    def get_transaction_history(self, address: str, limit: int = 10) -> List[Transaction]:
        """
        Get transaction history for an account
        
        Args:
            address: Account address
            limit: Maximum number of transactions to return
            
        Returns:
            List of transactions
        """
        # Filter transactions for this address (sender or receiver)
        account_txs = [tx for tx in self.transactions 
                      if tx.sender == address or tx.receiver == address]
        
        # Sort by timestamp (newest first)
        account_txs.sort(key=lambda tx: tx.timestamp, reverse=True)
        
        return account_txs[:limit]
    
    def get_wallet_info(self) -> Dict:
        """
        Get wallet information
        
        Returns:
            Dictionary with wallet information
        """
        total_accounts = len(self.accounts)
        total_transactions = len(self.transactions)
        pending_transactions = len([tx for tx in self.transactions if tx.status == "pending"])
        
        # Calculate total balances
        total_balances = {}
        for account in self.accounts.values():
            for token in ["FLX", "CHR", "PSY", "ATR", "RES"]:
                balance_attr = f"{token.lower()}_balance"
                if hasattr(account, balance_attr):
                    total_balances[token] = total_balances.get(token, 0.0) + getattr(account, balance_attr)
        
        return {
            "wallet_name": self.wallet_name,
            "wallet_id": self.wallet_id,
            "total_accounts": total_accounts,
            "total_transactions": total_transactions,
            "pending_transactions": pending_transactions,
            "total_balances": total_balances,
            "key_count": len(self.keys)
        }
    
    def backup_wallet(self, backup_path: str) -> bool:
        """
        Backup wallet data (public keys and account info only)
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Only backup public information
            backup_data = {
                "wallet_name": self.wallet_name,
                "wallet_id": self.wallet_id,
                "accounts": {addr: asdict(account) for addr, account in self.accounts.items()},
                "keys": {addr: {
                    "wallet_id": key.wallet_id,
                    "public_key": key.public_key,
                    "created_at": key.created_at,
                    "key_type": key.key_type,
                    "coherence_score": key.coherence_score,
                    "validation_timestamp": key.validation_timestamp
                } for addr, key in self.keys.items()},
                "transactions": [asdict(tx) for tx in self.transactions],
                "backup_timestamp": time.time()
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to backup wallet: {e}")
            return False
    
    def restore_wallet(self, backup_path: str) -> bool:
        """
        Restore wallet from backup
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Restore wallet data
            self.wallet_name = backup_data.get("wallet_name", self.wallet_name)
            self.wallet_id = backup_data.get("wallet_id", self.wallet_id)
            
            # Restore accounts
            accounts_data = backup_data.get("accounts", {})
            for addr, account_dict in accounts_data.items():
                account = WalletAccount(**account_dict)
                self.accounts[addr] = account
            
            # Restore keys (public info only)
            keys_data = backup_data.get("keys", {})
            for addr, key_dict in keys_data.items():
                # Note: private keys are not restored for security
                key = WalletKey(
                    wallet_id=key_dict["wallet_id"],
                    public_key=key_dict["public_key"],
                    private_key_handle="",  # Not restored for security
                    created_at=key_dict["created_at"],
                    key_type=key_dict["key_type"],
                    coherence_score=key_dict.get("coherence_score"),
                    validation_timestamp=key_dict.get("validation_timestamp")
                )
                self.keys[addr] = key
            
            # Restore transactions
            transactions_data = backup_data.get("transactions", [])
            for tx_dict in transactions_data:
                tx = Transaction(**tx_dict)
                self.transactions.append(tx)
            
            return True
        except Exception as e:
            print(f"Failed to restore wallet: {e}")
            return False

def demo_harmonic_wallet():
    """Demonstrate harmonic wallet capabilities"""
    print("💼 Harmonic Wallet Demo")
    print("=" * 25)
    
    # Create wallet instance
    wallet = HarmonicWallet("My Quantum Wallet")
    
    # Show initial wallet info
    print("\n💼 Wallet Information:")
    info = wallet.get_wallet_info()
    print(f"   Name: {info['wallet_name']}")
    print(f"   ID: {info['wallet_id']}")
    print(f"   Accounts: {info['total_accounts']}")
    print(f"   Transactions: {info['total_transactions']}")
    
    # Generate keypairs with harmonic validation
    print("\n🔐 Generating Harmonic Keypairs:")
    accounts = []
    
    for i in range(3):
        account_name = f"account-{i+1}"
        address = wallet.generate_harmonic_keypair(account_name)
        
        if address:
            accounts.append(address)
            account = wallet.get_account(address)
            key = wallet.keys[address]
            print(f"   {account_name}: {address[:16]}...")
            if account:
                print(f"      CHR Score: {account.chr_score:.3f}")
            if key:
                print(f"      Coherence: {key.coherence_score:.3f}")
            print(f"      Key Type: {key.key_type}")
        else:
            print(f"   Failed to generate keypair for {account_name}")
    
    # Show account balances
    print("\n💰 Account Balances:")
    for address in accounts:
        account = wallet.get_account(address)
        if account:
            print(f"   {address[:16]}...:")
            print(f"      FLX: {account.flx_balance:.2f}")
            print(f"      CHR: {account.chr_balance:.2f}")
            print(f"      PSY: {account.psy_balance:.2f}")
            print(f"      ATR: {account.atr_balance:.2f}")
            print(f"      RES: {account.res_balance:.2f}")
    
    # Simulate adding some initial balances
    print("\n💸 Adding Initial Balances:")
    if len(accounts) >= 2:
        sender = accounts[0]
        receiver = accounts[1]
        
        # Add some FLX to sender
        sender_account = wallet.get_account(sender)
        if sender_account:
            sender_account.flx_balance = 1000.0
            sender_account.chr_balance = 500.0
        
            print(f"   Added 1000.00 FLX to {sender[:16]}...")
            print(f"   Added 500.00 CHR to {sender[:16]}...")
    
    # Create transactions
    print("\n💱 Creating Transactions:")
    if len(accounts) >= 2:
        sender = accounts[0]
        receiver = accounts[1]
        
        # Create a transaction
        tx_id = wallet.create_transaction(
            sender=sender,
            receiver=receiver,
            token_type="FLX",
            amount=100.0,
            memo="Test transaction"
        )
        
        if tx_id:
            print(f"   Transaction created: {tx_id}")
            print(f"   Amount: 100.00 FLX")
            print(f"   Fee: 0.10 FLX")
        else:
            print("   Failed to create transaction")
        
        # Show updated balances
        sender_balance = wallet.get_balance(sender, "FLX")
        receiver_balance = wallet.get_balance(receiver, "FLX")
        print(f"   Sender FLX balance: {sender_balance:.2f}")
        print(f"   Receiver FLX balance: {receiver_balance:.2f}")
    
    # Sign and verify a message
    print("\n📝 Signing and Verifying Messages:")
    if accounts:
        address = accounts[0]
        message = "Hello, Quantum World!"
        
        signature = wallet.sign_message(address, message)
        if signature:
            print(f"   Message signed: {message}")
            print(f"   Signature: {signature[:16]}...")
            
            # Verify signature
            is_valid = wallet.verify_signature(address, message, signature)
            print(f"   Signature valid: {is_valid}")
        else:
            print("   Failed to sign message")
    
    # Show transaction history
    print("\n📜 Transaction History:")
    if accounts:
        address = accounts[0]
        history = wallet.get_transaction_history(address, limit=5)
        print(f"   Recent transactions for {address[:16]}...:")
        for tx in history:
            print(f"      {tx.tx_id[:8]}...: {tx.amount:.2f} {tx.token_type} to {tx.receiver[:8]}...")
    
    # Show final wallet info
    print("\n💼 Final Wallet Information:")
    info = wallet.get_wallet_info()
    print(f"   Accounts: {info['total_accounts']}")
    print(f"   Transactions: {info['total_transactions']}")
    print(f"   Pending: {info['pending_transactions']}")
    
    for token, balance in info['total_balances'].items():
        print(f"   Total {token}: {balance:.2f}")
    
    # Backup wallet
    print("\n💾 Backing Up Wallet:")
    backup_path = "wallet_backup.json"
    success = wallet.backup_wallet(backup_path)
    if success:
        print(f"   Wallet backed up to {backup_path}")
    else:
        print("   Failed to backup wallet")
    
    # Restore wallet (to demonstrate)
    print("\n🔄 Restoring Wallet:")
    new_wallet = HarmonicWallet("Restored Wallet")
    success = new_wallet.restore_wallet(backup_path)
    if success:
        print("   Wallet restored successfully")
        restored_info = new_wallet.get_wallet_info()
        print(f"   Restored accounts: {restored_info['total_accounts']}")
    else:
        print("   Failed to restore wallet")
    
    print("\n✅ Harmonic wallet demo completed!")

if __name__ == "__main__":
    demo_harmonic_wallet()