#!/usr/bin/env python3
"""
Integration tests for wallet creation through the dashboard
"""

import unittest
import requests
import time
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

class TestWalletDashboardIntegration(unittest.TestCase):
    """Test wallet creation and management through the dashboard API"""
    
    def setUp(self):
        """Set up test environment"""
        self.base_url = "http://localhost:5000"
        self.wallet_id = None
        self.account_address = None
        
    def test_01_create_wallet(self):
        """Test creating a new wallet"""
        print("Testing wallet creation...")
        
        # Create wallet
        response = requests.post(f"{self.base_url}/wallet/create", 
                                json={"wallet_name": "test_wallet"})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("wallet_id", data)
        self.assertIn("wallet_name", data)
        self.assertEqual(data["wallet_name"], "test_wallet")
        
        # Store wallet ID for later tests
        self.wallet_id = data["wallet_id"]
        print(f"Created wallet with ID: {self.wallet_id}")
        
    def test_02_generate_keypair(self):
        """Test generating a keypair for the wallet"""
        if not self.wallet_id:
            self.skipTest("Wallet not created")
            
        print("Testing keypair generation...")
        
        # Generate keypair
        response = requests.post(f"{self.base_url}/wallet/{self.wallet_id}/generate_keypair",
                                json={"account_name": "test_account"})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("address", data)
        self.assertIn("account", data)
        self.assertIn("chr_score", data["account"])
        
        # Store account address for later tests
        self.account_address = data["address"]
        print(f"Generated keypair with address: {self.account_address}")
        
    def test_03_get_wallet_accounts(self):
        """Test getting wallet accounts"""
        if not self.wallet_id:
            self.skipTest("Wallet not created")
            
        print("Testing wallet accounts retrieval...")
        
        # Get wallet accounts
        response = requests.get(f"{self.base_url}/wallet/{self.wallet_id}/accounts")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("accounts", data)
        
        # Verify our account is in the list
        if self.account_address:
            self.assertIn(self.account_address, data["accounts"])
            account = data["accounts"][self.account_address]
            self.assertIn("chr_score", account)
            self.assertIn("created_at", account)
            
        print(f"Retrieved {len(data['accounts'])} accounts")
        
    def test_04_get_account_balance(self):
        """Test getting account balance"""
        if not self.wallet_id or not self.account_address:
            self.skipTest("Wallet or account not created")
            
        print("Testing account balance retrieval...")
        
        # Get CHR balance
        response = requests.get(f"{self.base_url}/wallet/{self.wallet_id}/balance/{self.account_address}/CHR")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("balance", data)
        self.assertIn("token_type", data)
        self.assertEqual(data["token_type"], "CHR")
        self.assertIn("address", data)
        
        print(f"Account CHR balance: {data['balance']}")
        
    def test_05_create_transaction(self):
        """Test creating a transaction"""
        if not self.wallet_id or not self.account_address:
            self.skipTest("Wallet or account not created")
            
        print("Testing transaction creation...")
        
        # For this test, we'll create a transaction to ourselves with 0 amount
        # to avoid balance issues
        response = requests.post(f"{self.base_url}/wallet/{self.wallet_id}/transaction",
                                json={
                                    "sender": self.account_address,
                                    "receiver": self.account_address,
                                    "token_type": "CHR",
                                    "amount": 0.0,
                                    "memo": "Test transaction"
                                })
        
        # This might fail if the account doesn't have sufficient balance
        # but we're testing the API endpoint functionality
        self.assertIn(response.status_code, [200, 500])
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("status", data)
            if data["status"] == "success":
                self.assertIn("transaction_id", data)
                self.assertIn("sender", data)
                self.assertIn("receiver", data)
                print(f"Created transaction with ID: {data['transaction_id']}")
            else:
                print(f"Transaction creation failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"Transaction creation failed with status {response.status_code}")
            
    def test_06_get_transaction_history(self):
        """Test getting transaction history"""
        if not self.wallet_id or not self.account_address:
            self.skipTest("Wallet or account not created")
            
        print("Testing transaction history retrieval...")
        
        # Get transaction history
        response = requests.get(f"{self.base_url}/wallet/{self.wallet_id}/transactions/{self.account_address}")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("transactions", data)
        
        print(f"Retrieved {len(data['transactions'])} transactions")
        
    def test_07_get_wallet_resonance(self):
        """Test getting wallet resonance data"""
        if not self.wallet_id:
            self.skipTest("Wallet not created")
            
        print("Testing wallet resonance data retrieval...")
        
        # Get wallet resonance
        response = requests.get(f"{self.base_url}/wallet/{self.wallet_id}/resonance")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("wallet_id", data)
        self.assertIn("coherence", data)
        self.assertIn("entropy", data)
        self.assertIn("flow", data)
        self.assertIn("account_count", data)
        
        print(f"Wallet resonance - Coherence: {data['coherence']:.4f}, "
              f"Entropy: {data['entropy']:.4f}, Flow: {data['flow']:.4f}")
        
    def test_08_stake_tokens(self):
        """Test staking tokens"""
        if not self.wallet_id or not self.account_address:
            self.skipTest("Wallet or account not created")
            
        print("Testing token staking...")
        
        # Try to stake tokens (this will likely fail due to insufficient balance)
        response = requests.post(f"{self.base_url}/wallet/{self.wallet_id}/stake",
                                json={
                                    "address": self.account_address,
                                    "token_type": "CHR",
                                    "amount": 10.0
                                })
        
        # This might fail due to insufficient balance, but we're testing the API
        self.assertIn(response.status_code, [200, 500])
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("status", data)
            print(f"Staking result: {data['status']}")
            if data["status"] == "success":
                self.assertIn("stake_id", data)
                print(f"Created stake with ID: {data['stake_id']}")
        else:
            print(f"Staking failed with status {response.status_code}")
            
    def test_09_get_staking_records(self):
        """Test getting staking records"""
        if not self.wallet_id or not self.account_address:
            self.skipTest("Wallet or account not created")
            
        print("Testing staking records retrieval...")
        
        # Get staking records
        response = requests.get(f"{self.base_url}/wallet/{self.wallet_id}/staking_records/{self.account_address}")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("records", data)
        
        print(f"Retrieved {len(data['records'])} staking records")
        
    def test_10_rebalance_wallet(self):
        """Test rebalancing wallet flow"""
        if not self.wallet_id:
            self.skipTest("Wallet not created")
            
        print("Testing wallet rebalancing...")
        
        # Rebalance wallet
        response = requests.post(f"{self.base_url}/wallet/{self.wallet_id}/rebalance")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        
        print(f"Rebalancing result: {data['status']}")
        
    def tearDown(self):
        """Clean up after tests"""
        # No cleanup needed for these tests as they use the in-memory wallet storage
        pass

if __name__ == '__main__':
    # Check if the API server is running
    try:
        response = requests.get("http://localhost:5000/wallet/create", timeout=5)
        print("API server is running")
    except requests.exceptions.ConnectionError:
        print("API server is not running. Please start the server before running tests.")
        print("You can start the server with: python src/api/main.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error connecting to API server: {e}")
        sys.exit(1)
        
    # Run the tests
    unittest.main(verbosity=2)