#!/usr/bin/env python3
"""
Demo script for wallet creation and management
"""

import requests
import json
import time
import sys
import os

def demo_wallet_creation():
    """Demonstrate wallet creation and management"""
    print("💼 Quantum Currency Wallet Creation Demo")
    print("=" * 45)
    
    base_url = "http://localhost:5000"
    
    try:
        # Test connection to API server
        response = requests.get(f"{base_url}/", timeout=5)
        print("✅ Connected to API server")
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running. Please start the server before running this demo.")
        print("   You can start the server with: python src/api/main.py")
        return
    except Exception as e:
        print(f"❌ Error connecting to API server: {e}")
        return
    
    wallet_id = None
    account_address = None
    
    try:
        # 1. Create a new wallet
        print("\n1. Creating a new wallet...")
        response = requests.post(f"{base_url}/wallet/create", 
                                json={"wallet_name": "Demo Wallet"})
        
        if response.status_code == 200:
            data = response.json()
            wallet_id = data["wallet_id"]
            print(f"✅ Wallet created successfully!")
            print(f"   Wallet ID: {wallet_id}")
            print(f"   Wallet Name: {data['wallet_name']}")
        else:
            print(f"❌ Failed to create wallet: {response.status_code}")
            return
            
        # 2. Generate a keypair
        print("\n2. Generating a keypair...")
        response = requests.post(f"{base_url}/wallet/{wallet_id}/generate_keypair",
                                json={"account_name": "demo_account"})
        
        if response.status_code == 200:
            data = response.json()
            account_address = data["address"]
            print(f"✅ Keypair generated successfully!")
            print(f"   Account Address: {account_address}")
            print(f"   CHR Score: {data['account']['chr_score']:.4f}")
        else:
            print(f"❌ Failed to generate keypair: {response.status_code}")
            return
            
        # 3. Get wallet accounts
        print("\n3. Retrieving wallet accounts...")
        response = requests.get(f"{base_url}/wallet/{wallet_id}/accounts")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Retrieved {len(data['accounts'])} account(s)")
            for addr, account in data['accounts'].items():
                print(f"   Address: {addr[:16]}...")
                print(f"   CHR Score: {account['chr_score']:.4f}")
                print(f"   Created: {time.ctime(account['created_at'])}")
        else:
            print(f"❌ Failed to retrieve accounts: {response.status_code}")
            
        # 4. Get account balance
        print("\n4. Checking account balance...")
        response = requests.get(f"{base_url}/wallet/{wallet_id}/balance/{account_address}/CHR")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Balance retrieved successfully!")
            print(f"   Token Type: {data['token_type']}")
            print(f"   Balance: {data['balance']:.4f}")
        else:
            print(f"❌ Failed to retrieve balance: {response.status_code}")
            
        # 5. Get wallet resonance
        print("\n5. Checking wallet resonance...")
        response = requests.get(f"{base_url}/wallet/{wallet_id}/resonance")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Resonance data retrieved successfully!")
            print(f"   Coherence: {data['coherence']:.4f}")
            print(f"   Entropy: {data['entropy']:.4f}")
            print(f"   Flow: {data['flow']:.4f}")
            print(f"   Account Count: {data['account_count']}")
        else:
            print(f"❌ Failed to retrieve resonance data: {response.status_code}")
            
        # 6. Get transaction history
        print("\n6. Retrieving transaction history...")
        response = requests.get(f"{base_url}/wallet/{wallet_id}/transactions/{account_address}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Retrieved {len(data['transactions'])} transaction(s)")
        else:
            print(f"❌ Failed to retrieve transaction history: {response.status_code}")
            
        # 7. Get staking records
        print("\n7. Retrieving staking records...")
        response = requests.get(f"{base_url}/wallet/{wallet_id}/staking_records/{account_address}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Retrieved {len(data['records'])} staking record(s)")
        else:
            print(f"❌ Failed to retrieve staking records: {response.status_code}")
            
        print("\n🎉 Wallet creation and management demo completed successfully!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
if __name__ == "__main__":
    demo_wallet_creation()