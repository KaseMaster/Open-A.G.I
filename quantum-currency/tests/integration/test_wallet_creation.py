#!/usr/bin/env python3
"""
Test script to verify wallet creation through the dashboard
"""

import requests
import json
import time
import sys
import os

def test_wallet_creation():
    """Test wallet creation functionality"""
    print("Testing wallet creation through dashboard API...")
    
    base_url = "http://localhost:5000"
    
    try:
        # Test 1: Create a new wallet
        print("\n1. Creating a new wallet...")
        response = requests.post(f"{base_url}/wallet/create", 
                                json={"wallet_name": "Test Wallet"})
        
        if response.status_code == 200:
            data = response.json()
            wallet_id = data["wallet_id"]
            print(f"✅ Wallet created successfully!")
            print(f"   Wallet ID: {wallet_id}")
            print(f"   Wallet Name: {data['wallet_name']}")
        else:
            print(f"❌ Failed to create wallet: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
        # Test 2: Generate a keypair
        print("\n2. Generating a keypair...")
        response = requests.post(f"{base_url}/wallet/{wallet_id}/generate_keypair",
                                json={"account_name": "test_account"})
        
        if response.status_code == 200:
            data = response.json()
            account_address = data["address"]
            print(f"✅ Keypair generated successfully!")
            print(f"   Account Address: {account_address}")
            print(f"   CHR Score: {data['account']['chr_score']:.4f}")
        else:
            print(f"❌ Failed to generate keypair: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
        # Test 3: Get wallet accounts
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
            print(f"   Response: {response.text}")
            return False
            
        # Test 4: Get account balance
        print("\n4. Checking account balance...")
        response = requests.get(f"{base_url}/wallet/{wallet_id}/balance/{account_address}/CHR")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Balance retrieved successfully!")
            print(f"   Token Type: {data['token_type']}")
            print(f"   Balance: {data['balance']:.4f}")
        else:
            print(f"❌ Failed to retrieve balance: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
        print("\n🎉 All wallet creation tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running. Please start the server before running tests.")
        print("   You can start the server with: python src/api/main.py")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_wallet_creation()
    sys.exit(0 if success else 1)