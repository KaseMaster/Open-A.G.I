"""
Test script for CertifiedMath API Integration
"""
import asyncio
import httpx
import json
from datetime import datetime

async def test_certified_math_api():
    """Test the CertifiedMath API endpoints"""
    base_url = "http://localhost:8001"
    api_key = "aegis_admin"  # Using admin key for testing
    
    async with httpx.AsyncClient() as client:
        # Test health check
        print("Testing health check...")
        response = await client.get(f"{base_url}/health")
        print(f"Health check response: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
        
        # Test addition operation
        print("Testing addition operation...")
        add_request = {
            "operand_a": "123456789012345678901234567890",
            "operand_b": "987654321098765432109876543210",
            "pqc_cid": "test_pqc_cid_123"
        }
        
        response = await client.post(
            f"{base_url}/api/v1/certified-math/add",
            json=add_request,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        print(f"Addition response: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
        
        # Test subtraction operation
        print("Testing subtraction operation...")
        sub_request = {
            "operand_a": "987654321098765432109876543210",
            "operand_b": "123456789012345678901234567890",
            "pqc_cid": "test_pqc_cid_456"
        }
        
        response = await client.post(
            f"{base_url}/api/v1/certified-math/sub",
            json=sub_request,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        print(f"Subtraction response: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
        
        # Test multiplication operation
        print("Testing multiplication operation...")
        mul_request = {
            "operand_a": "123456789",
            "operand_b": "987654321",
            "pqc_cid": "test_pqc_cid_789"
        }
        
        response = await client.post(
            f"{base_url}/api/v1/certified-math/mul",
            json=mul_request,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        print(f"Multiplication response: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
        
        # Test division operation
        print("Testing division operation...")
        div_request = {
            "operand_a": "123456789012345678901234567890",
            "operand_b": "123456789",
            "pqc_cid": "test_pqc_cid_012"
        }
        
        response = await client.post(
            f"{base_url}/api/v1/certified-math/div",
            json=div_request,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        print(f"Division response: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
        
        # Test square root operation
        print("Testing square root operation...")
        sqrt_request = {
            "operand_a": "123456789012345678901234567890",
            "iterations": 10,
            "pqc_cid": "test_pqc_cid_345"
        }
        
        response = await client.post(
            f"{base_url}/api/v1/certified-math/sqrt",
            json=sqrt_request,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        print(f"Square root response: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
        
        # Test phi series operation
        print("Testing phi series operation...")
        phi_request = {
            "operand_a": "123456789012345678901234567890",
            "iterations": 5,
            "pqc_cid": "test_pqc_cid_678"
        }
        
        response = await client.post(
            f"{base_url}/api/v1/certified-math/phi-series",
            json=phi_request,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        print(f"Phi series response: {response.status_code}")
        print(f"Response: {response.json()}")
        print()

if __name__ == "__main__":
    print("Starting CertifiedMath API integration test...")
    print("=" * 50)
    asyncio.run(test_certified_math_api())
    print("=" * 50)
    print("Test completed!")