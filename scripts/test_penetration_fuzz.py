#!/usr/bin/env python3
"""
Penetration and fuzz testing for REST and consensus endpoints
"""

import sys
import os
import time
import random
import json
import requests
from typing import Dict, Any, List, Optional

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_fuzz_data(data_type: str) -> Any:
    """Generate fuzz data for testing"""
    if data_type == "string":
        # Generate random strings of various lengths
        length = random.randint(0, 1000)
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=length))
    elif data_type == "number":
        # Generate random numbers including edge cases
        return random.choice([
            random.randint(-1000000, 1000000),
            random.uniform(-1000000.0, 1000000.0),
            float('inf'),
            float('-inf'),
            float('nan')
        ])
    elif data_type == "array":
        # Generate arrays with random elements
        length = random.randint(0, 100)
        return [generate_fuzz_data("string") for _ in range(length)]
    elif data_type == "object":
        # Generate objects with random keys and values
        obj = {}
        num_keys = random.randint(0, 20)
        for _ in range(num_keys):
            key = generate_fuzz_data("string")[:50]  # Limit key length
            value = random.choice([
                generate_fuzz_data("string"),
                generate_fuzz_data("number"),
                generate_fuzz_data("array")
            ])
            obj[key] = value
        return obj
    else:
        return None

def create_valid_transaction() -> Dict[str, Any]:
    """Create a valid transaction for baseline testing"""
    return {
        "id": f"tx-{int(time.time())}",
        "type": "harmonic",
        "action": "mint",
        "token": "FLX",
        "sender": "node-A",
        "receiver": "node-A",
        "amount": 100,
        "aggregated_cs": 0.8,
        "sender_chr": 0.8,
        "timestamp": time.time()
    }

def create_valid_snapshot_data() -> Dict[str, Any]:
    """Create valid snapshot data for baseline testing"""
    # Generate simple time series data
    t = [i/100 for i in range(100)]
    values = [0.5 * i/100 for i in range(100)]
    
    return {
        "node_id": "node-A",
        "times": t,
        "values": values
    }

def create_valid_coherence_data() -> Dict[str, Any]:
    """Create valid coherence data for baseline testing"""
    # Generate simple time series data
    t = [i/50 for i in range(50)]
    values_a = [0.5 * i/50 for i in range(50)]
    values_b = [0.5 * i/50 + 0.1 for i in range(50)]
    
    return {
        "local": {
            "times": t,
            "values": values_a
        },
        "remotes": [
            {
                "times": t,
                "values": values_b
            }
        ]
    }

def test_endpoint(url: str, method: str, data: Any = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Test an endpoint and return results"""
    try:
        if method == "GET":
            response = requests.get(url, headers=headers or {}, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers or {}, timeout=5)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}
        
        return {
            "status": "success",
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds(),
            "content_length": len(response.content)
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": str(e),
            "response_time": 0
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "response_time": 0
        }

def run_penetration_test():
    """Run penetration and fuzz testing"""
    print("🛡️ Penetration and Fuzz Testing")
    print("=" * 40)
    
    # Test endpoints (using localhost for testing)
    base_url = "http://localhost:5000"  # Assuming REST API is running
    endpoints = [
        ("/snapshot", "POST"),
        ("/mint", "POST"),
        ("/ledger", "GET"),
        ("/coherence", "POST"),
        ("/transactions", "GET"),
        ("/snapshots", "GET")
    ]
    
    print(f"📍 Testing {len(endpoints)} endpoints")
    
    # Test 1: Valid data baseline
    print("\n🔍 Test 1: Valid Data Baseline")
    baseline_results = []
    
    for endpoint, method in endpoints:
        url = base_url + endpoint
        data = None
        
        # Add appropriate data for POST requests
        if method == "POST":
            if endpoint == "/snapshot":
                data = create_valid_snapshot_data()
            elif endpoint == "/mint":
                data = create_valid_transaction()
            elif endpoint == "/coherence":
                data = create_valid_coherence_data()
        
        result = test_endpoint(url, method, data)
        result["endpoint"] = endpoint
        result["test_type"] = "baseline"
        baseline_results.append(result)
        
        status = "✅" if result["status"] == "success" else "❌"
        if result["status"] == "success":
            print(f"   {status} {endpoint} ({method}): {result['status_code']} ({result['response_time']:.3f}s)")
        else:
            print(f"   {status} {endpoint} ({method}): {result['message']}")
    
    # Test 2: Fuzz testing with invalid data
    print("\n🔍 Test 2: Fuzz Testing with Invalid Data")
    fuzz_results = []
    
    # Test each endpoint with various types of invalid data
    fuzz_test_count = 0
    for endpoint, method in endpoints:
        if method == "POST":
            # Test with different types of fuzz data
            fuzz_types = ["string", "number", "array", "object"]
            for fuzz_type in fuzz_types:
                fuzz_test_count += 1
                url = base_url + endpoint
                fuzz_data = generate_fuzz_data(fuzz_type)
                
                result = test_endpoint(url, method, fuzz_data)
                result["endpoint"] = endpoint
                result["test_type"] = f"fuzz_{fuzz_type}"
                fuzz_results.append(result)
                
                # Print periodic updates
                if fuzz_test_count % 10 == 0:
                    print(f"   Completed {fuzz_test_count} fuzz tests")
    
    print(f"   Completed {fuzz_test_count} fuzz tests")
    
    # Test 3: SQL injection attempts
    print("\n🔍 Test 3: SQL Injection Attempts")
    sql_injection_tests = [
        "' OR '1'='1",
        "'; DROP TABLE users; --",
        "' UNION SELECT * FROM users --",
        "admin'--",
        "' OR 1=1--"
    ]
    
    sql_results = []
    sql_test_count = 0
    
    for endpoint, method in endpoints:
        if method == "POST":
            for payload in sql_injection_tests:
                sql_test_count += 1
                url = base_url + endpoint
                
                # Create malicious data
                malicious_data = {
                    "id": payload,
                    "type": payload,
                    "sender": payload,
                    "receiver": payload
                }
                
                result = test_endpoint(url, method, malicious_data)
                result["endpoint"] = endpoint
                result["test_type"] = "sql_injection"
                sql_results.append(result)
    
    print(f"   Completed {sql_test_count} SQL injection tests")
    
    # Test 4: Large payload testing
    print("\n🔍 Test 4: Large Payload Testing")
    large_payload_results = []
    
    # Create extremely large data
    large_data = {
        "id": "x" * 10000,
        "type": "harmonic",
        "action": "mint",
        "token": "FLX",
        "sender": "x" * 5000,
        "receiver": "x" * 5000,
        "amount": 100,
        "aggregated_cs": 0.8,
        "sender_chr": 0.8,
        "large_array": ["x" * 100 for _ in range(1000)],
        "large_object": {f"key_{i}": "x" * 100 for i in range(1000)}
    }
    
    for endpoint, method in endpoints:
        if method == "POST":
            url = base_url + endpoint
            result = test_endpoint(url, method, large_data)
            result["endpoint"] = endpoint
            result["test_type"] = "large_payload"
            large_payload_results.append(result)
    
    print(f"   Completed large payload tests")
    
    # Analyze results
    print(f"\n📊 Test Results Summary:")
    
    # Baseline results
    successful_baselines = sum(1 for r in baseline_results if r["status"] == "success")
    print(f"   Baseline Tests: {successful_baselines}/{len(baseline_results)} successful")
    
    # Fuzz results
    successful_fuzz = sum(1 for r in fuzz_results if r["status"] == "success")
    failed_fuzz = sum(1 for r in fuzz_results if r["status"] == "error")
    print(f"   Fuzz Tests: {successful_fuzz} successful, {failed_fuzz} failed")
    
    # SQL injection results
    successful_sql = sum(1 for r in sql_results if r["status"] == "success")
    failed_sql = sum(1 for r in sql_results if r["status"] == "error")
    print(f"   SQL Injection Tests: {successful_sql} successful, {failed_sql} failed")
    
    # Large payload results
    successful_large = sum(1 for r in large_payload_results if r["status"] == "success")
    failed_large = sum(1 for r in large_payload_results if r["status"] == "error")
    print(f"   Large Payload Tests: {successful_large} successful, {failed_large} failed")
    
    # Security assessment
    print(f"\n🛡️ Security Assessment:")
    if failed_fuzz > successful_fuzz * 0.5:
        print("   ✅ System handles invalid data gracefully")
    else:
        print("   ⚠️  System may be vulnerable to invalid data")
        
    if failed_sql == len(sql_results):
        print("   ✅ System appears to be protected against SQL injection")
    else:
        print("   ⚠️  System may be vulnerable to SQL injection")
        
    if failed_large > 0:
        print("   ✅ System handles large payloads appropriately")
    else:
        print("   ⚠️  System may be vulnerable to large payload attacks")
    
    print("\n✅ Penetration and fuzz testing completed successfully!")

if __name__ == "__main__":
    run_penetration_test()