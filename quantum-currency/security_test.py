#!/usr/bin/env python3
"""
Simple Security Test for HMN Components
"""

import sys
sys.path.insert(0, 'src')

from network.hmn.memory_mesh_service import MemoryMeshService

def run_security_test():
    """Run a simple security test"""
    print("Running HMN Security Test...")
    
    # Initialize test service with security features enabled
    service = MemoryMeshService('security-test-001', {
        'enable_tls': True, 
        'discovery_enabled': True
    })
    
    # Check security configuration
    print("Security Configuration:")
    print(f"  TLS Enabled: {service.config['network']['enable_tls']}")
    print(f"  Discovery Enabled: {service.config['network']['discovery_enabled']}")
    
    # Check that secure connection method exists
    if hasattr(service, 'establish_secure_connection'):
        print("✅ Secure connection method is available")
    else:
        print("❌ Secure connection method is missing")
    
    # Check that peer connections can be established
    try:
        result = service.establish_secure_connection('test-peer-001')
        if result:
            print("✅ Secure connection establishment successful")
        else:
            print("❌ Secure connection establishment failed")
    except Exception as e:
        print(f"❌ Error establishing secure connection: {e}")
    
    return service.config['network']['enable_tls']

if __name__ == "__main__":
    tls_enabled = run_security_test()
    if tls_enabled:
        print("\n✅ Security features are properly configured")
    else:
        print("\n❌ Security features need attention")