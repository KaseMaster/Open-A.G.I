#!/usr/bin/env python3
"""
Main entry point for Quantum Currency Full Verification Loop
"""

import asyncio
import sys
import os

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Add the verification module to the path
verification_path = os.path.join(src_path, 'verification')
sys.path.insert(0, verification_path)

from full_verification_loop import QuantumVerificationFramework

async def main():
    """Main entry point"""
    print("🔍 Quantum Currency Full Verification Loop")
    print("=" * 45)
    
    # Create verification framework
    framework = QuantumVerificationFramework()
    
    # Run continuous verification
    await framework.run_continuous_verification()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Verification loop stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error running verification loop: {e}")
        sys.exit(1)