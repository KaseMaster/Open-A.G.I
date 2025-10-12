import asyncio
import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p2p_network import P2PNetworkManager, NodeType


async def quick_test():
    net = P2PNetworkManager("bugfix_node_1", NodeType.FULL, 8090)
    # Start discovery in background for a short period
    start_task = asyncio.create_task(net.discovery_service.start_discovery())
    await asyncio.sleep(1.5)
    await net.stop_network()  # calls stop_discovery and disconnects
    # Give some time for tasks to settle
    await asyncio.sleep(0.5)
    print("Quick start/stop sanity test completed")


if __name__ == "__main__":
    asyncio.run(quick_test())