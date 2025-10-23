#!/usr/bin/env python3
"""
Simple WebSocket connection test for the optimized MTL system
"""

import asyncio
import websockets
import json
import sys

async def test_websocket_connection():
    """Test basic WebSocket connection without problematic parameters"""
    print(" Testing WebSocket connection...")

    # Test server connection parameters that work
    uri = "ws://localhost:8771"

    try:
        # Test connection with only supported parameters
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10,
            max_size=50 * 1024 * 1024
        ) as websocket:
            print(" WebSocket connection successful!")

            # Test sending a simple message
            test_message = {"type": "test", "message": "Hello from test"}
            await websocket.send(json.dumps(test_message))
            print(" Message sent successfully!")

            # Wait a moment for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f" Response received: {response[:100]}...")
            except asyncio.TimeoutError:
                print("ℹ️ No response received (expected for test)")

            return True

    except Exception as e:
        print(f" WebSocket connection failed: {e}")
        return False

async def main():
    """Main test function"""
    print(" WebSocket Connection Test")
    print("=" * 40)

    success = await test_websocket_connection()

    if success:
        print("\n WebSocket connection test PASSED!")
        print(" The optimized MTL system should work correctly now.")
    else:
        print("\n WebSocket connection test FAILED!")
        print(" Check that the server is running and port 8771 is available.")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
