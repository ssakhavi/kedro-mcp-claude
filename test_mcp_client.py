#!/usr/bin/env python3
"""
MCP Client test for Kedro MCP Server
Tests the server via stdio as a real MCP client would
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

class MCPClient:
    def __init__(self):
        self.process = None
        self.request_id = 1
    
    async def start_server(self):
        """Start the MCP server process"""
        server_path = Path(__file__).parent / "kedro_mcp_server.py"
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, str(server_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print("✅ MCP Server started")
    
    async def send_request(self, method: str, params: dict = None):
        """Send a JSON-RPC request to the server"""
        if not self.process:
            raise RuntimeError("Server not started")
        
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        if params:
            request["params"] = params
        
        self.request_id += 1
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from server")
        
        return json.loads(response_line.decode())
    
    async def initialize(self):
        """Initialize the MCP session"""
        response = await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        })
        print("✅ MCP Session initialized")
        return response
    
    async def list_tools(self):
        """List available tools"""
        response = await self.send_request("tools/list")
        return response
    
    async def call_tool(self, name: str, arguments: dict):
        """Call a tool"""
        response = await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        return response
    
    async def close(self):
        """Close the server process"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            print("✅ MCP Server closed")

async def test_mcp_server():
    """Test the MCP server via client"""
    client = MCPClient()
    
    try:
        # Start and initialize
        await client.start_server()
        await asyncio.sleep(1)  # Give server time to start
        await client.initialize()
        
        # Test listing tools
        print("\n=== Testing tools/list ===")
        tools_response = await client.list_tools()
        if "result" in tools_response:
            tools = tools_response["result"]["tools"]
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"- {tool['name']}: {tool['description']}")
        
        # Test tool calls
        print("\n=== Testing list_pipeline_types ===")
        response = await client.call_tool("list_pipeline_types", {})
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            print(content[:500] + "..." if len(content) > 500 else content)
        
        print("\n=== Testing identify_pipeline_type ===")
        response = await client.call_tool("identify_pipeline_type", {
            "requirements": "I need to train a machine learning model",
            "data_sources": ["training_data"],
            "outputs": ["trained_model"]
        })
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            print(content[:500] + "..." if len(content) > 500 else content)
        
        print("\n✅ All MCP tests completed successfully!")
        
    except Exception as e:
        print(f"❌ MCP test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_mcp_server())