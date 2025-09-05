#!/usr/bin/env python3
"""
Complete MCP Server Test Suite
Tests the Kedro MCP server via proper MCP protocol
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

async def test_mcp_protocol():
    """Test the MCP server using proper protocol flow"""
    server_path = Path(__file__).parent / "kedro_mcp_server.py"
    
    # Start server
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(server_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        print("✅ MCP Server started")
        
        # Test 1: Initialize
        print("\n=== Test 1: Initialize ===")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        process.stdin.write((json.dumps(init_request) + "\n").encode())
        await process.stdin.drain()
        
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        init_response = json.loads(response_line.decode())
        print(f"Initialize response: {init_response}")
        assert init_response.get("result", {}).get("protocolVersion") == "2024-11-05"
        print("✅ Initialize test passed")
        
        # Test 2: List Tools
        print("\n=== Test 2: List Tools ===")
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        process.stdin.write((json.dumps(tools_request) + "\n").encode())
        await process.stdin.drain()
        
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        tools_response = json.loads(response_line.decode())
        print(f"Tools list response: {json.dumps(tools_response, indent=2)}")
        
        tools = tools_response.get("result", {}).get("tools", [])
        assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"
        tool_names = [tool["name"] for tool in tools]
        expected_tools = ["identify_pipeline_type", "create_pipeline_scaffold", "list_pipeline_types"]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
        print("✅ List tools test passed")
        
        # Test 3: Call list_pipeline_types
        print("\n=== Test 3: Call list_pipeline_types ===")
        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "list_pipeline_types",
                "arguments": {}
            }
        }
        
        process.stdin.write((json.dumps(call_request) + "\n").encode())
        await process.stdin.drain()
        
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        call_response = json.loads(response_line.decode())
        
        result = call_response.get("result", {})
        content = result.get("content", [])
        assert len(content) > 0, "Expected content in response"
        assert "Data Engineering Pipeline" in content[0]["text"], "Expected pipeline types in response"
        print("✅ list_pipeline_types test passed")
        
        # Test 4: Call identify_pipeline_type
        print("\n=== Test 4: Call identify_pipeline_type ===")
        identify_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "identify_pipeline_type",
                "arguments": {
                    "requirements": "I need to train a machine learning model for sentiment analysis",
                    "data_sources": ["text_data", "labels"],
                    "outputs": ["trained_model", "metrics"]
                }
            }
        }
        
        process.stdin.write((json.dumps(identify_request) + "\n").encode())
        await process.stdin.drain()
        
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        identify_response = json.loads(response_line.decode())
        
        result = identify_response.get("result", {})
        content = result.get("content", [])
        assert len(content) > 0, "Expected content in response"
        response_text = content[0]["text"]
        assert "ML Training Pipeline" in response_text, "Expected ML Training recommendation"
        assert "ml_training" in response_text, "Expected pipeline type in response"
        print("✅ identify_pipeline_type test passed")
        
        # Test 5: Call create_pipeline_scaffold
        print("\n=== Test 5: Call create_pipeline_scaffold ===")
        scaffold_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "create_pipeline_scaffold",
                "arguments": {
                    "pipeline_name": "sentiment_analysis",
                    "pipeline_type": "ml_training",
                    "description": "Sentiment analysis model training pipeline"
                }
            }
        }
        
        process.stdin.write((json.dumps(scaffold_request) + "\n").encode())
        await process.stdin.drain()
        
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        scaffold_response = json.loads(response_line.decode())
        
        result = scaffold_response.get("result", {})
        content = result.get("content", [])
        assert len(content) > 0, "Expected content in response"
        response_text = content[0]["text"]
        assert "sentiment_analysis" in response_text, "Expected pipeline name in response"
        assert "pipeline.py" in response_text, "Expected pipeline.py in scaffold"
        assert "nodes.py" in response_text, "Expected nodes.py in scaffold"
        print("✅ create_pipeline_scaffold test passed")
        
        print("\n🎉 All MCP protocol tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean shutdown
        if process.returncode is None:
            process.terminate()
            await process.wait()
        print("✅ MCP Server stopped")
    
    return True

async def main():
    """Run the complete test suite"""
    print("Starting Kedro MCP Server Complete Test Suite\n")
    success = await test_mcp_protocol()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)