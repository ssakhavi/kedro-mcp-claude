#!/usr/bin/env python3
"""
Simple test for MCP Server - just check if it can be imported and basic functionality works
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path so we can import the server
sys.path.insert(0, str(Path(__file__).parent))

async def test_server_import_and_basic_functionality():
    """Test server import and basic functionality"""
    print("=== Testing Server Import ===")
    
    try:
        # Test import
        from kedro_mcp_server import list_tools, call_tool, PIPELINE_TYPES
        print("✅ Server imported successfully")
        
        # Test list_tools
        print("\n=== Testing list_tools function ===")
        tools_result = await list_tools()
        print(f"Number of tools: {len(tools_result.tools)}")
        for tool in tools_result.tools:
            print(f"- {tool.name}: {tool.description}")
        
        assert len(tools_result.tools) == 3, f"Expected 3 tools, got {len(tools_result.tools)}"
        print("✅ list_tools test passed")
        
        # Test pipeline types data
        print(f"\n=== Testing Pipeline Types ===")
        print(f"Number of pipeline types: {len(PIPELINE_TYPES)}")
        for key, value in PIPELINE_TYPES.items():
            print(f"- {key}: {value['name']}")
        
        assert len(PIPELINE_TYPES) == 6, f"Expected 6 pipeline types, got {len(PIPELINE_TYPES)}"
        print("✅ Pipeline types test passed")
        
        # Test call_tool - list_pipeline_types
        print(f"\n=== Testing call_tool: list_pipeline_types ===")
        result = await call_tool("list_pipeline_types", {})
        content = result.content[0].text
        assert "Data Engineering Pipeline" in content, "Expected pipeline types in content"
        print("✅ list_pipeline_types call test passed")
        
        # Test call_tool - identify_pipeline_type
        print(f"\n=== Testing call_tool: identify_pipeline_type ===")
        result = await call_tool("identify_pipeline_type", {
            "requirements": "Train a machine learning model",
            "data_sources": ["training_data"],
            "outputs": ["model"]
        })
        content = result.content[0].text
        assert "ML Training Pipeline" in content, "Expected ML Training recommendation"
        print("✅ identify_pipeline_type call test passed")
        
        # Test call_tool - create_pipeline_scaffold  
        print(f"\n=== Testing call_tool: create_pipeline_scaffold ===")
        result = await call_tool("create_pipeline_scaffold", {
            "pipeline_name": "test_pipeline",
            "pipeline_type": "ml_training"
        })
        content = result.content[0].text
        assert "test_pipeline" in content, "Expected pipeline name in scaffold"
        assert "pipeline.py" in content, "Expected pipeline.py in scaffold"
        print("✅ create_pipeline_scaffold call test passed")
        
        print(f"\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the basic functionality tests"""
    print("Starting Kedro MCP Server Basic Functionality Tests\n")
    success = await test_server_import_and_basic_functionality()
    
    if success:
        print("\n✅ All basic tests completed successfully!")
        print("\n📝 Summary:")
        print("- Server can be imported correctly")
        print("- All 3 tools are available and working")
        print("- Pipeline type identification works")
        print("- Pipeline scaffold generation works")
        print("- All 6 pipeline types are defined")
        return 0
    else:
        print("\n❌ Some basic tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)