#!/usr/bin/env python3
"""
Manual test script for Kedro MCP Server
"""

import asyncio
import json
from kedro_mcp_server import call_tool, list_tools

async def test_list_tools():
    """Test the list_tools functionality"""
    print("=== Testing list_tools ===")
    result = await list_tools()
    print(f"Available tools: {len(result.tools)}")
    for tool in result.tools:
        print(f"- {tool.name}: {tool.description}")
    print()

async def test_list_pipeline_types():
    """Test listing pipeline types"""
    print("=== Testing list_pipeline_types ===")
    result = await call_tool("list_pipeline_types", {})
    print(result.content[0].text)
    print()

async def test_identify_pipeline_type():
    """Test pipeline type identification"""
    print("=== Testing identify_pipeline_type ===")
    
    test_cases = [
        {
            "requirements": "I need to train a machine learning model for customer churn prediction",
            "data_sources": ["customer_data", "transaction_history"],
            "outputs": ["trained_model", "model_metrics"]
        },
        {
            "requirements": "Clean and transform raw data from multiple sources",
            "data_sources": ["raw_api_data", "csv_files"],
            "outputs": ["cleaned_data", "processed_data"]
        },
        {
            "requirements": "Monitor data quality and detect anomalies",
            "data_sources": ["production_data"],
            "outputs": ["quality_report", "alerts"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- Test Case {i} ---")
        result = await call_tool("identify_pipeline_type", test_case)
        print(result.content[0].text)
        print()

async def test_create_pipeline_scaffold():
    """Test pipeline scaffold generation"""
    print("=== Testing create_pipeline_scaffold ===")
    
    test_cases = [
        {
            "pipeline_name": "customer_churn",
            "pipeline_type": "ml_training",
            "description": "Customer churn prediction model"
        },
        {
            "pipeline_name": "data_processing",
            "pipeline_type": "data_engineering"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- Test Case {i} ---")
        result = await call_tool("create_pipeline_scaffold", test_case)
        # Print only first part to avoid overwhelming output
        content = result.content[0].text
        lines = content.split('\n')
        print('\n'.join(lines[:20]))
        print("... (truncated for brevity)")
        print()

async def main():
    """Run all tests"""
    print("Starting Kedro MCP Server Tests\n")
    
    try:
        await test_list_tools()
        await test_list_pipeline_types()
        await test_identify_pipeline_type()
        await test_create_pipeline_scaffold()
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())