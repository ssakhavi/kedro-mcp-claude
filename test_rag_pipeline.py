#!/usr/bin/env python3
"""
Test script to create RAG pipeline using Kedro MCP server
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path so we can import the server
sys.path.insert(0, str(Path(__file__).parent))

async def create_rag_pipeline():
    """Use MCP server to create RAG pipeline"""
    print("Creating RAG Pipeline using Kedro MCP Server\n")
    
    try:
        from kedro_mcp_server import call_tool
        
        # Step 1: Identify appropriate pipeline type for RAG
        print("=== Step 1: Identifying Pipeline Type for RAG ===")
        identify_result = await call_tool("identify_pipeline_type", {
            "requirements": "Build a RAG (Retrieval-Augmented Generation) system for document Q&A that embeds documents, stores them in a vector database, retrieves relevant context, and generates answers using LLM",
            "data_sources": ["documents", "text_files", "pdfs", "knowledge_base"],
            "outputs": ["embeddings", "vector_store", "qa_responses", "retrieved_context"]
        })
        
        print(identify_result.content[0].text)
        print("\n" + "="*60 + "\n")
        
        # Step 2: Generate pipeline scaffold
        print("=== Step 2: Generating RAG Pipeline Scaffold ===")
        scaffold_result = await call_tool("create_pipeline_scaffold", {
            "pipeline_name": "rag_system",
            "pipeline_type": "feature_engineering",  # Based on the recommendation
            "description": "RAG (Retrieval-Augmented Generation) system for document Q&A with embeddings and vector search"
        })
        
        print(scaffold_result.content[0].text)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await create_rag_pipeline()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)