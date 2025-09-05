#!/usr/bin/env python3
"""
Test MCP server support for RAG pipeline development
Tests various RAG-related scenarios and evaluates recommendations
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path so we can import the server
sys.path.insert(0, str(Path(__file__).parent))

async def test_rag_scenarios():
    """Test MCP server with various RAG-related scenarios"""
    print("Testing Kedro MCP Server Support for RAG Pipeline Development\n")
    
    try:
        from kedro_mcp_server import call_tool
        
        # Test scenarios for different RAG components
        rag_scenarios = [
            {
                "name": "Complete RAG System",
                "requirements": "Build a complete RAG system that ingests documents, creates embeddings, stores them in vector database, retrieves relevant context for queries, and generates answers using LLM",
                "data_sources": ["documents", "pdfs", "text_files", "web_content", "knowledge_base"],
                "outputs": ["embeddings", "vector_store", "retrieved_context", "generated_answers", "similarity_scores"]
            },
            {
                "name": "Document Processing & Embedding",
                "requirements": "Process and chunk documents, generate embeddings using transformer models, and prepare data for vector storage",
                "data_sources": ["raw_documents", "pdf_files", "text_corpus"],
                "outputs": ["document_chunks", "embeddings", "metadata", "processed_documents"]
            },
            {
                "name": "Vector Search & Retrieval",
                "requirements": "Implement semantic search over document embeddings, retrieve most relevant passages, and rank results by similarity",
                "data_sources": ["vector_database", "query_embeddings", "document_metadata"],
                "outputs": ["search_results", "ranked_passages", "similarity_scores", "retrieved_context"]
            },
            {
                "name": "LLM Integration & Answer Generation",
                "requirements": "Integrate with language models to generate contextual answers using retrieved documents as context",
                "data_sources": ["user_queries", "retrieved_context", "prompt_templates"],
                "outputs": ["generated_answers", "confidence_scores", "source_attributions"]
            },
            {
                "name": "RAG Data Quality & Monitoring",
                "requirements": "Monitor embedding quality, track retrieval accuracy, validate document processing, and ensure answer relevance",
                "data_sources": ["embeddings", "retrieval_logs", "user_feedback", "ground_truth"],
                "outputs": ["quality_metrics", "accuracy_reports", "monitoring_dashboards", "alerts"]
            },
            {
                "name": "RAG Model Training & Evaluation",
                "requirements": "Fine-tune embedding models, train retrieval rankers, evaluate RAG system performance, and optimize hyperparameters",
                "data_sources": ["training_data", "query_document_pairs", "relevance_labels"],
                "outputs": ["fine_tuned_models", "evaluation_metrics", "performance_reports", "optimized_parameters"]
            }
        ]
        
        results = {}
        
        for i, scenario in enumerate(rag_scenarios, 1):
            print(f"=== Scenario {i}: {scenario['name']} ===")
            
            result = await call_tool("identify_pipeline_type", {
                "requirements": scenario["requirements"],
                "data_sources": scenario["data_sources"],
                "outputs": scenario["outputs"]
            })
            
            content = result.content[0].text
            print(content)
            print("\n" + "-"*80 + "\n")
            
            # Extract recommendation
            if "Primary Recommendation:" in content:
                lines = content.split('\n')
                for line in lines:
                    if "Primary Recommendation:" in line:
                        recommendation = line.split("**Primary Recommendation:")[-1].split("**")[0].strip()
                        break
                else:
                    recommendation = "Unknown"
            else:
                recommendation = "No recommendation found"
            
            results[scenario["name"]] = {
                "recommendation": recommendation,
                "full_response": content
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_scaffold_generation():
    """Test scaffold generation for different RAG pipeline types"""
    print("\n" + "="*80)
    print("Testing Scaffold Generation for RAG Components")
    print("="*80 + "\n")
    
    try:
        from kedro_mcp_server import call_tool
        
        # Test scaffolds for different pipeline types relevant to RAG
        scaffold_tests = [
            {
                "name": "Document Processing (Data Engineering)",
                "pipeline_name": "document_processing",
                "pipeline_type": "data_engineering",
                "description": "Document ingestion, chunking, and preprocessing for RAG"
            },
            {
                "name": "Embedding Generation (Feature Engineering)",
                "pipeline_name": "embedding_generation", 
                "pipeline_type": "feature_engineering",
                "description": "Generate and transform document embeddings for vector search"
            },
            {
                "name": "RAG Model Training (ML Training)",
                "pipeline_name": "rag_training",
                "pipeline_type": "ml_training", 
                "description": "Train and validate RAG retrieval and generation models"
            },
            {
                "name": "RAG Inference (ML Inference)",
                "pipeline_name": "rag_inference",
                "pipeline_type": "ml_inference",
                "description": "RAG system inference for query answering"
            },
            {
                "name": "RAG Quality Monitoring (Data Quality)",
                "pipeline_name": "rag_quality",
                "pipeline_type": "data_quality",
                "description": "Monitor RAG system quality and performance"
            }
        ]
        
        scaffold_results = {}
        
        for test in scaffold_tests:
            print(f"=== Generating Scaffold: {test['name']} ===")
            
            result = await call_tool("create_pipeline_scaffold", {
                "pipeline_name": test["pipeline_name"],
                "pipeline_type": test["pipeline_type"],
                "description": test["description"]
            })
            
            content = result.content[0].text
            
            # Extract key information
            has_pipeline_py = "pipeline.py" in content
            has_nodes_py = "nodes.py" in content
            has_tests = "test_pipeline.py" in content
            has_config = "parameters_" in content
            
            scaffold_results[test["name"]] = {
                "pipeline_type": test["pipeline_type"],
                "has_complete_structure": has_pipeline_py and has_nodes_py and has_tests and has_config,
                "content_preview": content[:500] + "..." if len(content) > 500 else content
            }
            
            print(f"✅ Generated scaffold for {test['pipeline_name']} ({test['pipeline_type']})")
            print(f"   - Pipeline structure: {'✅' if has_pipeline_py else '❌'}")
            print(f"   - Node functions: {'✅' if has_nodes_py else '❌'}")
            print(f"   - Tests: {'✅' if has_tests else '❌'}")
            print(f"   - Configuration: {'✅' if has_config else '❌'}")
            print()
        
        return scaffold_results
        
    except Exception as e:
        print(f"❌ Scaffold test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def analyze_rag_support():
    """Analyze overall RAG support and identify gaps"""
    print("\n" + "="*80)
    print("RAG Support Analysis")
    print("="*80 + "\n")
    
    # Test recommendations
    scenario_results = await test_rag_scenarios()
    if not scenario_results:
        return False
    
    # Test scaffold generation
    scaffold_results = await test_scaffold_generation()
    if not scaffold_results:
        return False
    
    print("=== Analysis Results ===\n")
    
    # Analyze recommendations
    print("📊 **Pipeline Type Recommendations for RAG Scenarios:**")
    recommendation_counts = {}
    for scenario, result in scenario_results.items():
        rec = result["recommendation"]
        recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        print(f"   - {scenario}: {rec}")
    
    print(f"\n📈 **Recommendation Distribution:**")
    for rec, count in sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {rec}: {count} scenarios")
    
    # Analyze scaffold completeness
    print(f"\n🏗️ **Scaffold Generation Results:**")
    complete_scaffolds = sum(1 for result in scaffold_results.values() if result["has_complete_structure"])
    print(f"   - Complete scaffolds generated: {complete_scaffolds}/{len(scaffold_results)}")
    
    for name, result in scaffold_results.items():
        status = "✅" if result["has_complete_structure"] else "⚠️"
        print(f"   {status} {name} ({result['pipeline_type']})")
    
    # Identify strengths and gaps
    print(f"\n🎯 **RAG Support Assessment:**")
    
    strengths = []
    gaps = []
    
    # Check coverage of RAG components
    rag_components = {
        "Document Processing": any("Data Engineering" in result["recommendation"] for result in scenario_results.values()),
        "Embedding Generation": any("Feature Engineering" in result["recommendation"] for result in scenario_results.values()),
        "Model Training": any("ML Training" in result["recommendation"] for result in scenario_results.values()),
        "Quality Monitoring": any("Data Quality" in result["recommendation"] for result in scenario_results.values()),
        "Inference/Serving": any("ML Inference" in result["recommendation"] for result in scenario_results.values())
    }
    
    for component, supported in rag_components.items():
        if supported:
            strengths.append(f"✅ {component} - Pipeline type identified")
        else:
            gaps.append(f"❌ {component} - No clear pipeline type match")
    
    # Check scaffold quality
    if complete_scaffolds == len(scaffold_results):
        strengths.append("✅ Complete scaffold generation for all pipeline types")
    else:
        gaps.append(f"⚠️ Incomplete scaffolds: {len(scaffold_results) - complete_scaffolds} missing components")
    
    print("**Strengths:**")
    for strength in strengths:
        print(f"  {strength}")
    
    print("\n**Gaps/Limitations:**")
    for gap in gaps:
        print(f"  {gap}")
    
    # Overall assessment
    support_score = (len(strengths) / (len(strengths) + len(gaps))) * 100 if (strengths or gaps) else 0
    print(f"\n🏆 **Overall RAG Support Score: {support_score:.1f}%**")
    
    if support_score >= 80:
        print("✅ **Excellent RAG support** - MCP server can effectively guide RAG pipeline development")
    elif support_score >= 60:
        print("⚠️ **Good RAG support** - MCP server provides solid foundation with some gaps")
    elif support_score >= 40:
        print("⚠️ **Moderate RAG support** - MCP server partially supports RAG development")
    else:
        print("❌ **Limited RAG support** - Significant gaps in RAG pipeline guidance")
    
    return True

async def main():
    """Run RAG support analysis"""
    success = await analyze_rag_support()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)