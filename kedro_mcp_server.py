#!/usr/bin/env python3
"""
Kedro MCP Server

An MCP server that provides tools for working with Kedro ML/Data Science pipelines.
Provides capabilities to:
1. Identify pipeline types (data engineering, ML training, inference, etc.)
2. Generate Kedro pipeline scaffolds
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    EmbeddedResource,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
server = Server("kedro-mcp")

# Pipeline type definitions
PIPELINE_TYPES = {
    "data_engineering": {
        "name": "Data Engineering Pipeline",
        "description": "Focuses on data ingestion, cleaning, transformation, and preparation",
        "common_nodes": ["extract", "clean", "transform", "validate", "load"],
        "typical_inputs": ["raw_data", "external_apis", "databases"],
        "typical_outputs": ["cleaned_data", "transformed_data", "feature_store"]
    },
    "ml_training": {
        "name": "ML Training Pipeline", 
        "description": "Machine Learning model training and validation pipeline",
        "common_nodes": ["preprocess", "feature_engineering", "train", "validate", "evaluate"],
        "typical_inputs": ["training_data", "validation_data", "hyperparameters"],
        "typical_outputs": ["trained_model", "model_metrics", "evaluation_results"]
    },
    "ml_inference": {
        "name": "ML Inference Pipeline",
        "description": "Model serving and batch prediction pipeline", 
        "common_nodes": ["load_model", "preprocess", "predict", "postprocess"],
        "typical_inputs": ["trained_model", "inference_data", "model_config"],
        "typical_outputs": ["predictions", "prediction_metadata", "monitoring_metrics"]
    },
    "data_quality": {
        "name": "Data Quality Pipeline",
        "description": "Data validation, profiling, and quality monitoring",
        "common_nodes": ["profile", "validate", "monitor", "alert", "report"],
        "typical_inputs": ["input_data", "quality_rules", "baseline_stats"],
        "typical_outputs": ["quality_report", "anomaly_alerts", "data_profile"]
    },
    "feature_engineering": {
        "name": "Feature Engineering Pipeline",
        "description": "Feature creation, selection, and transformation pipeline",
        "common_nodes": ["extract_features", "transform_features", "select_features", "validate_features"],
        "typical_inputs": ["raw_data", "feature_definitions", "historical_features"],
        "typical_outputs": ["feature_store", "feature_metadata", "feature_lineage"]
    },
    "model_evaluation": {
        "name": "Model Evaluation Pipeline", 
        "description": "Comprehensive model testing and performance evaluation",
        "common_nodes": ["load_models", "run_tests", "compare_models", "generate_report"],
        "typical_inputs": ["candidate_models", "test_data", "evaluation_metrics"],
        "typical_outputs": ["model_comparison", "performance_report", "model_ranking"]
    }
}

def generate_kedro_scaffold(pipeline_name: str, pipeline_type: str) -> Dict[str, str]:
    """Generate Kedro pipeline scaffold files based on type."""
    
    pipeline_info = PIPELINE_TYPES.get(pipeline_type, PIPELINE_TYPES["data_engineering"])
    
    # Generate pipeline.py
    nodes_import = ", ".join(pipeline_info["common_nodes"])
    pipeline_py = f'''"""
This is a {pipeline_info['name'].lower()} '{pipeline_name}' pipeline.
{pipeline_info['description']}
"""

from kedro.pipeline import Node, Pipeline, node

from .nodes import {nodes_import}


def create_pipeline(**kwargs) -> Pipeline:
    """Create the {pipeline_name} pipeline."""
    
    pipeline_nodes = [
{chr(10).join([f'        node({node_name}, inputs="{node_name}_input", outputs="{node_name}_output", name="{node_name}_node"),' for node_name in pipeline_info["common_nodes"]])}
    ]
    
    return Pipeline(pipeline_nodes)
'''

    # Generate nodes.py
    nodes_py = f'''"""
Node functions for the {pipeline_name} pipeline.
"""

import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

{chr(10).join([f'''
def {node_name}(input_data: Any) -> Any:
    """
    {node_name.replace('_', ' ').title()} step for {pipeline_info['name'].lower()}.
    
    Args:
        input_data: Input data for {node_name}
        
    Returns:
        Processed data from {node_name} step
    """
    logger.info(f"Running {node_name} step")
    # TODO: Implement {node_name} logic
    return input_data''' for node_name in pipeline_info["common_nodes"]])}
'''

    # Generate __init__.py
    init_py = f'"""The {pipeline_name} pipeline."""'

    # Generate test file
    test_py = f'''"""
Tests for the {pipeline_name} pipeline.
"""

import pytest
from kedro.pipeline import Pipeline

from {pipeline_name}.pipeline import create_pipeline


class TestPipeline:
    def test_create_pipeline(self):
        """Test pipeline creation."""
        pipeline = create_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.nodes) == {len(pipeline_info["common_nodes"])}
        
    def test_pipeline_nodes(self):
        """Test individual pipeline nodes."""
        pipeline = create_pipeline()
        node_names = [node.name for node in pipeline.nodes]
        expected_nodes = {[f'"{node}_node"' for node in pipeline_info["common_nodes"]]}
        
        for expected_node in expected_nodes:
            assert expected_node in node_names
'''

    # Generate parameters YAML
    params_yaml = f'''# Parameters for {pipeline_name} pipeline
# {pipeline_info['description']}

{pipeline_name}:
  # Common parameters
  random_seed: 42
  
  # Input/Output configuration
  inputs:
{chr(10).join([f'    - {inp}' for inp in pipeline_info["typical_inputs"]])}
  
  outputs:
{chr(10).join([f'    - {out}' for out in pipeline_info["typical_outputs"]])}
  
  # Pipeline-specific parameters
  # TODO: Add specific parameters for {pipeline_info['name'].lower()}
'''

    return {
        "pipeline.py": pipeline_py,
        "nodes.py": nodes_py,
        "__init__.py": init_py,
        "tests/test_pipeline.py": test_py,
        f"config/parameters_{pipeline_name}.yml": params_yaml
    }


@server.list_tools()
async def list_tools() -> ListToolsResult:
    """List available Kedro MCP tools."""
    return ListToolsResult(
        tools=[
            Tool(
                name="identify_pipeline_type",
                description="Analyze and suggest the most appropriate Kedro pipeline type based on requirements",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "requirements": {
                            "type": "string",
                            "description": "Description of what the pipeline should accomplish"
                        },
                        "data_sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of data sources the pipeline will work with"
                        },
                        "outputs": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Expected outputs from the pipeline"
                        }
                    },
                    "required": ["requirements"]
                }
            ),
            Tool(
                name="create_pipeline_scaffold",
                description="Generate a Kedro pipeline scaffold with appropriate structure and boilerplate code",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pipeline_name": {
                            "type": "string",
                            "description": "Name of the pipeline to create"
                        },
                        "pipeline_type": {
                            "type": "string",
                            "enum": list(PIPELINE_TYPES.keys()),
                            "description": "Type of pipeline to create"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of the pipeline's purpose"
                        }
                    },
                    "required": ["pipeline_name", "pipeline_type"]
                }
            ),
            Tool(
                name="list_pipeline_types",
                description="List all available Kedro pipeline types with descriptions",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            )
        ]
    )


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """Handle tool calls."""
    
    if name == "identify_pipeline_type":
        requirements = arguments.get("requirements", "")
        data_sources = arguments.get("data_sources", [])
        outputs = arguments.get("outputs", [])
        
        # Simple keyword-based pipeline type identification
        requirements_lower = requirements.lower()
        
        scores = {}
        for pipeline_type, info in PIPELINE_TYPES.items():
            score = 0
            
            # Check requirements against description and common nodes
            description_words = info["description"].lower().split()
            nodes_words = " ".join(info["common_nodes"]).lower()
            
            for word in description_words + nodes_words.split():
                if word in requirements_lower:
                    score += 1
                    
            # Check data sources and outputs alignment
            for source in data_sources:
                if any(source.lower() in inp.lower() for inp in info["typical_inputs"]):
                    score += 2
                    
            for output in outputs:
                if any(output.lower() in out.lower() for out in info["typical_outputs"]):
                    score += 2
            
            scores[pipeline_type] = score
        
        # Get top recommendation
        recommended_type = max(scores, key=scores.get) if scores else "data_engineering"
        recommended_info = PIPELINE_TYPES[recommended_type]
        
        # Prepare recommendations
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        result = f"""**Pipeline Type Recommendation**

**Primary Recommendation: {recommended_info['name']}**
- Type: `{recommended_type}`
- Description: {recommended_info['description']}
- Score: {scores[recommended_type]}/10

**Reasoning:**
Based on your requirements: "{requirements}"
- Data sources: {', '.join(data_sources) if data_sources else 'Not specified'}
- Expected outputs: {', '.join(outputs) if outputs else 'Not specified'}

**Alternative Options:**
"""
        
        for pipeline_type, score in sorted_types[1:]:
            info = PIPELINE_TYPES[pipeline_type]
            result += f"\n- **{info['name']}** (`{pipeline_type}`) - Score: {score}/10\n  {info['description']}"
            
        result += f"""

**Next Steps:**
Use the `create_pipeline_scaffold` tool with:
- pipeline_name: your_pipeline_name
- pipeline_type: {recommended_type}
"""
        
        return CallToolResult(
            content=[TextContent(type="text", text=result)]
        )
    
    elif name == "create_pipeline_scaffold":
        pipeline_name = arguments["pipeline_name"]
        pipeline_type = arguments["pipeline_type"]
        description = arguments.get("description", "")
        
        if pipeline_type not in PIPELINE_TYPES:
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"Error: Unknown pipeline type '{pipeline_type}'. Available types: {list(PIPELINE_TYPES.keys())}"
                )]
            )
        
        # Generate scaffold files
        scaffold_files = generate_kedro_scaffold(pipeline_name, pipeline_type)
        pipeline_info = PIPELINE_TYPES[pipeline_type]
        
        result = f"""**Kedro Pipeline Scaffold Generated**

**Pipeline:** {pipeline_name}
**Type:** {pipeline_info['name']}
**Description:** {pipeline_info['description']}

**Generated Files:**

"""
        
        for filename, content in scaffold_files.items():
            result += f"""
**{filename}:**
```python
{content.strip()}
```

---
"""
        
        result += f"""
**Implementation Notes:**
- Common nodes for this pipeline type: {', '.join(pipeline_info['common_nodes'])}
- Typical inputs: {', '.join(pipeline_info['typical_inputs'])}
- Typical outputs: {', '.join(pipeline_info['typical_outputs'])}

**Next Steps:**
1. Create the pipeline directory structure in your Kedro project
2. Copy the generated files to appropriate locations
3. Update the parameter values in the config file
4. Implement the TODO items in nodes.py
5. Run `kedro pipeline create {pipeline_name}` to register the pipeline
"""
        
        return CallToolResult(
            content=[TextContent(type="text", text=result)]
        )
    
    elif name == "list_pipeline_types":
        result = "**Available Kedro Pipeline Types**\n\n"
        
        for pipeline_type, info in PIPELINE_TYPES.items():
            result += f"**{info['name']}** (`{pipeline_type}`)\n"
            result += f"- {info['description']}\n"
            result += f"- Common nodes: {', '.join(info['common_nodes'])}\n"
            result += f"- Typical inputs: {', '.join(info['typical_inputs'])}\n"
            result += f"- Typical outputs: {', '.join(info['typical_outputs'])}\n\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=result)]
        )
    
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")]
        )


async def main():
    """Main server entry point."""
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Kedro MCP Server")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="kedro-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())