# Kedro MCP Server

A Model Context Protocol (MCP) server for working with Kedro ML/Data Science pipelines. This server provides tools to identify appropriate pipeline types and generate Kedro pipeline scaffolds.

## Features

### Pipeline Type Detection
- **`identify_pipeline_type`**: Analyzes requirements and suggests the most appropriate Kedro pipeline type
- Supports 6 different pipeline types:
  - **Data Engineering**: Data ingestion, cleaning, transformation, and preparation
  - **ML Training**: Machine Learning model training and validation 
  - **ML Inference**: Model serving and batch prediction
  - **Data Quality**: Data validation, profiling, and quality monitoring
  - **Feature Engineering**: Feature creation, selection, and transformation
  - **Model Evaluation**: Comprehensive model testing and performance evaluation

### Pipeline Scaffold Generation
- **`create_pipeline_scaffold`**: Generates complete Kedro pipeline scaffolds with:
  - `pipeline.py` - Pipeline definition with appropriate nodes
  - `nodes.py` - Node function templates
  - `__init__.py` - Package initialization
  - `tests/test_pipeline.py` - Unit test templates
  - `config/parameters_*.yml` - Parameter configuration

### Pipeline Type Reference
- **`list_pipeline_types`**: Lists all available pipeline types with descriptions

## Installation

```bash
pip install -e .
```

## Usage

### As an MCP Server

Run the server:
```bash
python kedro_mcp_server.py
```

### Example Tool Calls

#### 1. Identify Pipeline Type
```json
{
  "name": "identify_pipeline_type",
  "arguments": {
    "requirements": "I need to train a machine learning model for customer churn prediction",
    "data_sources": ["customer_data", "transaction_history"],
    "outputs": ["trained_model", "model_metrics"]
  }
}
```

#### 2. Generate Pipeline Scaffold
```json
{
  "name": "create_pipeline_scaffold", 
  "arguments": {
    "pipeline_name": "churn_prediction",
    "pipeline_type": "ml_training",
    "description": "Customer churn prediction model training pipeline"
  }
}
```

#### 3. List Available Pipeline Types
```json
{
  "name": "list_pipeline_types",
  "arguments": {}
}
```

## Pipeline Types

| Type | Description | Common Nodes | Typical Use Cases |
|------|-------------|--------------|------------------|
| `data_engineering` | Data ingestion, cleaning, transformation | extract, clean, transform, validate, load | ETL/ELT processes, data preparation |
| `ml_training` | ML model training and validation | preprocess, feature_engineering, train, validate, evaluate | Model development, hyperparameter tuning |
| `ml_inference` | Model serving and batch prediction | load_model, preprocess, predict, postprocess | Batch scoring, real-time inference |
| `data_quality` | Data validation and monitoring | profile, validate, monitor, alert, report | Data quality assurance, anomaly detection |
| `feature_engineering` | Feature creation and transformation | extract_features, transform_features, select_features, validate_features | Feature development, feature store management |
| `model_evaluation` | Model testing and comparison | load_models, run_tests, compare_models, generate_report | A/B testing, model comparison, performance analysis |

## Integration with Kedro Projects

The generated scaffolds can be directly integrated into existing Kedro projects:

1. Use `identify_pipeline_type` to determine the best pipeline type for your use case
2. Use `create_pipeline_scaffold` to generate the pipeline structure
3. Copy the generated files into your Kedro project's `src/<package>/pipelines/` directory
4. Update the parameter files with your specific configuration
5. Implement the TODO items in the generated `nodes.py` file
6. Register the pipeline in your Kedro project

## Requirements

- Python 3.8+
- MCP (Model Context Protocol) 1.0+
- Kedro
- Pandas
- Pydantic 2.0+

## License

MIT License