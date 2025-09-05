"""
RAG (Retrieval-Augmented Generation) pipeline for document Q&A.

This pipeline implements a complete RAG system that:
1. Ingests and processes documents
2. Generates embeddings for document chunks
3. Stores embeddings in a vector database
4. Retrieves relevant context for queries
5. Generates answers using LLM with retrieved context
"""

from kedro.pipeline import Node, Pipeline, node

from .nodes import (
    ingest_documents,
    chunk_documents, 
    generate_embeddings,
    store_in_vector_db,
    retrieve_context,
    generate_answer
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the RAG system pipeline."""
    
    # Document ingestion and processing
    ingestion_nodes = [
        node(
            func=ingest_documents,
            inputs=["params:document_sources", "params:file_patterns"],
            outputs="raw_documents",
            name="ingest_documents_node",
            tags=["ingestion"]
        ),
        node(
            func=chunk_documents,
            inputs=["raw_documents", "params:chunk_size", "params:chunk_overlap"],
            outputs="document_chunks",
            name="chunk_documents_node",
            tags=["preprocessing"]
        ),
    ]
    
    # Embedding generation and storage
    embedding_nodes = [
        node(
            func=generate_embeddings,
            inputs=["document_chunks", "params:embedding_model"],
            outputs="document_embeddings",
            name="generate_embeddings_node",
            tags=["embeddings"]
        ),
        node(
            func=store_in_vector_db,
            inputs=["document_embeddings", "params:vector_db_config"],
            outputs="vector_store",
            name="store_in_vector_db_node",
            tags=["storage"]
        ),
    ]
    
    # Query processing and answer generation
    query_nodes = [
        node(
            func=retrieve_context,
            inputs=["params:user_query", "vector_store", "params:top_k"],
            outputs="retrieved_context",
            name="retrieve_context_node",
            tags=["retrieval"]
        ),
        node(
            func=generate_answer,
            inputs=["params:user_query", "retrieved_context", "params:llm_config"],
            outputs="generated_answer",
            name="generate_answer_node",
            tags=["generation"]
        ),
    ]
    
    return Pipeline([
        *ingestion_nodes,
        *embedding_nodes,
        *query_nodes,
    ])


def create_ingestion_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for document ingestion only."""
    return Pipeline([
        node(
            func=ingest_documents,
            inputs=["params:document_sources", "params:file_patterns"],
            outputs="raw_documents",
            name="ingest_documents_node",
        ),
        node(
            func=chunk_documents,
            inputs=["raw_documents", "params:chunk_size", "params:chunk_overlap"],
            outputs="document_chunks",
            name="chunk_documents_node",
        ),
        node(
            func=generate_embeddings,
            inputs=["document_chunks", "params:embedding_model"],
            outputs="document_embeddings",
            name="generate_embeddings_node",
        ),
        node(
            func=store_in_vector_db,
            inputs=["document_embeddings", "params:vector_db_config"],
            outputs="vector_store",
            name="store_in_vector_db_node",
        ),
    ])


def create_query_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for query processing only."""
    return Pipeline([
        node(
            func=retrieve_context,
            inputs=["params:user_query", "vector_store", "params:top_k"],
            outputs="retrieved_context",
            name="retrieve_context_node",
        ),
        node(
            func=generate_answer,
            inputs=["params:user_query", "retrieved_context", "params:llm_config"],
            outputs="generated_answer",
            name="generate_answer_node",
        ),
    ])