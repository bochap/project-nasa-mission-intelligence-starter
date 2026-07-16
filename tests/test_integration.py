import os
import sys
from pathlib import Path
import pytest

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap
import embedding_pipeline
import llm_client
import rag_client
import ragas_evaluator


@pytest.fixture(scope="session")
def openai_api_key():
    """Retrieve and validate the live OpenAI API key from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Skipping live tests: OPENAI_API_KEY env variable is not set.")
    return api_key


@pytest.fixture(scope="function")
def temporary_chroma_dir(tmp_path):
    """Returns an absolute path to a unique, randomized directory provided by pytest."""
    return str(tmp_path.resolve())


# =====================================================================
# 1. Pipeline & Vector Database Integration Tests
# =====================================================================


def test_pipeline_text_chunking_and_ingestion(openai_api_key, temporary_chroma_dir):
    """Test the embedding pipeline with live OpenAI embeddings and ChromaDB operations."""
    collection_name = "test_integration_nasa_collection"

    # Initialize embedding pipeline with live client connections
    pipeline = embedding_pipeline.ChromaEmbeddingPipelineTextOnly(
        openai_api_key=openai_api_key,
        chroma_persist_directory=temporary_chroma_dir,
        collection_name=collection_name,
        embedding_model="text-embedding-3-small",
        chunk_size=150,
        chunk_overlap=20,
    )

    # Verify live embedding generation is reachable
    sample_text = "Apollo 13 was launched on April 11, 1970."
    embedding = pipeline.get_embedding(sample_text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert isinstance(embedding[0], float)

    # Verify chunk_text functions smoothly with a real SpaCy pipeline
    metadata = {"mission": "apollo_13", "source": "test_script"}
    chunks = pipeline.chunk_text(
        "Houston, we've had a problem here. This is a sentence designed to test our segmentation boundary scanner.",
        metadata,
    )
    assert len(chunks) > 0
    assert chunks[0][1]["mission"] == "apollo_13"

    # Add document chunks directly to collection
    file_path = Path("test_synthetic_doc.txt")
    stats = pipeline.add_documents_to_collection(
        documents=chunks, file_path=file_path, batch_size=5, update_mode="replace"
    )
    assert stats["added"] > 0
    assert pipeline.collection.count() > 0

    # Query the collection
    query_results = pipeline.query_collection("problem", n_results=1)
    assert len(query_results["documents"][0]) > 0


# =====================================================================
# 2. LLM Client Integration Tests
# =====================================================================


def test_llm_client_generation(openai_api_key):
    """Verify live OpenAI chat completions adhere strictly to the specialized system prompt."""
    user_message = "What was the designation of the Apollo 13 onboard voice transcript?"
    context = (
        "[Source 0] Mission: Apollo 13 | Category: Command Module\n"
        "This document is the transcription of the Apollo 13 flightcrew communications recorded on the command module."
    )
    conversation_history = []

    response = llm_client.generate_response(
        api_key=openai_api_key,
        user_message=user_message,
        context=context,
        conversation_history=conversation_history,
        model="gpt-3.5-turbo",
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert "Apollo 13" in response or "cannot be found" in response


# =====================================================================
# 3. RAG Client Integration Tests
# =====================================================================


def test_rag_client_discovery_and_retrieval(openai_api_key, temporary_chroma_dir):
    """Test background database detection and semantic filtering on live datastores."""
    collection_name = "discovery_test_col"

    # Populate the unique database folder using the pipeline
    pipeline = embedding_pipeline.ChromaEmbeddingPipelineTextOnly(
        openai_api_key=openai_api_key,
        chroma_persist_directory=temporary_chroma_dir,
        collection_name=collection_name,
        embedding_model="text-embedding-3-small",
    )

    test_document = "The Saturn V launch vehicle powered the mission."
    test_metadata = {"mission": "apollo_11", "source": "test_launch_procedure", "document_category": "technical"}

    pipeline.add_documents_to_collection(
        documents=[(test_document, test_metadata)], file_path=Path("test_manual_input.txt"), update_mode="replace"
    )

    # Re-initialize via rag_client factory block to test context extraction workflows
    collection, success, _ = rag_client.initialize_rag_system(temporary_chroma_dir, collection_name, openai_api_key)
    assert success == "success"

    # Match query with active filter matching our metadatas
    retrieved = rag_client.retrieve_documents(collection, "Saturn", n_results=1, mission_filter="apollo_11")
    assert retrieved is not None
    assert len(retrieved["documents"][0]) == 1

    # Format retrieved document context output
    formatted_context = rag_client.format_context(retrieved["documents"][0], retrieved["metadatas"][0])
    assert "Saturn" in formatted_context
    assert "Apollo 11" in formatted_context


# =====================================================================
# 4. RAGAS Evaluation Integration Tests
# =====================================================================


def test_ragas_evaluator_execution(openai_api_key):
    """Run an end-to-end evaluation with live RAGAS framework metrics."""
    question = "Who managed the transcription of the Apollo 13 tapes?"
    answer = "David M. Goldenbaum, Test Division, Apollo Spacecraft Program Office."
    contexts = [
        "Transcription of these tapes was managed by David M. Goldenbaum, Test Division, Apollo Spacecraft Program Office."
    ]

    scores = ragas_evaluator.evaluate_response_quality(
        openai_key=openai_api_key, question=question, answer=answer, contexts=contexts
    )

    assert isinstance(scores, dict)
    assert "error" not in scores, f"RAGAS evaluation failed with error: {scores.get('error')}"

    # Confirm metrics are returned as raw floats
    for metric_name in ["faithfulness", "answer_relevancy", "rouge_score", "bleu_score"]:
        assert metric_name in scores
        assert isinstance(scores[metric_name], float)
        assert 0.0 <= scores[metric_name] <= 1.0


# =====================================================================
# 5. Core Provider Registry Tests
# =====================================================================


def test_provider_architectural_registries(openai_api_key, temporary_chroma_dir):
    """Ensure dynamic builders hook successfully into our central dependency registry."""
    from provider.plugins.openai_provider import OpenAIClientBuilder, RagasAsyncOpenAIBuilder
    from provider.plugins.chromadb_provider import ChromaPersistentRAGBuilder

    # Verify LLM Registry Resolution
    llm_builder = bootstrap.configure_llm("openai", OpenAIClientBuilder)
    assert isinstance(llm_builder, OpenAIClientBuilder)
    llm = llm_builder.with_api_key(openai_api_key).build()
    assert hasattr(llm, "generate_string_response")

    # Verify RAG Database Registry Resolution
    rag_builder = bootstrap.configure_rag("chroma_persistent", ChromaPersistentRAGBuilder)
    assert isinstance(rag_builder, ChromaPersistentRAGBuilder)
    rag_client_instance = rag_builder.with_path(temporary_chroma_dir).build()
    assert hasattr(rag_client_instance, "list_collections")

    # Verify Ragas Core Client Resolution
    ragas_builder = bootstrap.configure_ragas("async_openai", RagasAsyncOpenAIBuilder)
    assert isinstance(ragas_builder, RagasAsyncOpenAIBuilder)
    ragas_client = ragas_builder.with_api_key(openai_api_key).build()
    assert hasattr(ragas_client, "llm")
    assert hasattr(ragas_client, "embeddings")
