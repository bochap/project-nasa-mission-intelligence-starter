import chromadb
from pathlib import Path
import os
import textwrap

from provider.plugins.chromadb_provider import ChromaPersistentRAGBuilder
from bootstrap import configure_rag

# Treat these user filters as "no filter" to prevent crashing on empty strings or general options
_ALL_EQUIVALENT_FILTER = {"all", "none", "any", "everything", ""}
_MAX_ERROR_LENGTH = 50
_MAX_DOCUMENT_LENGTH = 1500


def _get_rag_client(
    path: str, api_key: str | None = None, model: str = "text-embedding-3-small"
) -> chromadb.PersistentClient:
    """
    Internal helper to construct and configure our custom RAG client builder.
    Using the registry framework to resolve the persistent backend class.
    """
    builder = configure_rag(ChromaPersistentRAGBuilder.registered_name, ChromaPersistentRAGBuilder).with_path(path)
    if api_key:
        builder = builder.with_openai_embedding(api_key=api_key, model_name=model)
    return builder.build()


def discover_chroma_backends(api_key: str | None = None) -> dict[str, dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    """
    chromadb.PersistentClient creates a SQLite database with a filename hardcoded to chroma.sqlite3
    https://github.com/chroma-core/chroma/blob/main/chromadb/db/impl/sqlite.py#L88-L90
    """
    chromadbDatastores = [p.parent for p in current_dir.rglob("chroma.sqlite3")]

    backends = {}
    for folder in chromadbDatastores:
        try:
            client = _get_rag_client(path=folder, api_key=api_key)

            collections = client.list_collections()

            for collection in collections:
                normalized_path = os.path.abspath(os.path.normpath(folder))
                key = f"{folder.name}:{collection.name}"
                try:
                    document_count = collection.count()
                except Exception:
                    document_count = "Unknown"
                data = {
                    "directory": normalized_path,
                    "path": normalized_path,
                    "collection_name": collection.name,
                    "display_name": f"{normalized_path}/{collection.name}",
                    "document_count": document_count,
                }
                # TODO: Add collection information to backends dictionary
                backends[key] = data
        except Exception as e:
            error_message = textwrap.shorten(str(e), _MAX_ERROR_LENGTH)
            data = {
                "directory": normalized_path,
                "path": normalized_path,
                "collection_name": "N/A",
                "display_name": f"{normalized_path} (Error: {error_message})",
                "document_count": "N/A",
            }
            backends[key] = data

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str, api_key: str | None = None):
    """Initialize the RAG system with specified backend (cached for performance)"""
    path = os.path.abspath(os.path.normpath(chroma_dir))
    client = _get_rag_client(path=path, api_key=api_key)
    return client.get_or_create_collection(name=collection_name), "success", ""


def retrieve_documents(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 3,
    mission_filter: str | None = None,
) -> dict | None:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    filter = (
        None if not mission_filter or mission_filter.lower() in _ALL_EQUIVALENT_FILTER else {"mission": mission_filter}
    )
    # using query_texts that will use the collection embedding function which is set when the collection is created
    return collection.query(query_texts=[query], n_results=n_results, where=filter)


def format_context(documents: list[str], metadatas: list[dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context = ["Context:"]
    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        mission = metadata.get("mission", "Unknown mission")
        mission = mission.replace("_", " ").title()
        source = metadata.get("source", "Unknown source")
        category = metadata.get("document_category", "Unknown category")
        category = category.replace("_", " ").title()
        context.append(f"[Source {index}] Mission: {mission} | Category: {category} | Source: {source}")
        context.append(textwrap.shorten(document, width=_MAX_DOCUMENT_LENGTH))
    context.append("")

    # TODO: Join all context parts with newlines and return formatted string
    return "\n".join(context)
