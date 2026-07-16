#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import spacy
import time
from typing import Any
from typing import NamedTuple

from provider.plugins.chromadb_provider import ChromaPersistentRAGBuilder
from provider.plugins.openai_provider import OpenAIClientBuilder
from bootstrap import configure_llm, configure_rag

logger = logging.getLogger("rag_system.embedding_pipeline")

class Sentence(NamedTuple):
    start: int
    end: int
    span: spacy.tokens.Span


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the embedding pipeline

        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk overlap {chunk_overlap} must be smaller than chunk size {chunk_size}")

        # Hook into our provider factory layer to get pre-configured infrastructure clients
        self.llm_client = configure_llm("openai", OpenAIClientBuilder).with_api_key(openai_api_key).build()
        self.rag_client = (
            configure_rag("chroma_persistent", ChromaPersistentRAGBuilder)
            .with_path(chroma_persist_directory)
            .with_openai_embedding(api_key=openai_api_key, model_name=embedding_model)
            .build()
        )
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("sentencizer")
        self.collection = self.rag_client.get_or_create_collection(
            name=collection_name,
        )
        self.embedding_model = embedding_model
        self.chroma_persist_directory = chroma_persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """
        Chunking function using a single-pass token cursor.
        """
        if not text or not (text := text.strip()):
            return []

        doc_len = len(text)

        # Handle short text blocks early
        if doc_len <= self.chunk_size:
            return [(text, {**metadata, "chunk_index": 0, "chunk_start": 0, "chunk_end": doc_len})]

        chunks = []
        chosen_start = 0
        chunk_index = 0

        while chosen_start < doc_len:
            # Look ahead far enough to capture full sentence boundaries past our chunk size
            lookahead_limit = min(chosen_start + (self.chunk_size * 2), doc_len)
            max_end = min(chosen_start + self.chunk_size, doc_len)

            # Optimization: Slice a localized fragment to keep spaCy RAM footprint small
            window_text = text[chosen_start:lookahead_limit]
            local_doc = self.nlp(window_text)

            chosen_end = max_end

            # Collect sentence boundaries detected within this narrow window snippet
            sentences = list(local_doc.sents)
            fitting_sents = []

            for s in sentences:
                abs_sent_end = chosen_start + s.end_char
                if abs_sent_end <= max_end:
                    fitting_sents.append(abs_sent_end)
                else:
                    break

            if fitting_sents:
                # Best Case: Break at the furthest sentence that fits within chunk_size
                chosen_end = fitting_sents[-1]
            else:
                # Oversized Block Fallback: Scan backward to snap directly to the closest word bound
                for token in reversed(local_doc):
                    abs_token_end = chosen_start + token.idx + len(token)
                    if abs_token_end <= max_end:
                        chosen_end = abs_token_end
                        break

            chunk_content = text[chosen_start:chosen_end].strip()
            if chunk_content:
                chunks.append(
                    (
                        chunk_content,
                        {**metadata, "chunk_index": chunk_index, "chunk_start": chosen_start, "chunk_end": chosen_end},
                    )
                )
                chunk_index += 1

            if chosen_end >= doc_len:
                break

            # Overlap Engine: Shift start bound back by overlap allowance
            raw_next_start = chosen_end - self.chunk_overlap
            if raw_next_start <= chosen_start:
                raw_next_start = chosen_start + (self.chunk_size - self.chunk_overlap)

            # Snap next start bound to a clean spaCy token index to prevent mid-word stuttering
            chosen_start = raw_next_start
            for token in local_doc:
                abs_token_start = chosen_start + token.idx
                if abs_token_start >= raw_next_start:
                    if abs_token_start < chosen_end:
                        chosen_start = abs_token_start
                    break

        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection

        Args:
            doc_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        # TODO: Query collection for document ID
        # TODO: Return True if exists, False otherwise
        try:
            response = self.collection.get(ids=[doc_id])
            return bool(response["ids"])
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False

    def update_document(self, doc_id: str, text: str, metadata: dict[str, Any]) -> bool:
        """
        Update an existing document in the collection

        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text)

            # Update the document
            self.collection.update(ids=[doc_id], documents=[text], metadatas=[metadata], embeddings=[embedding])
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)

        Args:
            source_pattern: Pattern to match source names

        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get()

            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs["metadatas"]):
                if source_pattern in metadata.get("source", ""):
                    ids_to_delete.append(all_docs["ids"][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0

        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> list[str]:
        """
        Get all document IDs for a specific file

        Args:
            file_path: Path to the file

        Returns:
            list of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)

            # Get all documents
            all_docs = self.collection.get()

            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs["metadatas"]):
                if metadata.get("source") == source and metadata.get("mission") == mission:
                    file_doc_ids.append(all_docs["ids"][i])

            return file_doc_ids

        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    def get_embedding(self, text: str) -> list[float]:
        """
        Get OpenAI embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.llm_client.generate_embeddings(input=text, model=self.embedding_model)
            logger.info(f"Generated embedding with {len(response)} dimensions")
            return [value for embedding in response for value in embedding]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_document_id(self, file_path: Path, metadata: dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs
        """
        # Format: mission_source_chunk_0001
        mission = metadata.get("mission", "unknown")
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", 0)
        document_id = f"{mission}_{source}_chunk_{chunk_index:04d}"
        document_id = document_id.replace(" ", "_").replace("-", "_").lower()
        return document_id

    def process_text_file(self, file_path: Path) -> list[tuple[str, dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction

        Args:
            file_path: Path to text file

        Returns:
            list of (text, metadata) tuples
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return []

            # Enhanced metadata extraction
            metadata = {
                "source": file_path.stem,
                "file_path": str(file_path),
                "file_type": "text",
                "content_type": "full_text",
                "mission": self.extract_mission_from_path(file_path),
                "data_type": self.extract_data_type_from_path(file_path),
                "document_category": self.extract_document_category_from_filename(file_path.name),
                "file_size": len(content),
                "processed_timestamp": datetime.now().isoformat(),
            }

            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if "apollo11" in path_str or "apollo_11" in path_str:
            return "apollo_11"
        elif "apollo13" in path_str or "apollo_13" in path_str:
            return "apollo_13"
        elif "challenger" in path_str:
            return "challenger"
        else:
            return "unknown"

    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if "transcript" in path_str:
            return "transcript"
        elif "textract" in path_str:
            return "textract_extracted"
        elif "audio" in path_str:
            return "audio_transcript"
        elif "flight_plan" in path_str:
            return "flight_plan"
        else:
            return "document"

    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()

        # Apollo transcript types
        if "pao" in filename_lower:
            return "public_affairs_officer"
        elif "cm" in filename_lower:
            return "command_module"
        elif "tec" in filename_lower:
            return "technical"
        elif "flight_plan" in filename_lower:
            return "flight_plan"

        # Challenger audio segments
        elif "mission_audio" in filename_lower:
            return "mission_audio"

        # NASA archive documents
        elif "ntrs" in filename_lower:
            return "nasa_archive"
        elif "19900066485" in filename_lower:
            return "technical_report"
        elif "19710015566" in filename_lower:
            return "mission_report"

        # General categories
        elif "full_text" in filename_lower:
            return "complete_document"
        else:
            return "general_document"

    def scan_text_files_only(self, base_path: str) -> list[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)

        Args:
            base_path: Base directory path

        Returns:
            list of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []

        # Define directories to scan
        data_dirs = ["apollo11", "apollo13", "challenger"]

        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")

                # Find only text files
                text_files = list(dir_path.glob("**/*.txt"))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")

        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (
                file_path.name.startswith(".")
                or "summary" in file_path.name.lower()
                or file_path.suffix.lower() != ".txt"
            ):
                continue
            filtered_files.append(file_path)

        logger.info(f"Total text files to process: {len(filtered_files)}")

        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1

        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")

        return filtered_files

    def add_documents_to_collection(
        self,
        documents: list[tuple[str, dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling

        Args:
            documents: list of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        "skip" - skip existing documents
                        "update" - update existing documents
                        "replace" - delete all existing documents from file and re-add

        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """
        if not documents:
            return {"added": 0, "updated": 0, "skipped": 0}

        if update_mode not in ["skip", "update", "replace"]:
            raise ValueError(f"Invalid update_mode: {update_mode}")

        logger.info(f"Update mode: {update_mode}")
        removed_document_ids = set()
        if update_mode == "replace":
            removed_document_ids = self.get_file_documents(file_path)
            if removed_document_ids:  # <-- Add this guard condition to prevent passing empty lists
                self.collection.delete(ids=removed_document_ids)
                logger.info(f"Deleted {len(removed_document_ids)} documents from file: {file_path}")
            removed_document_ids = set(removed_document_ids)

        added = 0
        updated = 0
        skipped = 0
        while documents:
            unique_document_ids = set()
            existing_document_ids = set()
            # Generate batch
            batch = []
            for _ in range(0, min(batch_size, len(documents))):
                text, metadata = documents.pop(0)
                document_id = self.generate_document_id(file_path, metadata)
                embedding = self.get_embedding(text)
                batch.append((document_id, embedding, metadata, text))
                unique_document_ids.add(document_id)

            if update_mode == "replace":
                # replace: use document ids removed due to file delete that exist in batch to update counts
                batch_updated = len(unique_document_ids.intersection(removed_document_ids))
                updated += batch_updated
                added += len(unique_document_ids) - batch_updated
            else:
                existing_document_ids = set(self.collection.get(ids=list(unique_document_ids))["ids"])
                if update_mode == "update":
                    # update: use existing document ids found in backend to set updated and added count
                    updated += len(existing_document_ids)
                    added += len(unique_document_ids) - len(existing_document_ids)
                else:
                    # update: use existing document ids found in backend to set skipped and added count
                    skipped += len(existing_document_ids)
                    added += len(unique_document_ids) - len(existing_document_ids)

            # Populate batch data for upsert call
            upsert_document_ids = []
            upsert_embeddings = []
            upsert_metadatas = []
            upsert_documents = []
            for document_id, embedding, metadata, text in batch:
                if update_mode == "skip" and document_id in existing_document_ids:
                    continue
                upsert_document_ids.append(document_id)
                upsert_embeddings.append(embedding)
                upsert_metadatas.append(metadata)
                upsert_documents.append(text)

            if not upsert_document_ids:
                continue
            self.collection.upsert(
                ids=upsert_document_ids,
                embeddings=upsert_embeddings,
                metadatas=upsert_metadatas,
                documents=upsert_documents,
            )
            logger.info(f"Upserted {len(upsert_document_ids)} documents")

        return {"added": added, "updated": updated, "skipped": skipped}

    def process_all_text_data(self, base_path: str, update_mode: str = "skip") -> dict[str, int]:
        """
        Process all text files and add to ChromaDB

        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        "skip" - skip existing documents (default)
                        "update" - update existing documents
                        "replace" - delete all existing documents from file and re-add

        Returns:
            Statistics about processed files
        """
        # TODO: Get files to process
        files_processed = 0
        documents_added = 0
        documents_updated = 0
        documents_skipped = 0
        errors = 0
        total_chunks = 0
        missions = {}

        logger.info(f"Update mode: {update_mode}")
        logger.info(f"Scanning directory: {base_path}")
        filepaths_to_process = self.scan_text_files_only(base_path)
        if not filepaths_to_process:
            logger.info("No text files found to process")
            return {
                "files_processed": files_processed,
                "documents_added": documents_added,
                "documents_updated": documents_updated,
                "documents_skipped": documents_skipped,
                "errors": errors,
                "total_chunks": total_chunks,
                "missions": missions,
            }
        logger.info(f"Found {len(filepaths_to_process)} text files to process")
        for index, file_path in enumerate(filepaths_to_process, 1):
            logger.info(f"Processing file {index}/{len(filepaths_to_process)}: {file_path}")
            try:
                documents = self.process_text_file(file_path)
                total_chunks += len(documents)
                file_stats = self.add_documents_to_collection(
                    documents=documents, file_path=file_path, update_mode=update_mode
                )
                documents_added += file_stats.get("added", 0)
                documents_updated += file_stats.get("updated", 0)
                documents_skipped += file_stats.get("skipped", 0)
                mission = self.extract_mission_from_path(file_path)
                if mission not in missions:
                    missions[mission] = {"files": 0, "chunks": 0, "added": 0, "updated": 0, "skipped": 0}
                missions[mission]["files"] += 1
                missions[mission]["chunks"] += len(documents)
                missions[mission]["added"] += file_stats["added"]
                missions[mission]["updated"] += file_stats["updated"]
                missions[mission]["skipped"] += file_stats["skipped"]
                logger.info(
                    f"Completed {file_path.name} (Added: {file_stats['added']}, Updated: {file_stats['updated']}, Skipped: {file_stats['skipped']})"
                )
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                errors += 1
                continue

        logger.info(
            f"Processing complete. (Files processed: {len(filepaths_to_process)}, Total chunks created: {total_chunks}"
        )
        return {
            "files_processed": files_processed,
            "documents_added": documents_added,
            "documents_updated": documents_updated,
            "documents_skipped": documents_skipped,
            "errors": errors,
            "total_chunks": total_chunks,
            "missions": missions,
        }

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the ChromaDB collection"""
        try:
            return {
                "collection_name": self.collection.name,
                "document_count": self.collection.count(),
                "metadata": self.collection.metadata or {},
                "embedding_model": self.embedding_model,
                "chroma_persist_directory": self.chroma_persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}

    def query_collection(self, query_text: str, n_results: int = 5) -> dict[str, Any]:
        """
        Query the collection for testing

        Args:
            query_text: Query text
            n_results: Number of results to return

        Returns:
            Query results
        """
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def get_collection_stats(self) -> dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            # Get all documents to analyze
            all_docs = self.collection.get()

            if not all_docs["metadatas"]:
                return {"error": "No documents in collection"}

            stats = {
                "total_documents": len(all_docs["metadatas"]),
                "missions": {},
                "data_types": {},
                "document_categories": {},
                "file_types": {},
            }

            # Analyze metadata
            for metadata in all_docs["metadatas"]:
                mission = metadata.get("mission", "unknown")
                data_type = metadata.get("data_type", "unknown")
                doc_category = metadata.get("document_category", "unknown")
                file_type = metadata.get("file_type", "unknown")

                # Count by mission
                stats["missions"][mission] = stats["missions"].get(mission, 0) + 1

                # Count by data type
                stats["data_types"][data_type] = stats["data_types"].get(data_type, 0) + 1

                # Count by document category
                stats["document_categories"][doc_category] = stats["document_categories"].get(doc_category, 0) + 1

                # Count by file type
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ChromaDB Embedding Pipeline for NASA Data")
    parser.add_argument("--data-path", default=".", help="Path to data directories")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Collection name")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--chunk-size", type=int, default=500, help="Text chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap size")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument(
        "--update-mode",
        choices=["skip", "update", "replace"],
        default="skip",
        help="How to handle existing documents: skip, update, or replace",
    )
    parser.add_argument("--test-query", help="Test query after processing")
    parser.add_argument("--stats-only", action="store_true", help="Only show collection statistics")
    parser.add_argument("--delete-source", help="Delete all documents from a specific source pattern")

    args = parser.parse_args()

    if not args.openai_key:
        logger.error("API Key missing. Provide via OPENAI_API_KEY env var or --openai-key.")
        sys.exit(1)

    # Initialize pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return

    # If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return

    # Process all data
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()

    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)

    end_time = time.time()
    processing_time = end_time - start_time

    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")

    # Mission breakdown
    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats["missions"].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(
            f"  Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}"
        )

    # Collection info
    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")

    # Test query if provided
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and "documents" in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results["documents"][0][:3]):  # Show top 3
                logger.info(f"Result {i + 1}: {doc[:200]}...")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
