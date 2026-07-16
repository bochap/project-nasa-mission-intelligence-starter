from typing import Any, Sequence, Self
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction, OpenAIEmbeddingFunction
from chromadb.errors import ChromaError
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction

from provider.protocol import BaseBuilder, RAGProtocol, RAGCollectionProtocol
from provider.plugins.error import RagListCollectionError
from provider.plugins.shared import open_ai_url_override, require_params
from provider.builder_registry import BuilderRegistry

_REQUIRED_ADD_DOCUMENTS_PARAMS: set[str] = {"ids", "documents", "embeddings"}
_REQUIRED_GET_OR_CREATE_COLLECTION_PARAMS: set[str] = {"name"}


class ChromaPersistentRAG(RAGProtocol, RAGCollectionProtocol[Collection]):
    def __init__(self, path: str, embedding_function: EmbeddingFunction[Documents]):
        self._client = chromadb.PersistentClient(path=path)
        self._embedding_function = embedding_function

    @require_params(required_set=_REQUIRED_ADD_DOCUMENTS_PARAMS, context="adding documents")
    def add_documents(self, **kwargs) -> None:
        pass

    def list_collections(self) -> Sequence[Collection]:
        try:
            return self._client.list_collections()
        except ChromaError as e:
            raise RagListCollectionError(
                provider="chromadb", message=f"code: {e.code}, name: {e.name}, {e.message}", original_exception=e
            ) from e
        except Exception as e:
            raise RagListCollectionError(provider="chromadb", message=str(e), original_exception=e) from e

    @require_params(required_set=_REQUIRED_GET_OR_CREATE_COLLECTION_PARAMS, context="get or create collection")
    def get_or_create_collection(self, **kwargs: Any) -> Collection:
        try:
            # Always override the embedding function with the configured version
            parameters = {**kwargs, "embedding_function": self._embedding_function}
            return self._client.get_or_create_collection(**parameters)
        except ChromaError as e:
            raise RagListCollectionError(
                provider="chromadb", message=f"code: {e.code}, name: {e.name}, {e.message}", original_exception=e
            ) from e
        except Exception as e:
            raise RagListCollectionError(provider="chromadb", message=str(e), original_exception=e) from e


@BuilderRegistry().register_rag("chroma_persistent")
class ChromaPersistentRAGBuilder(BaseBuilder[ChromaPersistentRAG]):
    registered_name = "chroma_persistent"

    def __init__(self):
        self._path: str = "./chroma_db"
        self.embedding_function = DefaultEmbeddingFunction()

    def with_path(self, path: str) -> Self:
        self._path = path
        return self

    def with_openai_embedding(
        self,
        api_key: str | str = None,
        model_name: str = "text-embedding-ada-002",
        organization_id: str | None = None,
        api_base: str | None = None,
        api_type: str | None = None,
        api_version: str | None = None,
        deployment_id: str | None = None,
        default_headers: dict[str, str] | None = None,
        dimensions: int | int = None,
        api_key_env_var: str = "CHROMA_OPENAI_API_KEY",
    ) -> Self:
        if base_url := open_ai_url_override(api_key):
            api_base = base_url
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key,
            model_name,
            organization_id,
            api_base,
            api_type,
            api_version,
            deployment_id,
            default_headers,
            dimensions,
            api_key_env_var,
        )
        return self

    def build(self) -> ChromaPersistentRAG:
        return ChromaPersistentRAG(path=self._path, embedding_function=self.embedding_function)
