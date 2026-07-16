from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, Sequence, TypeVar


T = TypeVar("T")
TCollection = TypeVar("TCollection", covariant=True)


class LLMProtocol(Protocol):
    def generate_string_response(self, **kwargs: Any) -> str: ...


class LLMEmbedderProtocol(Protocol):
    def generate_embeddings(self, **kwargs: Any) -> list[list[float]]: ...


class RAGProtocol(Protocol):
    def add_documents(self, **kwargs: Any) -> None: ...


class RAGCollectionProtocol(Protocol, Generic[TCollection]):
    def list_collections(self) -> Sequence[TCollection]: ...
    def get_or_create_collection(self, **kwargs: Any) -> TCollection: ...


class BaseBuilder(ABC, Generic[T]):
    @abstractmethod
    def build(self) -> T: ...
