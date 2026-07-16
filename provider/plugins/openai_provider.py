from typing import Any, Self
from openai import AsyncOpenAI, OpenAI, OpenAIError
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from provider.protocol import BaseBuilder, LLMProtocol, LLMEmbedderProtocol
from provider.model import RagasAsyncClient
from provider.plugins.error import LLMGenerationError
from provider.plugins.shared import open_ai_url_override
from provider.builder_registry import BuilderRegistry


# =====================================================================
# OpenAI Implementations
# =====================================================================
class OpenAILLM(LLMProtocol, LLMEmbedderProtocol):
    _REQUIRED_GENERATE_STRING_RESPONSE_PARAMS: set[str] = {"model", "messages"}

    def __init__(self, api_key: str, base_url: str | None):
        parameters = {"api_key": api_key}
        if base_url:
            parameters["base_url"] = base_url
        self._client = OpenAI(**parameters)

    def generate_string_response(self, **kwargs: Any) -> str:
        missing_params = self._REQUIRED_GENERATE_STRING_RESPONSE_PARAMS - kwargs.keys()
        if missing_params:
            missing_str = ", ".join(f"'{p}'" for p in missing_params)
            raise ValueError(f"Missing required parameter(s) for string response generation: {missing_str}")
        try:
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except OpenAIError as e:
            raise LLMGenerationError(provider="openai", message=str(e), original_exception=e) from e

        except Exception as e:
            # 3. Catch unexpected OS/network errors
            raise LLMGenerationError(
                provider="openai", message=f"Unexpected system error: {str(e)}", original_exception=e
            )

    def generate_embeddings(self, **kwargs: Any) -> list[list[float]]:
        res = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in res.data]


class BaseOpenAIBuilder:
    """Shared parent configuration layer for all OpenAI-based builders."""

    def __init__(self) -> None:
        self._api_key: str = ""
        self._base_url: str | None = None

    def with_api_key(self, key: str) -> Self:
        self._api_key = key
        if base_url := open_ai_url_override(key):
            self._base_url = base_url
        return self


# Builders registered into their respective slots
@BuilderRegistry().register_llm("openai")
class OpenAIClientBuilder(BaseOpenAIBuilder, BaseBuilder[OpenAILLM]):
    registered_name = "openai"

    def build(self) -> OpenAILLM:
        return OpenAILLM(self._api_key, self._base_url)


@BuilderRegistry().register_ragas("async_openai")
class RagasAsyncOpenAIBuilder(BaseOpenAIBuilder, BaseBuilder[RagasAsyncClient]):
    registered_name = "async_openai"
    """Modern Ragas v0.4+ Builder using native provider factories."""

    def __init__(self) -> None:
        super().__init__()  # Automatically handles self._api_key and self._base_url
        self._llm_model: str = "gpt-3.5-turbo"
        self._embedding_model: str = "text-embedding-3-small"

    def with_models(self, llm: str, embedding: str) -> Self:
        self._llm_model = llm
        self._embedding_model = embedding
        return self

    def build(self) -> RagasAsyncClient:
        """Assembles native Python OpenAI clients into modern Ragas factories."""
        # 1. Spin up a standard Python OpenAI client using your core proxy rules
        native_client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

        # 2. Configure the LLM factory engine natively
        configured_llm = llm_factory(model=self._llm_model, client=native_client)

        # 3. Configure the Embeddings factory engine natively
        configured_embeddings = embedding_factory(provider="openai", model=self._embedding_model, client=native_client)

        # 4. Wrap both models inside your provider-managed data container
        return RagasAsyncClient(llm=configured_llm, embeddings=configured_embeddings)
