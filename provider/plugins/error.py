# registry.py (or exceptions.py)
class ProviderError(Exception):
    def __init__(self, message: str, provider: str, original_exception: Exception):
        super().__init__(message)
        self.provider = provider
        self.original_exception = original_exception


class LLMGenerationError(ProviderError):
    """Raised when an LLM provider fails to generate a response."""

    def __init__(self, provider: str, message: str, original_exception: Exception):
        super().__init__(
            message=f"[{provider.upper()}] Generation failed: {message}",
            provider=provider,
            original_exception=original_exception,
        )


class RagListCollectionError(ProviderError):
    """Raised when an Rag provider fails to list collections."""

    def __init__(self, provider: str, message: str, original_exception: Exception):
        super().__init__(
            message=f"[{provider.upper()}] List collection failed: {message}",
            provider=provider,
            original_exception=original_exception,
        )


class RagGetOrCreateCollection(ProviderError):
    """Raised when an Rag provider fails to get or create collections."""

    def __init__(self, provider: str, message: str, original_exception: Exception):
        super().__init__(
            message=f"[{provider.upper()}] Get or create collection failed: {message}",
            provider=provider,
            original_exception=original_exception,
        )
