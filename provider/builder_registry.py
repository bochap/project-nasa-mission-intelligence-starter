from typing import Any, Type, TypeVar, Callable

from provider.protocol import BaseBuilder

# Create a generic type variable that represents the specific Builder class being passed in
T = TypeVar("T", bound=BaseBuilder[Any])


class BuilderRegistry:
    _instance: "BuilderRegistry | None" = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "BuilderRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._llm_builders = {}
            cls._instance._rag_builders = {}
            cls._instance._ragas_builders = {}
            cls._instance._ragas_metric_builders = {}
        return cls._instance

    def register_llm(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator mapping a plugin string name to its custom LLM builder type."""
        key = name.lower().strip()
        if key in self._llm_builders:
            raise ValueError(f"Registration Collision... '{key}'")

        def decorator(builder_cls: Type[T]) -> Type[T]:
            self._llm_builders[key] = builder_cls
            return builder_cls

        return decorator

    def register_rag(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator mapping a plugin string name to its custom RAG backend builder type."""
        key = name.lower().strip()
        if key in self._rag_builders:
            raise ValueError(f"Registration Collision... '{key}'")

        def decorator(builder_cls: Type[T]) -> Type[T]:
            self._rag_builders[key] = builder_cls
            return builder_cls

        return decorator

    def register_ragas(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator mapping a plugin string name to its custom Ragas client builder type."""
        key = name.lower().strip()
        if key in self._ragas_builders:
            raise ValueError(f"Registration Collision... '{key}'")

        def decorator(builder_cls: Type[T]) -> Type[T]:
            self._ragas_builders[key] = builder_cls
            return builder_cls

        return decorator

    def register_ragas_metric(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator mapping a plugin string name to its custom Ragas metric builder type."""
        key = name.lower().strip()
        if key in self._ragas_metric_builders:
            raise ValueError(f"Registration Collision... '{key}'")

        def decorator(builder_cls: Type[T]) -> Type[T]:
            self._ragas_metric_builders[key] = builder_cls
            return builder_cls

        return decorator

    # The factory getters instantiate and return the specific requested builder class type
    def get_llm(self, name: str, expected_type: Type[T]) -> T:
        key = name.lower().strip()
        if key not in self._llm_builders:
            raise KeyError(f"No LLM builder registered under provider token: '{name}'")
        return self._llm_builders[key]()

    def get_rag(self, name: str, expected_type: Type[T]) -> T:
        key = name.lower().strip()
        if key not in self._rag_builders:
            raise KeyError(f"No RAG builder registered under provider token: '{name}'")
        return self._rag_builders[key]()

    def get_ragas(self, name: str, expected_type: Type[T]) -> T:
        key = name.lower().strip()
        if key not in self._ragas_builders:
            raise KeyError(f"No Ragas builder registered under provider token: '{name}'")
        return self._ragas_builders[key]()

    def get_ragas_metric(self, name: str, expected_type: Type[T]) -> T:
        key = name.lower().strip()
        if key not in self._ragas_metric_builders:
            raise KeyError(f"No Ragas metric builder registered under metric: '{name}'")
        return self._ragas_metric_builders[key]()
