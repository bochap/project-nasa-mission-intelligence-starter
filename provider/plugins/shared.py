from typing import Any, Callable
from functools import wraps


def require_params(required_set: set[str], context: str) -> Callable[..., Any]:
    """Decorator to enforce that specific keyword arguments are present before execution."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            missing_params = required_set - kwargs.keys()
            if missing_params:
                missing_str = ", ".join(f"'{p}'" for p in missing_params)
                raise ValueError(f"Missing required parameter(s) for {context}: {missing_str}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def open_ai_url_override(api_key: str) -> str | None:
    return "https://openai.vocareum.com/v1" if api_key or api_key.startswith("voc") else None
