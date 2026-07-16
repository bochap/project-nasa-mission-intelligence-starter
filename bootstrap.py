# bootstrap.py
from dotenv import load_dotenv
load_dotenv()

import logging
import importlib
import os
import pkgutil
import sys
from typing import Any, TypeVar, Type

from provider.builder_registry import BuilderRegistry


# =====================================================================
# 1. Unified Global Logger Initialization
# =====================================================================
# Read the environment variable once for the entire application lifecycle
env_log_level = os.getenv("NASA_LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, env_log_level, logging.INFO)

# Setup the root logger globally. All other files importing 'logging' will inherit this.
logging.basicConfig(
    level=numeric_level,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Initialize this specific module's logger
logger = logging.getLogger("rag_system.bootstrap")

# =====================================================================
# 2. Global Unhandled Exception Hook
# =====================================================================


def global_exception_handler(exctype: type[BaseException], value: BaseException, traceback: Any) -> None:
    """
    Intercepts any unhandled exception that bubbles up to the top level.
    Logs it cleanly via the framework logging channel instead of crashing silently.
    """
    # Don't intercept KeyboardInterrupt (Ctrl+C) so users can always exit the app cleanly
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, traceback)
        return

    # Log the complete error stack trace safely to your logger files/console
    logger.critical(f"Unhandled System Exception occurred: {str(value)}", exc_info=(exctype, value, traceback))


# Overwrite Python's default emergency crash handler with our custom hook
sys.excepthook = global_exception_handler

logger.info("Global unhandled exception interceptor mounted successfully.")


# =====================================================================
# 3. Automatic Nested Plugin Discovery Engine
# =====================================================================


def load_plugins(plugins_package_name: str = "provider.plugins") -> None:
    """
    Dynamically discovers and imports all submodules inside provider/plugins/.
    Triggers decorators and logs status info.
    """
    logger.info(f"Initiating dynamic plugin scanning for package: '{plugins_package_name}'")

    try:
        package = importlib.import_module(plugins_package_name)
    except ModuleNotFoundError:
        logger.warning(f"Plugin container folder '{plugins_package_name}' was not found. Zero plugins registered.")
        return

    package_path = getattr(package, "__path__", None)
    if not package_path:
        logger.error(f"Failed to extract resource paths from package '{plugins_package_name}'.")
        return

    plugin_count = 0
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"{plugins_package_name}.{module_name}"

        if full_module_name not in sys.modules:
            try:
                importlib.import_module(full_module_name)
                logger.debug(f"Successfully mounted plugin module: {module_name}")
                plugin_count += 1
            except Exception as e:
                logger.error(f"Failed to initialize plugin file '{module_name}': {str(e)}", exc_info=True)

    logger.info(f"Plugin discovery sequence completed. Auto-mounted {plugin_count} backend plugin assets.")


# Auto-trigger plugin scanning and setup immediately upon importing this file
load_plugins()

# =====================================================================
# 4. Public Fluent Gateway Shortcuts
# =====================================================================
# Define the TypeVar for type linking
T = TypeVar("T")


def configure_llm(provider_name: str, builder_type: Type[T]) -> T:
    """Returns a fresh instance of the specified concrete LLM Builder class."""
    return BuilderRegistry().get_llm(provider_name, expected_type=builder_type)


def configure_rag(provider_name: str, builder_type: Type[T]) -> T:
    """Returns a fresh instance of the specified concrete RAG Builder class."""
    return BuilderRegistry().get_rag(provider_name, expected_type=builder_type)


def configure_ragas(provider_name: str, builder_type: Type[T]) -> T:
    """Returns a fresh instance of the specified concrete RAGAS Client Builder class."""
    return BuilderRegistry().get_ragas(provider_name, expected_type=builder_type)


def configure_ragas_metric(metric_name: str, builder_type: Type[T]) -> T:
    """Returns a fresh instance of the specified concrete LLM Builder class."""
    return BuilderRegistry().get_ragas_metric(metric_name, expected_type=builder_type)
