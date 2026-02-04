"""
LLMSummarizer PluMA Plugin Package

Local LLM-based summarization of pipeline findings for multi-omics analysis.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from .domains import DomainConfig, DOMAIN_CONFIGS, get_domain_config, list_domains
from .hardware import (
    HardwareInfo,
    ModelRecommendation,
    MODEL_RECOMMENDATIONS,
    detect_hardware,
    get_system_ram,
    detect_gpu,
    select_best_model,
    get_model_size_estimate,
)
from .ollama_client import (
    OllamaClient,
    is_ollama_running,
    start_ollama_server,
    ensure_ollama_running,
    list_local_models,
    is_model_available,
    download_model,
)
from .rag import (
    ensure_rag_database,
    download_rag_database,
    query_literature_database,
    format_literature_context,
    detect_github_repo,
)
from .LLMSummarizer import LLMSummarizer

__all__ = [
    # Main plugin
    "LLMSummarizer",
    # Domain configuration
    "DomainConfig",
    "DOMAIN_CONFIGS",
    "get_domain_config",
    "list_domains",
    # Hardware detection
    "HardwareInfo",
    "ModelRecommendation",
    "MODEL_RECOMMENDATIONS",
    "detect_hardware",
    "get_system_ram",
    "detect_gpu",
    "select_best_model",
    "get_model_size_estimate",
    # Ollama client
    "OllamaClient",
    "is_ollama_running",
    "start_ollama_server",
    "ensure_ollama_running",
    "list_local_models",
    "is_model_available",
    "download_model",
    # RAG
    "ensure_rag_database",
    "download_rag_database",
    "query_literature_database",
    "format_literature_context",
    "detect_github_repo",
]

__version__ = "0.1.0"
