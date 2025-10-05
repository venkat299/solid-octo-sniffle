"""Dependency helpers for the FastAPI application."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from job_role_analyzer import (
    Database,
    JobRoleAnalyzer,
    LLMInterface,
    TemplateRenderer,
    load_config,
)
from job_role_analyzer.embeddings import SentenceTransformerEmbeddingProvider

from .llm import HeuristicLLMClient


@lru_cache(maxsize=1)
def get_analyzer() -> JobRoleAnalyzer:
    config = load_config()
    database = Database(config.database_path)
    llm_client = _build_llm_client()
    llm_interface = LLMInterface(llm_client, TemplateRenderer())
    embedding_provider = _build_embedding_provider(config.embedding_model)
    return JobRoleAnalyzer(database, llm_interface, embedding_provider)


def _build_llm_client() -> HeuristicLLMClient:
    return HeuristicLLMClient()


def _build_embedding_provider(model_name: str) -> SentenceTransformerEmbeddingProvider | Any:
    try:
        return SentenceTransformerEmbeddingProvider(model_name)
    except ModuleNotFoundError:
        from .fallbacks import HashingEmbeddingProvider

        return HashingEmbeddingProvider()
