"""Dependency helpers for the FastAPI application."""
from __future__ import annotations

from functools import lru_cache

from job_role_analyzer import (
    Database,
    JobRoleAnalyzer,
    LLMInterface,
    TemplateRenderer,
    load_config,
)
from job_role_analyzer.embeddings import SentenceTransformerEmbeddingProvider

from .llm import LLMStudioClient


@lru_cache(maxsize=1)
def get_analyzer() -> JobRoleAnalyzer:
    config = load_config()
    database = Database(config.database_path)
    llm_client = _build_llm_client(config)
    llm_interface = LLMInterface(llm_client, TemplateRenderer())
    embedding_provider = _build_embedding_provider(config.embedding_model)
    return JobRoleAnalyzer(database, llm_interface, embedding_provider)


def _build_llm_client(config) -> LLMStudioClient:
    llm_config = config.get_llm_config("job_role_analyzer")
    if not llm_config.base_url:
        raise ValueError("LLM configuration for 'job_role_analyzer' must include a base_url")
    return LLMStudioClient(
        base_url=llm_config.base_url,
        completion_path=llm_config.completion_path,
        api_key=llm_config.api_key,
        model=llm_config.model,
        timeout=llm_config.timeout,
    )


def _build_embedding_provider(model_name: str) -> SentenceTransformerEmbeddingProvider:
    return SentenceTransformerEmbeddingProvider(model_name)
