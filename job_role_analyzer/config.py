from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - executed when PyYAML is unavailable
    yaml = None  # type: ignore[assignment]


def _parse_simple_mapping(text: str) -> Dict[str, Any]:
    """Parse a minimal subset of YAML supporting dot-delimited keys."""

    def assign(target: Dict[str, Any], path: list[str], value: Any) -> None:
        cursor = target
        for segment in path[:-1]:
            cursor = cursor.setdefault(segment, {})
            if not isinstance(cursor, dict):
                raise ValueError("Cannot assign nested configuration value to a non-mapping node.")
        cursor[path[-1]] = value

    result: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError("Configuration file must contain key/value pairs.")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("Configuration keys must be non-empty strings.")
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            value = value[1:-1]
        elif value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        else:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
        key_path = [segment.strip() for segment in key.split(".") if segment.strip()]
        if not key_path:
            raise ValueError("Configuration keys must contain at least one non-empty segment.")
        assign(result, key_path, value)
    return result


@dataclass
class LLMEndpointConfig:
    base_url: str
    completion_path: str = "/api/v1/completions"
    api_key: str | None = None
    model: str | None = None
    timeout: float = 30.0

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "LLMEndpointConfig":
        defaults: Dict[str, Any] = {
            "base_url": "",
            "completion_path": cls.completion_path,
            "api_key": None,
            "model": None,
            "timeout": cls.timeout,
        }
        defaults.update(data)
        return cls(**defaults)


@dataclass
class AnalyzerConfig:
    job_role_similarity_threshold: float = 0.85
    embedding_model: str = "BAAI/bge-small-en"
    similarity_backend: str = "faiss"
    max_competencies: int = 5
    min_competencies: int = 3
    database_path: str = "job_roles.db"
    prompts_path: str = "job_role_analyzer/prompts"
    llmstudio_base_url: str | None = None
    llmstudio_completion_path: str = "/api/v1/completions"
    llmstudio_api_key: str | None = None
    llmstudio_model: str | None = None
    llmstudio_timeout: float = 30.0
    llm_targets: Dict[str, LLMEndpointConfig] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "AnalyzerConfig":
        payload = dict(data)
        llm_targets_payload = payload.pop("llm_targets", None)

        defaults: Dict[str, Any] = {}
        for item in fields(cls):
            if item.default is not MISSING:
                defaults[item.name] = item.default
            elif item.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                defaults[item.name] = item.default_factory()
        defaults.update(payload)

        if isinstance(llm_targets_payload, dict):
            targets: Dict[str, LLMEndpointConfig] = {}
            for name, cfg in llm_targets_payload.items():
                if isinstance(cfg, dict):
                    targets[name] = LLMEndpointConfig.from_mapping(cfg)
            defaults["llm_targets"] = targets

        return cls(**defaults)

    def get_llm_config(self, target: str) -> LLMEndpointConfig:
        if target in self.llm_targets:
            return self.llm_targets[target]
        if not self.llmstudio_base_url:
            raise ValueError(
                f"No LLM configuration defined for '{target}' and no default llmstudio_base_url provided."
            )
        return LLMEndpointConfig(
            base_url=self.llmstudio_base_url,
            completion_path=self.llmstudio_completion_path,
            api_key=self.llmstudio_api_key,
            model=self.llmstudio_model,
            timeout=self.llmstudio_timeout,
        )


def load_config(path: str | os.PathLike[str] | None = None) -> AnalyzerConfig:
    config_path = Path(path or "config.yaml")
    if not config_path.exists():
        return AnalyzerConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        if yaml is None:
            payload = _parse_simple_mapping(handle.read())
        else:
            payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a mapping at the root level.")
    return AnalyzerConfig.from_mapping(payload)
