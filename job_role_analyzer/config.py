from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - executed when PyYAML is unavailable
    yaml = None  # type: ignore[assignment]


def _parse_simple_mapping(text: str) -> Dict[str, Any]:
    """Parse a minimal subset of YAML consisting of key/value pairs."""

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
        result[key] = value
    return result


@dataclass
class AnalyzerConfig:
    job_role_similarity_threshold: float = 0.85
    embedding_model: str = "BAAI/bge-small-en"
    similarity_backend: str = "faiss"
    max_competencies: int = 5
    min_competencies: int = 3
    database_path: str = "job_roles.db"
    prompts_path: str = "job_role_analyzer/prompts"

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "AnalyzerConfig":
        kwargs = {**cls().__dict__, **data}
        return cls(**kwargs)


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
