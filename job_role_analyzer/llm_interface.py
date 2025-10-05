from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Protocol

try:
    from jinja2 import Template
except ModuleNotFoundError:  # pragma: no cover - executed when Jinja2 isn't installed
    class Template:  # type: ignore[override]
        def __init__(self, content: str) -> None:
            self._content = content

        def render(self, **kwargs: Any) -> str:
            rendered = self._content
            for key, value in kwargs.items():
                rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
            return rendered

from .config import load_config


class PromptNotFoundError(FileNotFoundError):
    """Raised when a prompt file cannot be located."""


class LLMClient(Protocol):
    """Protocol for classes that can execute prompts against an LLM."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        ...


class TemplateRenderer:
    """Renders Jinja2 templates stored on disk."""

    def __init__(self, prompts_path: str | None = None) -> None:
        self._config = load_config()
        base_path = Path(prompts_path or self._config.prompts_path)
        self._base_path = base_path.resolve()

    def load(self, prompt_name: str) -> Template:
        prompt_path = self._base_path / f"{prompt_name}.txt"
        if not prompt_path.exists():
            raise PromptNotFoundError(f"Prompt template '{prompt_name}' not found at {prompt_path}")
        with prompt_path.open("r", encoding="utf-8") as handle:
            content = handle.read()
        return Template(content)


class LLMInterface:
    """Abstraction around prompt rendering and LLM execution."""

    def __init__(self, client: LLMClient, renderer: TemplateRenderer | None = None) -> None:
        self.client = client
        self.renderer = renderer or TemplateRenderer()

    def run_prompt(self, prompt_name: str, input_vars: Dict[str, Any], *, as_json: bool = False) -> Any:
        template = self.renderer.load(f"jd_analysis/{prompt_name}")
        rendered_prompt = template.render(**input_vars)
        response = self.client.complete(rendered_prompt)
        if as_json:
            return json.loads(response)
        return response
