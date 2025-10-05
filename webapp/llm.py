"""HTTP client integration for delegating prompts to LLMStudio."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from job_role_analyzer.llm_interface import LLMClient


logger = logging.getLogger(__name__)


class LLMStudioClient(LLMClient):
    """LLM client that forwards prompts to an LLMStudio deployment over HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        completion_path: str = "/api/v1/completions",
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required for LLMStudioClient")
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)
        self._completion_path = self._normalize_path(completion_path)
        self._api_key = api_key
        self._model = model

    def complete(self, prompt: str, **kwargs: Any) -> str:
        payload: dict[str, Any] = {}
        if self._model:
            payload["model"] = self._model

        payload.setdefault("stream", False)
        payload_messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        extra_payload = kwargs.get("extra_payload")
        if isinstance(extra_payload, dict):
            payload.update(extra_payload)
            if isinstance(extra_payload.get("messages"), list):
                payload_messages = extra_payload["messages"]  # type: ignore[assignment]

        payload.setdefault("messages", payload_messages)

        request_url = f"{self._base_url}{self._completion_path}"
        logger.info("LLMStudio POST %s", request_url)

        response = self._client.post(
            self._completion_path,
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        text = self._extract_text(response.json())
        if text is None:
            raise ValueError("Unable to parse completion text from LLMStudio response")
        return text.strip()

    def close(self) -> None:
        self._client.close()

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    @staticmethod
    def _extract_text(payload: Any) -> str | None:
        if isinstance(payload, str):
            return payload

        if isinstance(payload, dict):
            for key in ("result", "completion", "text"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value

            choices = payload.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    text = LLMStudioClient._extract_from_choice(choice)
                    if text:
                        return text

            data_entries = payload.get("data")
            if isinstance(data_entries, list):
                for entry in data_entries:
                    text = LLMStudioClient._extract_text(entry)
                    if text:
                        return text

            message = payload.get("message")
            if isinstance(message, dict):
                return LLMStudioClient._extract_message_content(message)

        if isinstance(payload, list):
            texts: list[str] = []
            for item in payload:
                text = LLMStudioClient._extract_text(item)
                if text:
                    texts.append(text)
            if texts:
                return "".join(texts)

        return None

    @staticmethod
    def _extract_from_choice(choice: Any) -> str | None:
        if not isinstance(choice, dict):
            return None
        text_value = choice.get("text")
        if isinstance(text_value, str):
            return text_value
        message = choice.get("message")
        if isinstance(message, dict):
            return LLMStudioClient._extract_message_content(message)
        delta = choice.get("delta")
        if isinstance(delta, dict):
            return LLMStudioClient._extract_message_content(delta)
        return None

    @staticmethod
    def _extract_message_content(message: dict[str, Any]) -> str | None:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_part = item.get("text")
                    if isinstance(text_part, str):
                        parts.append(text_part)
            if parts:
                return "".join(parts)
        return None
