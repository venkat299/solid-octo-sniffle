"""Utility LLM clients for interactive experiences."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

from job_role_analyzer.llm_interface import LLMClient


class HeuristicLLMClient(LLMClient):
    """Rule-based LLM client that generates deterministic outputs for demos."""

    def complete(self, prompt: str, **_: Any) -> str:  # pragma: no cover - thin wrapper
        if "core competencies" in prompt.lower():
            return self._generate_competencies(prompt)
        return self._summarize(prompt)

    def _summarize(self, prompt: str) -> str:
        description = _extract_section(prompt, "Job Description:")
        sentences = _split_sentences(description)[:4]
        summary_parts = []
        if sentences:
            summary_parts.append(sentences[0])
        if len(sentences) > 1:
            summary_parts.append(sentences[1])
        highlighted_skills = _extract_keywords(description)[:4]
        if highlighted_skills:
            summary_parts.append(
                "Key skills include " + ", ".join(highlighted_skills) + "."
            )
        if len(sentences) > 2:
            summary_parts.append("Additional responsibilities span " + sentences[2].lower())
        return " ".join(summary_parts).strip()

    def _generate_competencies(self, prompt: str) -> str:
        summary = _extract_section(prompt, "Normalized Summary:")
        description = _extract_section(prompt, "Full Job Description:")
        keywords = _extract_keywords("\n".join([summary, description]))
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
        if not unique_keywords:
            unique_keywords = ["Communication", "Problem Solving", "Team Leadership"]
        competencies = []
        baseline_level = _infer_level(description)
        for keyword in unique_keywords[:5]:
            competencies.append(
                {
                    "name": keyword,
                    "level": baseline_level,
                    "type": _infer_type(keyword),
                }
            )
        while len(competencies) < 3:
            competencies.append(
                {
                    "name": f"Core Skill {len(competencies) + 1}",
                    "level": baseline_level,
                    "type": "technical",
                }
            )
        return json.dumps(competencies)


def _extract_section(prompt: str, heading: str) -> str:
    pattern = re.compile(rf"{re.escape(heading)}\s*(.*)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(prompt)
    if not match:
        return ""
    text = match.group(1)
    terminator_match = re.search(r"\n[A-Z][^\n]+:\s*$", text, re.MULTILINE)
    if terminator_match:
        text = text[: terminator_match.start()].strip()
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _extract_keywords(text: str) -> list[str]:
    normalized = text.lower()
    candidates = {
        "python": ["python"],
        "javascript": ["javascript", "react", "node"],
        "system design": ["architecture", "design", "system"],
        "data analysis": ["analytics", "data", "insights"],
        "cloud platforms": ["aws", "azure", "gcp", "cloud"],
        "project leadership": ["lead", "leadership", "manage"],
        "communication": ["communication", "stakeholder", "collaborate"],
        "testing": ["test", "qa", "quality"],
        "security": ["security", "compliance"],
        "devops": ["devops", "ci/cd", "pipeline"],
    }
    matched = []
    for keyword, synonyms in candidates.items():
        if any(term in normalized for term in synonyms):
            matched.append(keyword.title())
    if not matched:
        words = re.findall(r"[A-Za-z]{4,}", text)
        matched = [word.title() for word in words[:5]]
    return matched


def _infer_level(description: str) -> int:
    normalized = description.lower()
    if any(term in normalized for term in ["senior", "lead", "principal"]):
        return 5
    if "mid" in normalized or "intermediate" in normalized:
        return 4
    if "junior" in normalized or "entry" in normalized:
        return 2
    return 3


def _infer_type(keyword: str) -> str:
    technical_terms: Iterable[str] = {
        "Python",
        "Javascript",
        "System Design",
        "Cloud Platforms",
        "Testing",
        "Security",
        "Devops",
        "Data Analysis",
    }
    if keyword in technical_terms:
        return "technical"
    if keyword.lower() in {"communication", "project leadership"}:
        return "leadership"
    return "conceptual"
