from __future__ import annotations

import json
from typing import Any, List, Sequence

from .config import load_config
from .data_models import Competency, JobRoleSummary, JobRoleWithCompetencies
from .db import Database
from .llm_interface import LLMInterface
from .similarity import EmbeddingProvider, SimilarityChecker


class JobRoleAnalyzer:
    """Coordinates job role normalization, competency extraction, and persistence."""

    def __init__(
        self,
        db: Database,
        llm_interface: LLMInterface,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.db = db
        self.llm_interface = llm_interface
        self.similarity_checker = SimilarityChecker(db, embedding_provider)
        self.config = load_config()

    def analyze(
        self,
        *,
        job_title: str,
        job_description: str,
        years_of_experience: int,
    ) -> JobRoleWithCompetencies:
        similar = self.similarity_checker.find_similar_role(job_description)
        if similar:
            existing = self.db.get_job_role_with_competencies(similar[0].job_role_id)
            if existing:
                return existing

        summary_text = self.llm_interface.run_prompt(
            "normalize_jd",
            {
                "job_title": job_title,
                "job_description": job_description,
                "years_of_experience": years_of_experience,
            },
        ).strip()

        job_role = JobRoleSummary(
            job_title=job_title,
            normalized_summary=summary_text,
            years_experience=years_of_experience,
        )

        competencies_payload = self.llm_interface.run_prompt(
            "extract_competencies",
            {
                "job_title": job_title,
                "normalized_summary": summary_text,
                "years_of_experience": years_of_experience,
                "job_description": job_description,
            },
            as_json=True,
        )

        competencies = self._parse_competencies(competencies_payload)

        embedding = self.similarity_checker.compute_embedding(job_description)
        self.db.add_job_role(job_role, competencies, embedding)
        self.similarity_checker.add_to_index(job_role, embedding)

        return JobRoleWithCompetencies(job_role=job_role, competencies=competencies)

    def _parse_competencies(self, payload: Any) -> List[Competency]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError("LLM response for competencies must be valid JSON.") from exc
        if not isinstance(payload, Sequence):
            raise ValueError("Competency payload must be a sequence of objects.")
        competencies: List[Competency] = []
        for entry in payload:
            competencies.append(Competency.model_validate(entry))
        if len(competencies) < self.config.min_competencies:
            raise ValueError(
                f"At least {self.config.min_competencies} competencies are required; received {len(competencies)}."
            )
        if len(competencies) > self.config.max_competencies:
            competencies = competencies[: self.config.max_competencies]
        return competencies
