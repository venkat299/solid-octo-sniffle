from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Sequence
from uuid import UUID

from .config import load_config
from .data_models import Competency, JobRoleSummary, JobRoleWithCompetencies


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS job_roles (
        job_role_id TEXT PRIMARY KEY,
        job_title TEXT NOT NULL,
        normalized_summary TEXT NOT NULL,
        years_experience INTEGER NOT NULL,
        embedding_vector TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS competencies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_role_id TEXT NOT NULL,
        name TEXT NOT NULL,
        level INTEGER NOT NULL,
        type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_role_id) REFERENCES job_roles(job_role_id)
    )
    """,
)


class Database:
    def __init__(self, path: str | None = None) -> None:
        config = load_config()
        db_path = Path(path or config.database_path)
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connection:
            for statement in SCHEMA_STATEMENTS:
                self._connection.execute(statement)

    def close(self) -> None:
        self._connection.close()

    def add_job_role(
        self,
        job_role: JobRoleSummary,
        competencies: Sequence[Competency],
        embedding: Sequence[float] | None = None,
    ) -> None:
        payload = (
            str(job_role.job_role_id),
            job_role.job_title,
            job_role.normalized_summary,
            job_role.years_experience,
            json.dumps(list(embedding)) if embedding is not None else None,
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO job_roles (
                    job_role_id, job_title, normalized_summary, years_experience, embedding_vector
                ) VALUES (?, ?, ?, ?, ?)
                """,
                payload,
            )
            self._connection.execute(
                "DELETE FROM competencies WHERE job_role_id = ?",
                (str(job_role.job_role_id),),
            )
            competency_rows = [
                (str(job_role.job_role_id), comp.name, comp.level, comp.type)
                for comp in competencies
            ]
            self._connection.executemany(
                """
                INSERT INTO competencies (job_role_id, name, level, type)
                VALUES (?, ?, ?, ?)
                """,
                competency_rows,
            )

    def iter_job_role_embeddings(self) -> Iterable[tuple[JobRoleSummary, List[float]]]:
        cursor = self._connection.execute(
            "SELECT job_role_id, job_title, normalized_summary, years_experience, embedding_vector FROM job_roles"
        )
        rows = cursor.fetchall()
        for row in rows:
            embedding = json.loads(row["embedding_vector"]) if row["embedding_vector"] else []
            job_role = JobRoleSummary(
                job_role_id=UUID(row["job_role_id"]),
                job_title=row["job_title"],
                normalized_summary=row["normalized_summary"],
                years_experience=row["years_experience"],
            )
            yield job_role, embedding

    def get_job_role_with_competencies(self, job_role_id: UUID) -> JobRoleWithCompetencies | None:
        cursor = self._connection.execute(
            "SELECT job_role_id, job_title, normalized_summary, years_experience FROM job_roles WHERE job_role_id = ?",
            (str(job_role_id),),
        )
        job_row = cursor.fetchone()
        if job_row is None:
            return None
        job_role = JobRoleSummary(
            job_role_id=UUID(job_row["job_role_id"]),
            job_title=job_row["job_title"],
            normalized_summary=job_row["normalized_summary"],
            years_experience=job_row["years_experience"],
        )
        comp_cursor = self._connection.execute(
            "SELECT name, level, type FROM competencies WHERE job_role_id = ? ORDER BY id",
            (str(job_role_id),),
        )
        competencies = [
            Competency(name=row["name"], level=row["level"], type=row["type"])
            for row in comp_cursor.fetchall()
        ]
        return JobRoleWithCompetencies(job_role=job_role, competencies=competencies)
