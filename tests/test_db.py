from uuid import uuid4

from job_role_analyzer.data_models import Competency, JobRoleSummary
from job_role_analyzer.db import Database


def test_database_persists_job_role_and_competencies(tmp_path):
    db_path = tmp_path / "roles.sqlite"
    database = Database(path=str(db_path))
    try:
        job_role = JobRoleSummary(
            job_role_id=uuid4(),
            job_title="Platform Engineer",
            normalized_summary="Maintains platform infrastructure",
            years_experience=6,
        )
        competencies = [
            Competency(name="Kubernetes", level=4, type="technical"),
            Competency(name="Infrastructure", level=5, type="technical"),
            Competency(name="Collaboration", level=3, type="soft"),
        ]

        database.add_job_role(job_role, competencies, embedding=[0.1, 0.2, 0.3])

        stored = database.get_job_role_with_competencies(job_role.job_role_id)
        assert stored is not None
        assert stored.job_role == job_role
        assert stored.competencies == competencies

        embeddings = list(database.iter_job_role_embeddings())
        assert len(embeddings) == 1
        retrieved_role, vector = embeddings[0]
        assert retrieved_role.job_role_id == job_role.job_role_id
        assert vector == [0.1, 0.2, 0.3]
    finally:
        database.close()
