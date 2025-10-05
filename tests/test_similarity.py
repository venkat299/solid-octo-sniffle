from uuid import uuid4

from job_role_analyzer.data_models import Competency, JobRoleSummary
from job_role_analyzer.db import Database
from job_role_analyzer.similarity import SimilarityChecker


class StaticEmbeddingProvider:
    def __init__(self, vector):
        self.vector = vector

    def embed(self, text):
        return list(self.vector)


def _store_role(database, vector):
    role = JobRoleSummary(
        job_role_id=uuid4(),
        job_title="Role",
        normalized_summary="Summary",
        years_experience=3,
    )
    competencies = [
        Competency(name="Skill A", level=3),
        Competency(name="Skill B", level=4),
        Competency(name="Skill C", level=2),
    ]
    database.add_job_role(role, competencies, embedding=vector)
    return role


def test_similarity_checker_returns_best_match(tmp_path):
    db_path = tmp_path / "similarity.db"
    database = Database(path=str(db_path))
    try:
        matching_role = _store_role(database, [1.0, 0.0, 0.0])
        _store_role(database, [0.0, 1.0, 0.0])

        checker = SimilarityChecker(database, StaticEmbeddingProvider([0.9, 0.1, 0.0]))
        best_match = checker.find_similar_role("Highly related description")

        assert best_match is not None
        job_role, score = best_match
        assert job_role.job_role_id == matching_role.job_role_id
        assert score > 0.9
    finally:
        database.close()


def test_similarity_checker_returns_none_when_below_threshold(tmp_path):
    db_path = tmp_path / "threshold.db"
    database = Database(path=str(db_path))
    try:
        _store_role(database, [1.0, 0.0, 0.0])

        checker = SimilarityChecker(database, StaticEmbeddingProvider([0.1, 0.9, 0.0]))
        assert checker.find_similar_role("Unrelated description") is None
    finally:
        database.close()
