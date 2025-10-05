import json
from uuid import uuid4

from job_role_analyzer.analyzer import JobRoleAnalyzer
from job_role_analyzer.data_models import Competency, JobRoleSummary
from job_role_analyzer.db import Database


class StaticEmbeddingProvider:
    def __init__(self, vector):
        self.vector = list(vector)

    def embed(self, text):  # noqa: D401 - simple protocol implementation
        return list(self.vector)


class RecordingLLMInterface:
    def __init__(self, normalize_response="", competencies=None):
        self.normalize_response = normalize_response
        self.competencies = competencies or []
        self.calls = []

    def run_prompt(self, prompt_name, input_vars, *, as_json=False):
        self.calls.append((prompt_name, input_vars, as_json))
        if prompt_name == "normalize_jd":
            return self.normalize_response
        if prompt_name == "extract_competencies":
            payload = self.competencies
            if as_json:
                return payload
            return json.dumps(payload)
        raise AssertionError(f"Unexpected prompt {prompt_name}")


class FailingLLMInterface:
    def run_prompt(self, *args, **kwargs):  # noqa: D401 - simple stub
        raise AssertionError("LLM should not be invoked when a similar role exists")


def _sample_competencies():
    return [
        {"name": "Python", "level": 4, "type": "technical"},
        {"name": "System Design", "level": 3, "type": "technical"},
        {"name": "Leadership", "level": 2, "type": "soft"},
    ]


def test_analyzer_reuses_existing_role(tmp_path):
    db_path = tmp_path / "roles.db"
    database = Database(path=str(db_path))
    try:
        existing_role = JobRoleSummary(
            job_role_id=uuid4(),
            job_title="Senior Backend Engineer",
            normalized_summary="Existing summary",
            years_experience=7,
        )
        existing_competencies = [Competency.model_validate(item) for item in _sample_competencies()]
        database.add_job_role(existing_role, existing_competencies, embedding=[1.0, 0.0, 0.0])

        analyzer = JobRoleAnalyzer(
            database,
            FailingLLMInterface(),
            StaticEmbeddingProvider([1.0, 0.0, 0.0]),
        )

        result = analyzer.analyze(
            job_title="Senior Backend Engineer",
            job_description="Deep expertise in Python services",
            years_of_experience=7,
        )

        assert result.job_role.job_role_id == existing_role.job_role_id
        assert [comp.model_dump() for comp in result.competencies] == [
            comp.model_dump() for comp in existing_competencies
        ]
    finally:
        database.close()


def test_analyzer_creates_new_role_and_persists(tmp_path):
    db_path = tmp_path / "new_roles.db"
    database = Database(path=str(db_path))
    try:
        competencies = _sample_competencies()
        llm = RecordingLLMInterface(
            normalize_response="Summarized role description",
            competencies=competencies,
        )
        analyzer = JobRoleAnalyzer(
            database,
            llm,
            StaticEmbeddingProvider([0.3, 0.4, 0.5]),
        )

        result = analyzer.analyze(
            job_title="Data Scientist",
            job_description="Analyze datasets and build predictive models",
            years_of_experience=5,
        )

        assert result.job_role.job_title == "Data Scientist"
        assert len(result.competencies) == 3
        assert llm.calls[0][0] == "normalize_jd"
        assert llm.calls[1][0] == "extract_competencies"

        stored = database.get_job_role_with_competencies(result.job_role.job_role_id)
        assert stored is not None
        assert stored.job_role.normalized_summary == "Summarized role description"
        assert [comp.model_dump() for comp in stored.competencies] == competencies
    finally:
        database.close()
