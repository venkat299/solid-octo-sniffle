import json

from job_role_analyzer.llm_interface import LLMInterface


class EchoClient:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        response = [{"name": "Python", "level": 5, "type": "technical"}]
        return json.dumps(response)


def test_llm_interface_renders_prompt_and_parses_json():
    client = EchoClient()
    interface = LLMInterface(client)

    result = interface.run_prompt(
        "extract_competencies",
        {
            "job_title": "ML Engineer",
            "normalized_summary": "Builds ML systems",
            "years_of_experience": 4,
            "job_description": "Design machine learning models",
        },
        as_json=True,
    )

    assert result == [{"name": "Python", "level": 5, "type": "technical"}]
    assert "ML Engineer" in client.prompts[0]
    assert "Design machine learning models" in client.prompts[0]
