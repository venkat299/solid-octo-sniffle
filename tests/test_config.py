from job_role_analyzer import config as config_module


def test_load_config_without_yaml(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "job_role_similarity_threshold: 0.9",
                "max_competencies: 6",
                "min_competencies: 2",
                'embedding_model: "custom-model"',
            ]
        )
    )

    monkeypatch.setattr(config_module, "yaml", None)

    config = config_module.load_config(path=config_file)

    assert config.job_role_similarity_threshold == 0.9
    assert config.max_competencies == 6
    assert config.min_competencies == 2
    assert config.embedding_model == "custom-model"
    assert config.similarity_backend == "faiss"


def test_load_config_with_llm_targets(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "llm_targets:",
                "  job_role_analyzer:",
                "    base_url: \"http://192.168.0.132:1234\"",
                "    model: \"openai/gpt-oss-20b\"",
                "    timeout: 45",
            ]
        )
    )

    config = config_module.load_config(path=config_file)
    llm_cfg = config.get_llm_config("job_role_analyzer")

    assert llm_cfg.base_url == "http://192.168.0.132:1234"
    assert llm_cfg.model == "openai/gpt-oss-20b"
    assert llm_cfg.timeout == 45
    assert llm_cfg.completion_path == "/api/v1/completions"
