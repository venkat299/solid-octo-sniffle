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
