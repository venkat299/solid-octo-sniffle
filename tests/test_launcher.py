import types

import pytest

from webapp import launcher


class DummyResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class DummyConfig:
    def __init__(self, base_url: str = "http://service") -> None:
        self.base_url = base_url
        self.completion_path = "/api/v1/completions"
        self.api_key = None
        self.model = None
        self.timeout = 1.0


def test_wait_for_llm_success(monkeypatch):
    responses = [DummyResponse(503), DummyResponse(200)]

    def fake_request(url, timeout):
        return responses.pop(0)

    launcher._wait_for_llm(  # noqa: SLF001 - invoking helper for coverage
        DummyConfig(),
        retry_interval=0,
        max_attempts=5,
        request_factory=fake_request,
    )


def test_wait_for_llm_raises(monkeypatch):
    def fake_request(url, timeout):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        launcher._wait_for_llm(
            DummyConfig(),
            retry_interval=0,
            max_attempts=2,
            request_factory=fake_request,  # type: ignore[arg-type]
        )


def test_main_exits_when_llm_unavailable(monkeypatch):
    dummy_config = DummyConfig()

    def fake_load_config():
        namespace = types.SimpleNamespace()
        namespace.get_llm_config = lambda _: dummy_config
        return namespace

    def failing_wait(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(launcher, "load_config", fake_load_config)
    monkeypatch.setattr(launcher, "_wait_for_llm", failing_wait)

    with pytest.raises(SystemExit):
        launcher.main(["--llm-attempts", "1"])
