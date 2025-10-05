from webapp.llm import LLMStudioClient


def test_extract_text_from_standard_choices():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello there",
                }
            }
        ]
    }

    assert LLMStudioClient._extract_text(payload) == "Hello there"


def test_extract_text_from_segmented_content():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Always"},
                        {"type": "output_text", "text": " rhyme"},
                    ],
                }
            }
        ]
    }

    assert LLMStudioClient._extract_text(payload) == "Always rhyme"


def test_complete_sends_chat_messages(monkeypatch):
    sent: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "Hi"}}]}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            sent["init"] = {"args": args, "kwargs": kwargs}

        def post(self, path, json, headers):
            sent["path"] = path
            sent["json"] = json
            sent["headers"] = headers
            return DummyResponse()

        def close(self) -> None:
            sent["closed"] = True

    monkeypatch.setattr("webapp.llm.httpx.Client", DummyClient)

    client = LLMStudioClient("http://service", model="openai/gpt-oss-20b")
    client.complete("Describe the day")

    assert sent["path"] == "/api/v1/completions"
    payload = sent["json"]
    assert isinstance(payload, dict)
    assert payload["messages"][0]["content"] == "Describe the day"
    assert payload["messages"][0]["role"] == "user"
    assert payload["model"] == "openai/gpt-oss-20b"
    assert payload["stream"] is False
