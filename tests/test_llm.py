import json

import pytest
import requests
import responses
from pydantic import BaseModel

from config import Settings
from services.llm.base import NullProvider, extract_json
from services.llm.factory import get_llm
from services.llm.ollama import OllamaProvider
from services.llm.openai_compat import OpenAICompatProvider
from tests.fakes import FakeLLM
from utils.errors import LLMResponseError, LLMUnavailableError
from utils.language_guard import is_neutral, violations


class Answer(BaseModel):
    label: str
    score: int


# ----------------------------------------------------------- structured output

def test_extract_json_handles_fences_and_chatter():
    assert json.loads(extract_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(extract_json('Sure! Here it is: {"a": {"b": 2}} Hope that helps')) == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_generate_structured_validates():
    llm = FakeLLM(['{"label": "ok", "score": 3}'])
    assert llm.generate_structured(Answer, "sys", "user") == Answer(label="ok", score=3)
    assert "JSON Schema" in llm.calls[0][0]


def test_generate_structured_repairs_with_the_validation_error():
    llm = FakeLLM(['{"label": "ok", "score": "high"}', '{"label": "ok", "score": 5}'])
    assert llm.generate_structured(Answer, "sys", "user").score == 5
    assert "not valid" in llm.calls[1][1]


def test_generate_structured_gives_up_after_the_repair_budget():
    llm = FakeLLM(["nonsense"] * 3)
    with pytest.raises(LLMResponseError):
        llm.generate_structured(Answer, "sys", "user", max_repairs=2)
    assert len(llm.calls) == 3


def test_unavailable_llm_error_propagates():
    with pytest.raises(LLMUnavailableError):
        FakeLLM(available=False).complete("s", "u")


def test_null_provider():
    ok, message = NullProvider().check_available()
    assert not ok and "none" in message.lower()
    with pytest.raises(LLMUnavailableError):
        NullProvider().complete("s", "u")


# --------------------------------------------------------------------- ollama

BASE = "http://localhost:11434"


def _ollama() -> OllamaProvider:
    return OllamaProvider(BASE, "llama3.1:8b", timeout=5)


@responses.activate
def test_ollama_chat_payload_and_reply():
    responses.add(responses.POST, f"{BASE}/api/chat", json={"message": {"content": "hello"}})
    assert _ollama().complete("sys", "hi") == "hello"
    body = json.loads(responses.calls[0].request.body)
    assert body["model"] == "llama3.1:8b" and body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@responses.activate
def test_ollama_schema_falls_back_to_plain_json_mode_on_400():
    responses.add(responses.POST, f"{BASE}/api/chat", status=400, json={"error": "bad format"})
    responses.add(responses.POST, f"{BASE}/api/chat", json={"message": {"content": '{"label": "x", "score": 1}'}})
    assert _ollama().generate_structured(Answer, "s", "u").label == "x"
    assert json.loads(responses.calls[1].request.body)["format"] == "json"


@responses.activate
def test_ollama_connection_error_message_is_actionable():
    responses.add(responses.POST, f"{BASE}/api/chat", body=requests.ConnectionError("refused"))
    with pytest.raises(LLMUnavailableError) as info:
        _ollama().complete("s", "u")
    assert "ollama serve" in info.value.user_message


@responses.activate
def test_ollama_missing_model_and_timeout():
    responses.add(responses.POST, f"{BASE}/api/chat", status=404, json={"error": "model not found"})
    with pytest.raises(LLMUnavailableError) as info:
        _ollama().complete("s", "u")
    assert "ollama pull llama3.1:8b" in info.value.user_message
    responses.replace(responses.POST, f"{BASE}/api/chat", body=requests.Timeout())
    with pytest.raises(LLMUnavailableError) as info:
        _ollama().complete("s", "u")
    assert "too long" in info.value.user_message


@responses.activate
def test_ollama_availability_checks_the_model_is_installed():
    responses.add(responses.GET, f"{BASE}/api/tags", json={"models": [{"name": "llama3.1:8b"}]})
    assert _ollama().check_available()[0] is True
    responses.replace(responses.GET, f"{BASE}/api/tags", json={"models": [{"name": "mistral:latest"}]})
    ok, message = _ollama().check_available()
    assert not ok and "not installed" in message
    responses.replace(responses.GET, f"{BASE}/api/tags", body=requests.ConnectionError())
    assert _ollama().check_available()[0] is False


# ------------------------------------------------------------- openai-compatible

OAI = "https://api.example.test/v1"


def _oai(key: str = "sk-secret-123456") -> OpenAICompatProvider:
    return OpenAICompatProvider(OAI, key, "gpt-x", timeout=5)


@responses.activate
def test_openai_compat_request_and_reply():
    responses.add(responses.POST, f"{OAI}/chat/completions", json={"choices": [{"message": {"content": "hey"}}]})
    assert _oai().complete("sys", "hi") == "hey"
    request = responses.calls[0].request
    assert request.headers["Authorization"] == "Bearer sk-secret-123456"
    assert json.loads(request.body)["model"] == "gpt-x"


@responses.activate
def test_openai_compat_drops_response_format_when_unsupported():
    responses.add(responses.POST, f"{OAI}/chat/completions", status=400, json={"error": {"message": "response_format unsupported"}})
    responses.add(responses.POST, f"{OAI}/chat/completions", json={"choices": [{"message": {"content": '{"label": "y", "score": 2}'}}]})
    assert _oai().generate_structured(Answer, "s", "u").score == 2
    assert "response_format" not in json.loads(responses.calls[1].request.body)


@responses.activate
@pytest.mark.parametrize("status, needle", [(401, "API key"), (429, "rate limit")])
def test_openai_compat_error_statuses(status, needle):
    responses.add(responses.POST, f"{OAI}/chat/completions", status=status, json={"error": "x"})
    with pytest.raises(LLMUnavailableError) as info:
        _oai().complete("s", "u")
    assert needle.lower() in info.value.user_message.lower()


@responses.activate
def test_openai_compat_never_leaks_the_key_in_errors():
    responses.add(responses.POST, f"{OAI}/chat/completions", body=requests.ConnectionError("boom Bearer sk-secret-123456 sk-secret-123456"))
    with pytest.raises(LLMUnavailableError) as info:
        _oai().complete("s", "u")
    assert "sk-secret-123456" not in str(info.value) + (info.value.detail or "") + info.value.user_message


def test_openai_availability_needs_a_key():
    assert _oai("").check_available()[0] is False
    assert _oai().check_available()[0] is True


# --------------------------------------------------------------------- factory

def test_factory_selects_provider(monkeypatch):
    def make(**kw):
        return get_llm(Settings(_env_file=None, **kw))

    assert isinstance(make(llm_provider="ollama"), OllamaProvider)
    assert isinstance(make(llm_provider="none"), NullProvider)
    assert isinstance(make(llm_provider="openai", openai_api_key="sk-x"), OpenAICompatProvider)
    keyless = make(llm_provider="openai")
    assert isinstance(keyless, NullProvider) and "OPENAI_API_KEY" in keyless.check_available()[1]


# --------------------------------------------------------------- language guard

@pytest.mark.parametrize(
    "text",
    [
        "This company is a scam.", "Looks fraudulent to me", "The listing is fake", "It seems suspicious",
        "You should definitely apply", "You must learn Docker", "Definitely good employer",
        "Don't apply here", "A great company", "This is a legit opportunity",
    ],
)
def test_guard_flags_verdicts(text):
    assert not is_neutral(text), text


@pytest.mark.parametrize(
    "text",
    [
        "Docker appears in 7 of 18 sampled listings.",
        "Experience requirements vary across current listings; this is a useful signal to verify.",
        "Potential discrepancy: unverified information requiring further verification.",
        "The supplied listing states that freshers are eligible.",
    ],
)
def test_guard_allows_neutral_language(text):
    assert violations(text) == []
