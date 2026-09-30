from config import Settings
from utils.errors import ScoutLensError, SerpApiError


def _settings(**env: str) -> Settings:
    # _env_file=None so a developer's real .env never leaks into tests
    return Settings(_env_file=None, **env)


def test_defaults_have_no_key():
    s = _settings()
    assert not s.has_serpapi_key
    assert s.serpapi_key_value() is None
    assert s.llm_provider == "ollama"


def test_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "  abc123  ")
    s = _settings()
    assert s.has_serpapi_key
    assert s.serpapi_key_value() == "abc123"


def test_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "   ")
    assert not _settings().has_serpapi_key


def test_public_status_and_repr_never_expose_secrets(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "super-secret-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-also-secret")
    s = _settings()
    assert "super-secret-key" not in repr(s)
    assert "sk-also-secret" not in repr(s)
    assert "super-secret-key" not in str(s.public_status())
    assert s.public_status()["serpapi_configured"] is True


def test_errors_carry_user_safe_message():
    err = SerpApiError(kind="invalid_key", detail="401 for url ...api_key=SECRET")
    assert "SECRET" not in err.user_message
    assert err.is_fatal
    assert isinstance(err, ScoutLensError)
