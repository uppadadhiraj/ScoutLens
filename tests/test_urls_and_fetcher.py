import socket

import pytest
import requests
import responses

from services.scraping import fetcher as fetcher_module
from services.scraping.fetcher import USER_AGENT, PageFetcher
from utils.errors import ExtractionError, InvalidURLError
from utils.urls import normalize_url, validate_public_url

PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Resolve every hostname to a public address unless a test overrides it (no real DNS)."""

    def fake_getaddrinfo(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    fetcher_module._last_request.clear()


def _fetcher() -> PageFetcher:
    return PageFetcher(sleep=lambda _s: None)


# ----------------------------------------------------------------------- URLs

def test_normalize_adds_https_and_strips_fragment():
    assert normalize_url("  example.com/jobs/1#apply ") == "https://example.com/jobs/1"


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "ftp://example.com/x", "javascript:alert(1)", "https://user:pw@example.com/",
     "https://example.com:8080/", "https://exa mple.com", "file:///etc/passwd", "https://", "x" * 3000],
)
def test_invalid_urls_are_rejected(bad):
    with pytest.raises(InvalidURLError):
        normalize_url(bad)


@pytest.mark.parametrize(
    "blocked",
    ["http://127.0.0.1/", "http://localhost/admin", "http://169.254.169.254/latest/meta-data/",
     "http://10.0.0.5/", "http://192.168.1.1/", "http://[::1]/", "http://printer.local/", "http://0.0.0.0/"],
)
def test_private_and_loopback_hosts_are_blocked(blocked):
    with pytest.raises(InvalidURLError):
        validate_public_url(blocked)


def test_hostname_resolving_to_private_ip_is_blocked(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))]
    )
    with pytest.raises(InvalidURLError):
        validate_public_url("https://sneaky.example.com/")


def test_hostname_with_any_private_address_is_blocked(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ],
    )
    with pytest.raises(InvalidURLError):
        validate_public_url("https://mixed.example.com/")


def test_unresolvable_host(monkeypatch):
    def boom(*a, **k):
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(InvalidURLError):
        validate_public_url("https://does-not-exist.example/")


def test_public_url_passes():
    assert validate_public_url("https://example.com/jobs") == "https://example.com/jobs"


# -------------------------------------------------------------------- fetcher

ROBOTS_ALLOW = "User-agent: *\nAllow: /\n"


@responses.activate
def test_fetch_success_sends_identifying_user_agent():
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/job", body="<html>hi</html>", content_type="text/html; charset=utf-8")
    page = _fetcher().fetch("https://example.com/job")
    assert page.status == 200 and page.content == b"<html>hi</html>" and not page.truncated
    assert responses.calls[1].request.headers["User-Agent"] == USER_AGENT


@responses.activate
def test_robots_disallow_blocks_the_page_fetch():
    responses.add(responses.GET, "https://example.com/robots.txt", body="User-agent: *\nDisallow: /jobs\n")
    with pytest.raises(ExtractionError) as info:
        _fetcher().fetch("https://example.com/jobs/1")
    assert "robots.txt" in info.value.user_message and "paste" in info.value.user_message.lower()
    assert len(responses.calls) == 1  # only robots.txt was requested


@responses.activate
def test_robots_rule_for_our_agent_is_honoured():
    responses.add(responses.GET, "https://example.com/robots.txt", body="User-agent: ScoutLens\nDisallow: /\n")
    with pytest.raises(ExtractionError):
        _fetcher().fetch("https://example.com/job")


@responses.activate
def test_missing_robots_txt_means_allowed():
    responses.add(responses.GET, "https://example.com/robots.txt", status=404)
    responses.add(responses.GET, "https://example.com/job", body="<html>ok</html>", content_type="text/html")
    assert _fetcher().fetch("https://example.com/job").status == 200


@responses.activate
def test_unverifiable_robots_txt_is_treated_as_disallow():
    responses.add(responses.GET, "https://example.com/robots.txt", status=503)
    with pytest.raises(ExtractionError):
        _fetcher().fetch("https://example.com/job")


@responses.activate
@pytest.mark.parametrize("status", [401, 403, 429, 999])
def test_access_refused_statuses_give_paste_hint(status):
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/job", status=status)
    with pytest.raises(ExtractionError) as info:
        _fetcher().fetch("https://example.com/job")
    assert str(status) in info.value.user_message and "paste" in info.value.user_message.lower()


@responses.activate
def test_404_and_500_messages():
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/gone", status=404)
    responses.add(responses.GET, "https://example.com/err", status=500)
    with pytest.raises(ExtractionError, match="not found"):
        _fetcher().fetch("https://example.com/gone")
    with pytest.raises(ExtractionError, match="error"):
        _fetcher().fetch("https://example.com/err")


@responses.activate
def test_non_html_content_is_rejected():
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/a.pdf", body=b"%PDF", content_type="application/pdf")
    with pytest.raises(ExtractionError, match="web page"):
        _fetcher().fetch("https://example.com/a.pdf")


@responses.activate
def test_oversized_body_is_truncated(monkeypatch):
    monkeypatch.setattr(fetcher_module, "MAX_BYTES", 1000)
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/big", body="a" * 50_000, content_type="text/html")
    page = _fetcher().fetch("https://example.com/big")
    assert page.truncated and len(page.content) == 1000


@responses.activate
def test_redirect_is_followed_and_revalidated():
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/old", status=301, headers={"Location": "/new"})
    responses.add(responses.GET, "https://example.com/new", body="<html>new</html>", content_type="text/html")
    assert _fetcher().fetch("https://example.com/old").url == "https://example.com/new"


@responses.activate
def test_redirect_to_internal_address_is_blocked():
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/hop", status=302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})
    with pytest.raises(InvalidURLError):
        _fetcher().fetch("https://example.com/hop")
    assert all("169.254" not in c.request.url for c in responses.calls)


@responses.activate
def test_redirect_loop_gives_up():
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/loop", status=302, headers={"Location": "/loop"})
    with pytest.raises(ExtractionError, match="redirected"):
        _fetcher().fetch("https://example.com/loop")


@responses.activate
def test_timeout_and_connection_errors_are_wrapped():
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/slow", body=requests.Timeout())
    responses.add(responses.GET, "https://example.com/down", body=requests.ConnectionError("boom"))
    with pytest.raises(ExtractionError, match="too long"):
        _fetcher().fetch("https://example.com/slow")
    with pytest.raises(ExtractionError, match="could not be reached"):
        _fetcher().fetch("https://example.com/down")


@responses.activate
def test_per_host_throttle_sleeps_between_requests():
    sleeps: list[float] = []
    now = [100.0]
    f = PageFetcher(sleep=sleeps.append, clock=lambda: now[0], min_interval=1.0)
    responses.add(responses.GET, "https://example.com/robots.txt", body=ROBOTS_ALLOW)
    responses.add(responses.GET, "https://example.com/job", body="<html>x</html>", content_type="text/html")
    f.fetch("https://example.com/job")  # robots.txt then page, clock frozen => page request must wait
    assert sleeps and sleeps[0] == pytest.approx(1.0)
