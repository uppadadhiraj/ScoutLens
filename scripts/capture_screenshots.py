"""Capture README screenshots by driving the real app through the demo flow.

Optional developer tool (not needed to run ScoutLens):

    pip install playwright          # uses your installed Chrome/Edge, no browser download
    python scripts/capture_screenshots.py [--out docs/screenshots] [--browser chrome|msedge]

The demo uses fictional, recorded data, so nothing here contacts SerpApi or needs an API key.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8765
TABS = ["Overview", "Company", "Hiring & market", "Personal fit", "Evidence & method"]


def wait_for_server(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as resp:
                if resp.read() == b"ok":
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("Streamlit did not start in time")


def snap(page, path: Path, width: int = 1360, max_height: int = 7000) -> None:
    """Resize the viewport to the content height (Streamlit scrolls an inner container), then capture."""
    page.set_viewport_size({"width": width, "height": 900})  # reset so a shorter page is not padded by the previous one
    page.wait_for_timeout(700)
    height = page.evaluate("() => (document.querySelector('[data-testid=\"stMain\"]') || document.body).scrollHeight")
    page.set_viewport_size({"width": width, "height": max(700, min(int(height) + 40, max_height))})
    page.wait_for_timeout(300)
    page.screenshot(path=str(path), type="jpeg", quality=82)
    print("saved", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    parser.add_argument("--browser", default="chrome", choices=["chrome", "msedge"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    env = {**os.environ, "DB_PATH": str(Path(tempfile.mkdtemp()) / "shots.db"), "LLM_PROVIDER": "none", "SERPAPI_KEY": ""}
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{PORT}"
    try:
        wait_for_server(url)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.browser, headless=True)
            page = browser.new_page(viewport={"width": 1360, "height": 900})
            page.goto(url)
            page.get_by_text("before you invest your time.").wait_for(timeout=30000)
            snap(page, out / "01-home.jpg")

            page.get_by_role("button", name="Try the demo").click()
            page.get_by_role("button", name="Open report →").wait_for(timeout=60000)
            snap(page, out / "02-investigation.jpg")

            page.get_by_role("button", name="Open report →").click()
            page.get_by_role("tab", name="Overview").wait_for(timeout=30000)
            for index, name in enumerate(TABS, start=3):
                page.get_by_role("tab", name=name).click()
                cap = 2700 if name == "Evidence & method" else 7000  # the evidence list is long; keep the top section
                snap(page, out / f"{index:02d}-report-{name.lower().replace(' & ', '-').replace(' ', '-')}.jpg", max_height=cap)

            page.get_by_text("History", exact=True).first.click()
            page.get_by_text("Previous investigations").wait_for(timeout=15000)
            snap(page, out / "08-history.jpg")
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
