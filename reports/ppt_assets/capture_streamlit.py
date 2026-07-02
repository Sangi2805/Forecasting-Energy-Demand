"""Capture a full-page screenshot of the running Streamlit app."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "streamlit_app.png"
PORT = 8502
URL = f"http://localhost:{PORT}"


def capture() -> Path:
    streamlit = BASE_DIR / ".venv" / "bin" / "streamlit"
    proc = subprocess.Popen(
        [
            str(streamlit),
            "run",
            "app/streamlit_app.py",
            "--server.headless=true",
            f"--server.port={PORT}",
            "--browser.gatherUsageStats=false",
        ],
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        from playwright.sync_api import sync_playwright

        deadline = time.time() + 45
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            while time.time() < deadline:
                try:
                    page.goto(URL, wait_until="networkidle", timeout=8000)
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("Streamlit did not become ready in time")

            page.wait_for_timeout(3000)
            for selector in (
                "summary:has-text('Compare all models')",
                "[data-testid='stExpander'] summary",
                "text=Compare all models",
            ):
                try:
                    page.locator(selector).first.click(timeout=2500)
                    page.wait_for_timeout(2000)
                    break
                except Exception:
                    continue

            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(OUT), full_page=True)
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    if not OUT.exists():
        raise RuntimeError("Screenshot was not created")
    print(f"Saved Streamlit screenshot → {OUT}")
    return OUT


if __name__ == "__main__":
    try:
        capture()
    except Exception as exc:
        print(f"Streamlit capture failed: {exc}", file=sys.stderr)
        sys.exit(1)
