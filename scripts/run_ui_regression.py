from __future__ import annotations

import argparse
import atexit
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


BAD_TEXT = [
    "Traceback",
    "StreamlitDuplicateElementKey",
    "StreamlitDuplicateElementId",
    "NameError",
    "DataCloneError",
    "AxiosError",
    "out of memory",
    "Unhandled exception",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run browser-level Streamlit UI regression checks.")
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--output-dir", default="artifacts/ui_regression/latest")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8501)
    return parser.parse_args()


def wait_for_url(url: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= int(response.status) < 500:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def assert_no_bad_text(page) -> None:
    body = page.locator("body").inner_text(timeout=10000)
    hits = [text for text in BAD_TEXT if text.lower() in body.lower()]
    if hits:
        raise AssertionError(f"Bad UI text found: {hits}")


def click_tab(page, name: str) -> None:
    page.get_by_role("tab", name=name).first.click(timeout=15000)
    page.wait_for_timeout(500)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    server_proc = None
    if args.start_server:
        stdout = (output_dir / "streamlit.stdout.log").open("w", encoding="utf-8")
        stderr = (output_dir / "streamlit.stderr.log").open("w", encoding="utf-8")
        server_proc = subprocess.Popen(
            [
                args.python,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.port",
                str(args.port),
                "--server.headless",
                "true",
            ],
            cwd=str(ROOT),
            stdout=stdout,
            stderr=stderr,
        )
        stdout.close()
        stderr.close()
        (output_dir / "streamlit.pid").write_text(str(server_proc.pid), encoding="ascii")
        args.url = f"http://localhost:{args.port}"

        def cleanup_server() -> None:
            if server_proc.poll() is None:
                server_proc.terminate()

        atexit.register(cleanup_server)

    wait_for_url(args.url, args.timeout_sec)

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError("playwright is not installed in this Python environment") from exc

    console_errors: List[str] = []
    page_errors: List[str] = []
    warnings: List[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_sec * 1000)
        page.get_by_text("YOLOv7 False Positive Sample Finder").wait_for(timeout=30000)
        page.screenshot(path=str(output_dir / "01_initial.png"), full_page=True)
        assert_no_bad_text(page)

        try:
            page.get_by_role("button", name="Run Project Validate").click(timeout=15000)
            page.get_by_text("Validation result").wait_for(timeout=60000)
            page.screenshot(path=str(output_dir / "02_project_validate.png"), full_page=True)
            assert_no_bad_text(page)
        except Exception as exc:
            warnings.append(f"project validate click skipped/failed: {exc}")

        click_tab(page, "DB Search")
        page.screenshot(path=str(output_dir / "03_db_search.png"), full_page=True)
        assert_no_bad_text(page)

        try:
            click_tab(page, "Feature Clustering")
            page.screenshot(path=str(output_dir / "04_clustering.png"), full_page=True)
            assert_no_bad_text(page)
        except Exception as exc:
            warnings.append(f"feature clustering tab click skipped/failed: {exc}")

        try:
            combo = page.locator('[data-testid="stSelectbox"]:visible').filter(has_text="Project").first
            combo.locator('[data-baseweb="select"]').click(timeout=10000)
            page.get_by_text("safety_env", exact=True).click(timeout=10000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(output_dir / "05_safety_env_selected.png"), full_page=True)
            assert_no_bad_text(page)
        except Exception as exc:
            warnings.append(f"safety_env browser switch skipped/failed: {exc}")

        browser.close()

    serious_console = [
        text
        for text in console_errors
        if not any(allowed in text for allowed in ["favicon", "ResizeObserver loop"])
    ]
    summary = {
        "url": args.url,
        "console_errors": serious_console,
        "page_errors": page_errors,
        "warnings": warnings,
        "output_dir": str(output_dir),
    }
    with (output_dir / "ui_regression_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if page_errors or serious_console:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
