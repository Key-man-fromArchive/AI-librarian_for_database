"""SSE encoding, error extraction, and the pre-publication scanner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from librarian_core.sse import encode_chunk, encode_error, encode_event, extract_error_message

REPO = Path(__file__).resolve().parent.parent
SCANNER = REPO / "tools" / "scan_secrets.py"


# --- SSE --------------------------------------------------------------------


def test_chunk_blocks_are_bare_data_lines():
    block = encode_chunk("hello")
    assert block.startswith("data: ")
    assert block.endswith("\n\n")
    assert json.loads(block[6:].strip())["chunk"] == "hello"


def test_non_ascii_survives_encoding():
    assert "한글" in encode_chunk("한글")


def test_named_events_carry_their_name():
    assert encode_event("citations", {"citations": []}).startswith("event: citations\ndata: ")


def test_error_message_is_extracted_from_a_named_block():
    block = encode_error("temperature is not supported", code="http_400")
    assert extract_error_message(block) == "temperature is not supported"


def test_extraction_falls_back_to_code_then_default():
    assert extract_error_message('event: error\ndata: {"code": "rate_limited"}\n\n') == "rate_limited"
    assert extract_error_message("event: error\ndata: \n\n") == "provider error"


def test_malformed_payload_still_yields_something_readable():
    assert "not json" in extract_error_message("event: error\ndata: not json at all\n\n")


def test_extraction_of_a_block_without_data_returns_the_default():
    assert extract_error_message("event: error\n\n", default="fallback") == "fallback"


# --- scanner ----------------------------------------------------------------


def run_scanner(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCANNER), *[str(p) for p in paths]],
        capture_output=True,
        text=True,
        check=False,
    )


# Fixtures are assembled at runtime rather than written as literals: a test file
# containing a real-looking key would itself trip the scanner, and the
# repository-is-clean test below must stay meaningful.
FAKE_KEY = "sk-" + "proj-" + "abcdefghijklmnopqrstuvwxyz" + "012345"
FAKE_EMAIL = "jane.doe" + "@" + "acme-research" + ".co.kr"


def test_scanner_reports_this_repository_clean():
    """The repository must stay publishable — this is the gate that keeps it so."""
    result = run_scanner(REPO)
    assert result.returncode == 0, result.stdout


def test_scanner_catches_a_planted_key(tmp_path):
    (tmp_path / "leak.py").write_text(f'KEY = "{FAKE_KEY}"\n')
    result = run_scanner(tmp_path)
    assert result.returncode == 1
    assert "openai-key" in result.stdout


def test_scanner_catches_a_real_looking_email(tmp_path):
    (tmp_path / "contact.md").write_text(f"Reach the team at {FAKE_EMAIL}\n")
    result = run_scanner(tmp_path)
    assert result.returncode == 1
    assert "email" in result.stdout


def test_scanner_allows_documentation_placeholders(tmp_path):
    (tmp_path / "docs.md").write_text(
        'api_key="your-api-key"\ncontact: user@example.com\nsk-...\ntoken = "changeme"\n'
    )
    assert run_scanner(tmp_path).returncode == 0, run_scanner(tmp_path).stdout


def test_scanner_never_prints_a_full_secret(tmp_path):
    (tmp_path / "leak.py").write_text(f'KEY = "{FAKE_KEY}"\n')
    result = run_scanner(tmp_path)
    assert FAKE_KEY not in result.stdout


def test_scanner_honours_an_inline_justified_skip(tmp_path):
    marker = "scan-secrets" + ":allow"
    (tmp_path / "ok.py").write_text(f'KEY = "{FAKE_KEY}"  # {marker}\n')
    assert run_scanner(tmp_path).returncode == 0
