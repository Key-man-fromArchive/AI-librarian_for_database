#!/usr/bin/env python3
"""Pre-publication scanner for secrets, personal data, and internal identifiers.

This repository exists because a working system was extracted from a private
codebase. That extraction nearly published, in one pass:

* an evaluation goldset containing real product formulations and experimental
  values from the origin company;
* a real employee's name, used as an example inside a docstring;
* internal hostnames in configuration defaults.

None of it looked like a secret. No key, no password, no token — which is
exactly why a key-shaped-string scanner would have passed it. The categories
below are chosen to catch what that class of tool misses.

Usage::

    python tools/scan_secrets.py            # scan the working tree
    python tools/scan_secrets.py --staged   # scan staged files only
    python tools/scan_secrets.py path/...   # scan specific paths

Exit code 1 means findings. Wire it into pre-commit and CI.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
}
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".whl",
    ".so",
    ".dylib",
    ".dll",
    ".pyc",
    ".lock",
    ".bin",
    ".woff",
    ".woff2",
}
MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    severity: str
    hint: str


RULES: list[Rule] = [
    # --- Credentials -------------------------------------------------------
    Rule(
        "openai-key",
        re.compile(r"\bsk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{20,}"),
        "critical",
        "OpenAI-style API key",
    ),
    Rule("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), "critical", "Anthropic API key"),
    Rule("google-key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"), "critical", "Google API key"),
    Rule("aws-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "critical", "AWS access key id"),
    Rule("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "critical", "GitHub token"),
    Rule("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "critical", "Slack token"),
    Rule(
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
        "critical",
        "private key block",
    ),
    Rule(
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "critical",
        "JWT — may embed real claims",
    ),
    Rule(
        "db-url",
        re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb)(?:\+\w+)?://[^\s:@/\"']+:[^\s@\"']+@"),
        "critical",
        "connection string with a password",
    ),
    # The capture group isolates the *value*, so the allowlist judges the secret
    # rather than the whole assignment line.
    Rule(
        "assigned-secret",
        re.compile(r"(?i)\b(?:api[_-]?key|secret|passwd|password|token)\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']"),
        "high",
        "hard-coded credential (placeholders are fine — check the value)",
    ),
    # --- Personal data -----------------------------------------------------
    Rule("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "high", "email address"),
    Rule("krn-phone", re.compile(r"\b01[016-9][-. ]?\d{3,4}[-. ]?\d{4}\b"), "high", "phone number"),
    Rule("krn-rrn", re.compile(r"\b\d{6}[-\s]?[1-4]\d{6}\b"), "critical", "national identification number"),
    Rule(
        "credit-card",
        re.compile(r"\b(?:4\d{3}|5[1-5]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        "critical",
        "payment card number",
    ),
    # --- Internal infrastructure -------------------------------------------
    Rule(
        "private-ip",
        re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
        "medium",
        "private network address",
    ),
    Rule(
        "internal-host",
        re.compile(r"\b[a-z0-9][a-z0-9.-]*\.(?:home\.arpa|internal|intranet|corp|lan)\b"),
        "medium",
        "internal hostname",
    ),
    Rule("onion", re.compile(r"\b[a-z2-7]{16,56}\.onion\b"), "medium", "hidden service address"),
]

#: Values that legitimately match a rule. Keep this list short and specific —
#: an over-broad allowlist is how the next leak gets through.
ALLOWLIST = {
    "sk-or-...",
    "sk-ant-...",
    "your-api-key",
    "your-key",
    "sk-...",
    "user@example.com",
    "owner@example.com",
    "you@example.com",
    "bob@example.com",
    "alice@example.com",
    "admin@example.com",
    "noreply@example.com",
    "test@example.com",
    "grantee@example.com",
    "name@example.com",
}
ALLOWLIST_DOMAINS = ("example.com", "example.org", "example.net", "localhost", "test.invalid")

#: Lines carrying this marker are skipped. Justify every use in review.
INLINE_SKIP = "scan-secrets:allow"


@dataclass
class Finding:
    path: Path
    line_no: int
    rule: Rule
    excerpt: str


def _is_allowlisted(match: str, rule: Rule) -> bool:
    lowered = match.lower()
    if lowered in {a.lower() for a in ALLOWLIST}:
        return True
    if rule.name == "email" and lowered.endswith(ALLOWLIST_DOMAINS):
        return True
    # Documented placeholders: "sk-...", "AIza<your key>", "token = \"changeme\"",
    # "your-api-key", "YOUR_TOKEN_HERE".
    return bool(
        re.search(
            # Kept deliberately narrow: broad words like "example" or "test"
            # would allowlist real values that merely contain them.
            r"(\.\.\.|<[^>]+>|xxx+|changeme|placeholder|redacted|\byour[-_]|_here\b|dummy)",
            lowered,
        )
    )


def _redact(text: str) -> str:
    """Never echo a full secret — the report itself would become one."""
    stripped = text.strip()
    if len(stripped) <= 12:
        return stripped
    return f"{stripped[:6]}…{stripped[-4:]} ({len(stripped)} chars)"


def scan_file(path: Path) -> list[Finding]:
    try:
        if path.stat().st_size > MAX_BYTES:
            return []
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    findings: list[Finding] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        if INLINE_SKIP in line:
            continue
        for rule in RULES:
            for match in rule.pattern.findall(line):
                text = match if isinstance(match, str) else next((m for m in match if m), "")
                if not text or _is_allowlisted(text, rule):
                    continue
                findings.append(Finding(path, line_no, rule, _redact(text)))
    return findings


def iter_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for base in paths:
        if base.is_file():
            out.append(base)
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            out.append(path)
    return out


def staged_files() -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [Path(p) for p in result.stdout.split("\n") if p.strip() and Path(p).is_file()]


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(".")])
    parser.add_argument("--staged", action="store_true", help="scan staged files only")
    parser.add_argument(
        "--fail-on",
        choices=["critical", "high", "medium"],
        default="medium",
        help="minimum severity that fails the run (default: medium)",
    )
    args = parser.parse_args()

    files = staged_files() if args.staged else iter_files(args.paths or [Path(".")])
    findings: list[Finding] = []
    for path in files:
        findings.extend(scan_file(path))

    if not findings:
        print(f"clean — {len(files)} files scanned, no findings")
        return 0

    findings.sort(key=lambda f: (SEVERITY_ORDER[f.rule.severity], str(f.path), f.line_no))
    print(f"{len(findings)} finding(s) across {len({f.path for f in findings})} file(s):\n")
    for finding in findings:
        print(f"  [{finding.rule.severity:8}] {finding.path}:{finding.line_no}")
        print(f"             {finding.rule.name} — {finding.rule.hint}")
        print(f"             {finding.excerpt}\n")

    threshold = SEVERITY_ORDER[args.fail_on]
    blocking = [f for f in findings if SEVERITY_ORDER[f.rule.severity] <= threshold]
    if blocking:
        print(f"FAILED: {len(blocking)} finding(s) at or above '{args.fail_on}'.")
        print(f"If a finding is a false positive, add '{INLINE_SKIP}' to the line and say why in review.")
        return 1

    print("No findings above the failure threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
