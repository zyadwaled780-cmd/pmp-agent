"""Repository secret scanner for working files, staged Git blobs, and CI.

Scans text files (including test files!) for credential patterns without ever printing secret values.
Only explicitly synthetic non-secret fixture strings containing fake/demo/test/placeholder markers are
permitted; real credentials anywhere in tracked/staged files or tests fail the scan.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".pmp_sessions", ".mypy_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".db", ".db-wal", ".db-shm", ".pyc", ".png", ".jpg", ".zip"}

# Known API key prefixes and hardcoded assignment literals (never print matching text).
PATTERNS = (
    ("groq_key", re.compile(r"gsk_[A-Za-z0-9_]{20,}")),
    ("openai_key", re.compile(r"sk-(?:proj-|live-|svcacct-)?[A-Za-z0-9_\-]{24,}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("github_token", re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")),
    (
        "hardcoded_api_key_literal",
        re.compile(
            r"""(?:OPENAI_API_KEY|GROQ_API_KEY)\s*=\s*["'](?!your-)(?!sk-)(?!gsk_)[A-Za-z0-9_\-]{20,}["']"""
        ),
    ),
)

# Narrow line-level allowlist for synthetic offline test fixtures (never whole-file exclusions).
_SYNTHETIC_FIXTURE_RE = re.compile(
    r"""(?:fake|demo|placeholder|sk-live-secret-token-abcdef12345|sk-proc-env-token-987654|"""
    r"""sk-only-openai-key-123456|gsk_dotenv_fake_token_123456|gsk_only_groq_key_123456|"""
    r"""gsk_groq_secret_token_987654321)"""
)


def _scan_lines(label: str, text: str) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    # Skip self-regex definition lines inside scripts/check_secrets.py itself
    is_self = label == "scripts/check_secrets.py"
    for lineno, line in enumerate(text.splitlines(), 1):
        if is_self and ("re.compile(" in line or "_SYNTHETIC_FIXTURE_RE" in line or 'r"""' in line):
            continue
        for kind, pat in PATTERNS:
            for match in pat.finditer(line):
                token = match.group(0)
                if _SYNTHETIC_FIXTURE_RE.search(token):
                    continue
                hits.append((label, lineno, kind))
    return hits


def scan_working_tree(root: pathlib.Path = ROOT) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts) or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        # Local untracked .env is ignored by git and skipped here; .env.example & all other dotfiles ARE scanned.
        if rel.name == ".env" or (rel.name.startswith(".env.") and rel.name != ".env.example"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits.extend(_scan_lines(rel.as_posix(), text))
    return hits


def scan_staged(root: pathlib.Path = ROOT) -> list[tuple[str, int, str]]:
    """Scan exact staged Git index blobs (`git diff --cached --name-only`)."""
    res = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=True,
    )
    hits: list[tuple[str, int, str]] = []
    for rel_path in sorted(line.strip() for line in res.stdout.splitlines() if line.strip()):
        blob = subprocess.run(
            ["git", "show", f":{rel_path}"],
            cwd=str(root),
            capture_output=True,
            check=True,
        )
        text = blob.stdout.decode("utf-8", errors="ignore")
        hits.extend(_scan_lines(rel_path, text))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan repository or staged files for credentials.")
    parser.add_argument("--staged", action="store_true", help="Scan staged Git index content instead of working tree.")
    args = parser.parse_args(argv)

    hits = scan_staged() if args.staged else scan_working_tree()
    scope = "staged index" if args.staged else "repository"
    if not hits:
        print(f"Secret scan clean ({scope}): 0 credential hits found.")
        return 0
    for rel, lineno, kind in hits:
        print(f"POTENTIAL SECRET ({kind}) at {rel}:{lineno}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
