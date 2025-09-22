#!/usr/bin/env python3
"""
Conventional Commits validator for commit-msg hook.

Usage (pre-commit passes the commit message file path):
    python3 scripts/commit_msg_lint.py .git/COMMIT_EDITMSG

Rules (header only):
  - Allow "Merge ..." and "Revert: ..." to pass
  - Format: <type>(<scope>)?!: <subject>
  - Types: feat, fix, docs, style,
           refactor, perf, test, build,
           ci, chore, revert, deps, security
  - Header length soft limit: 100 chars
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CONVENTIONAL_RE = re.compile(
    (
        r"^(?:(?:revert|Revert):\s+)?"
        r"(?P<type>"
        r"feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|deps|security)"
        r"(?:\([\w\-.\/\s]+\))?"  # optional (scope)
        r"!?"  # optional breaking change marker !
        r":\s+"
        r".+$"
    )
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: commit_msg_lint.py <commit_message_file>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Commit message file not found: {path}", file=sys.stderr)
        return 2

    content = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        print("Empty commit message.", file=sys.stderr)
        return 1

    first_line = content.splitlines()[0].strip()

    # Allow merge commits without enforcement
    if first_line.lower().startswith("merge "):
        return 0

    # Soft length check
    if len(first_line) > 100:
        msg = "Commit header too long (" f"{len(first_line)} > 100). Keep it concise."
        print(msg, file=sys.stderr)
        return 1

    if not CONVENTIONAL_RE.match(first_line):
        help_lines = [
            "Commit message must follow Conventional Commits:",
            "  <type>(<scope>)?: <subject>",
            (
                "Types: feat, fix, docs, style, refactor, perf, test, build, "
                "ci, chore, revert, deps, security"
            ),
            f"Got: {first_line}",
        ]
        print("\n".join(help_lines), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
