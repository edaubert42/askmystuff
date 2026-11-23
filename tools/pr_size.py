"""Fail when a PR changes far more than the 400 lines CLAUDE.md allows.

Counts added plus deleted lines between BASE and HEAD; lock files are generated and not counted.
The owner's `big-ok` label skips the check (see the `size` job in ci.yml).

Usage: python tools/pr_size.py [BASE]   (default origin/main)
"""

import subprocess
import sys

LIMIT = 500  # About 400 plus a margin.
IGNORED = {"uv.lock", "frontend/package-lock.json"}


def changed_lines(base: str) -> int:
    numstat = subprocess.run(
        ["git", "diff", "--numstat", f"{base}...HEAD"], check=True, capture_output=True, text=True
    ).stdout
    total = 0
    for line in numstat.splitlines():
        added, deleted, path = line.split("\t", 2)
        if path not in IGNORED and added != "-":  # "-" marks a binary file.
            total += int(added) + int(deleted)
    return total


def main(argv: list[str]) -> int:
    lines = changed_lines(argv[0] if argv else "origin/main")
    if lines > LIMIT:
        print(f"PR size: {lines} changed lines, limit {LIMIT} (lock files not counted).")
        print("Split the work into smaller issues, or the owner labels the PR `big-ok`.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
