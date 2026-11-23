"""Fail when a PR description closes no issue or is too long for the owner to read.

Without `Closes #n` the reviewer has no acceptance criteria to check the PR against.
Details belong in docstrings, tests or an ADR, not in the description.
Mermaid diagrams and HTML comments (the template's guidance) are not counted.

Usage: python tools/pr_description.py FILE
"""

import re
import sys
from pathlib import Path

LIMIT = 150

_MERMAID = re.compile(r"^```mermaid$.*?^```$", re.MULTILINE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CLOSES = re.compile(r"^Closes #\d+\s*$", re.MULTILINE)


def count_words(text: str) -> int:
    text = _COMMENT.sub(" ", _MERMAID.sub(" ", text))
    return sum(1 for token in text.split() if re.search(r"\w", token))


def has_closing_line(text: str) -> bool:
    return _CLOSES.search(_COMMENT.sub(" ", text)) is not None


def main(path: str) -> int:
    text = Path(path).read_text(encoding="utf-8")
    failed = False
    if not has_closing_line(text):
        print("PR description: no `Closes #n` line with an issue number.")
        print("Open an issue for this work and close it with `Closes #n` on the first line.")
        failed = True
    words = count_words(text)
    if words > LIMIT:
        print(f"PR description: {words} words, limit {LIMIT} (Mermaid diagrams not counted).")
        print("Keep what the owner must know; move the rest to docstrings, tests or an ADR.")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
