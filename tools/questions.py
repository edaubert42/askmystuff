"""Say what comes next on an issue where the coder stopped with questions.

The coder asks in a comment starting `Questions:`. The answerer replies `Answers:`, with an
`Owner:` line for whatever only the owner can settle. Read from the issue's comments:

  answer   the coder asked for the first time: the answerer answers
  restart  everything is answered, by the answerer or by the owner: the coder starts again
  owner    a question waits for the owner (one left by the answerer, or a second round)
  none     no question waits

Usage: python3 tools/questions.py next ISSUE
GITHUB_REPOSITORY_OWNER names the owner; only the owner's comments count as answers.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

Run = Callable[..., subprocess.CompletedProcess[str]]

AGENT = "claude[bot]"
_FOR_OWNER = re.compile(r"^Owner:", re.MULTILINE)


def _comments(run: Run, number: int) -> list[dict[str, Any]]:
    path = f"repos/{{owner}}/{{repo}}/issues/{number}/comments?per_page=100"
    # One comment per line: `--slurp` is missing from older gh versions.
    done = run(["gh", "api", "--paginate", "--jq", ".[]", path], capture_output=True, text=True)
    done.check_returncode()
    return [json.loads(line) for line in done.stdout.splitlines()]


def _kind(comment: dict[str, Any], owner: str) -> str | None:
    user: dict[str, Any] = comment.get("user") or {}  # None for a deleted account
    login = str(user.get("login") or "")
    body = str(comment.get("body") or "")
    if login == AGENT and body.startswith("Questions:"):
        return "questions"
    if login == AGENT and body.startswith("Answers:"):
        return "for owner" if _FOR_OWNER.search(body) else "answered"
    if login.casefold() == owner.casefold():
        return "owner"
    return None  # anyone else's comment changes nothing


def next_step(comments: list[dict[str, Any]], owner: str) -> str:
    kinds = [k for k in (_kind(c, owner) for c in comments) if k]
    if not kinds:
        return "none"
    last = kinds[-1]
    if last == "questions":
        return "answer" if kinds.count("questions") == 1 else "owner"
    if last == "answered":
        return "restart"
    if last == "for owner":
        return "owner"
    agent = [k for k in kinds if k != "owner"]
    return "restart" if agent and agent[-1] in ("questions", "for owner") else "none"


def main(argv: Sequence[str] | None = None, run: Run = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description="Say what comes next on the coder's questions.")
    parser.add_argument("command", choices=["next"])
    parser.add_argument("issue", type=int)
    args = parser.parse_args(argv)
    print(next_step(_comments(run, args.issue), os.environ["GITHUB_REPOSITORY_OWNER"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
