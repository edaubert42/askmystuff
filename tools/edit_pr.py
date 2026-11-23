"""Edit the title or the body of one pull request, and nothing else.

Agents use this instead of `gh api` so that adding a field cannot close,
retarget or merge a pull request. It sends only `title` and `body`.

Usage: python3 tools/edit_pr.py PR [--title TITLE] [--body-file FILE]
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

Run = Callable[..., subprocess.CompletedProcess[str]]


def main(argv: Sequence[str] | None = None, run: Run = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description="Edit the title or the body of one PR.")
    parser.add_argument("pr", type=int)
    parser.add_argument("--title")
    parser.add_argument("--body-file", type=Path)
    args = parser.parse_args(argv)

    payload: dict[str, str] = {}
    if args.title is not None:
        payload["title"] = args.title
    if args.body_file is not None:
        payload["body"] = args.body_file.read_text()
    if not payload:
        parser.error("give --title, --body-file or both")

    endpoint = f"repos/{{owner}}/{{repo}}/pulls/{args.pr}"
    command = ["gh", "api", "-X", "PATCH", endpoint, "--input", "-"]
    return run(command, input=json.dumps(payload), text=True).returncode


if __name__ == "__main__":
    sys.exit(main())
