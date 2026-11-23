"""Make an issue labeled `ready-for-agent` wait for the issues it is blocked by.

The owner can label a whole backlog at once: the coder starts each issue only when every
issue its `## Blocked by` section names is closed, and only if the owner added the label.

Usage:
  python3 tools/blockers.py check ISSUE       prints `ready`, or why the coder must not start
  python3 tools/blockers.py unblocked CLOSED  prints the issues CLOSED was the last blocker of

GITHUB_REPOSITORY_OWNER names who may start the coder.
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

READY = "ready-for-agent"
_SECTION = re.compile(r"^## Blocked by\s*$(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)


def blocked_by(body: str) -> list[int]:
    """The issues named in the `## Blocked by` section, in order."""
    section = _SECTION.search(body)
    return [int(n) for n in re.findall(r"#(\d+)", section.group(1))] if section else []


def _gh(run: Run, *args: str) -> Any:
    done = run(["gh", *args], capture_output=True, text=True)
    done.check_returncode()
    return json.loads(done.stdout)


def _issue(run: Run, number: int) -> dict[str, Any]:
    return _gh(run, "api", f"repos/{{owner}}/{{repo}}/issues/{number}")


def _labeled_by_owner(run: Run, number: int, owner: str) -> bool:
    """True when the label is on the issue and the owner put it there last."""
    path = f"repos/{{owner}}/{{repo}}/issues/{number}/events?per_page=100"
    by: str | None = None
    # One event per line: `--slurp` is missing from older gh versions.
    done = run(["gh", "api", "--paginate", "--jq", ".[]", path], capture_output=True, text=True)
    done.check_returncode()
    for line in done.stdout.splitlines():
        event: dict[str, Any] = json.loads(line)
        label: dict[str, Any] = event.get("label") or {}
        actor: dict[str, Any] = event.get("actor") or {}  # None for a deleted account
        if label.get("name") != READY:
            continue
        if event["event"] == "labeled":
            by = str(actor.get("login") or "")
        elif event["event"] == "unlabeled":
            by = None
    return by is not None and by.casefold() == owner.casefold()


def _completed(issue: dict[str, Any]) -> bool:
    """A blocker closed as not planned keeps blocking: what it was to deliver is missing."""
    return issue["state"] == "closed" and issue.get("state_reason") == "completed"


def check(run: Run, number: int, owner: str) -> str:
    issue = _issue(run, number)
    if issue["state"] != "open":
        return "closed"
    if not _labeled_by_owner(run, number, owner):
        return f"not labeled {READY} by the owner"
    # The coder's branch exists once it started: two blockers closed at once start it once.
    if _gh(run, "api", f"repos/{{owner}}/{{repo}}/git/matching-refs/heads/agent/issue-{number}-"):
        return "already started"
    waiting = [n for n in blocked_by(issue["body"] or "") if not _completed(_issue(run, n))]
    return f"waiting for {', '.join(f'#{n}' for n in waiting)}" if waiting else "ready"


def unblocked(run: Run, closed: int, owner: str) -> list[int]:
    if not _completed(_issue(run, closed)):
        return []
    listed = ["issue", "list", "--label", READY, "--state", "open", "--json", "number,body"]
    candidates = _gh(run, *listed, "--limit", "100")
    return [
        issue["number"]
        for issue in candidates
        if closed in blocked_by(issue["body"] or "")
        and check(run, issue["number"], owner) == "ready"
    ]


def main(argv: Sequence[str] | None = None, run: Run = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description="Make an issue wait for its blockers.")
    parser.add_argument("command", choices=["check", "unblocked"])
    parser.add_argument("issue", type=int)
    args = parser.parse_args(argv)
    owner = os.environ["GITHUB_REPOSITORY_OWNER"]
    if args.command == "check":
        print(check(run, args.issue, owner))
    else:
        for number in unblocked(run, args.issue, owner):
            print(number)
    return 0


if __name__ == "__main__":
    sys.exit(main())
