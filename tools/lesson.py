"""Gather what went wrong on merged PRs, and open the learner agent's lessons as issues.

The learner agent (.github/workflows/agent-learner.yml) only reads. `history` writes what it
reads: on recent merged PRs, the reviewer's `changes needed` findings, the fixer's `Fix CI:`
commits and the owner's `@claude` corrections, and nobody else's text; then every `lesson`
issue with its kind. `open` runs after the agent, without it, and turns each lesson it
returned into an issue labeled `lesson`, unless code can tell the lesson is not backed:
- a kind is proposed once: a `lesson` issue of that kind, open or closed, covers it;
- a rule needs 3 evidence PRs, a check (for a one-off mistake) needs 1;
- each evidence PR must be merged into main and have a finding.
Whether CLAUDE.md or a check already covers a mistake is the agent's judgment.

Usage:
  python3 tools/lesson.py history [--limit N]
  python3 tools/lesson.py open LESSONS_JSON [--dry-run]

GITHUB_REPOSITORY_OWNER names whose `@claude` comments count. The report goes to stdout and,
when set, to GITHUB_STEP_SUMMARY. With --dry-run it shows each issue and opens nothing.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

Run = Callable[..., subprocess.CompletedProcess[str]]

REVIEWER = "claude[bot]"
LABEL = "lesson"
LIMIT = 20  # recent merged PRs the agent reads
RULE_EVIDENCE = 3  # PRs a rule needs; a check needs one
TEXTS = ("kind", "proposes", "title", "mistake", "build")
_KIND = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_KIND_MARK = re.compile(r"<!-- lesson-kind: ([a-z0-9-]+) -->")
_MENTION = re.compile(r"(?<![\w`])@([A-Za-z0-9][\w/-]*)")


@dataclass(frozen=True)
class Finding:
    source: str
    text: str


@dataclass(frozen=True)
class LessonIssue:
    number: int
    state: str
    title: str
    kind: str | None


@dataclass(frozen=True)
class Lesson:
    kind: str
    proposes: str
    title: str
    mistake: str
    build: str
    criteria: list[str]
    evidence: list[int]


def _gh(run: Run, *args: str) -> Any:
    done = run(["gh", *args], capture_output=True, text=True)
    done.check_returncode()
    return json.loads(done.stdout)


def _list(run: Run, path: str) -> list[dict[str, Any]]:
    url = f"repos/{{owner}}/{{repo}}/{path}?per_page=100"
    return [item for page in _gh(run, "api", "--paginate", "--slurp", url) for item in page]


def _by(item: dict[str, Any], login: str) -> bool:
    user: dict[str, Any] = item.get("user") or {}  # None for a deleted account
    return str(user.get("login") or "").casefold() == login.casefold()


def findings(run: Run, owner: str, pr: int) -> list[Finding]:
    """What went wrong on one PR, as told by the reviewer, the fixer and the owner only."""
    comments = _list(run, f"issues/{pr}/comments")
    notes = comments + _list(run, f"pulls/{pr}/comments") + _list(run, f"pulls/{pr}/reviews")
    messages = [c["commit"]["message"] for c in _list(run, f"pulls/{pr}/commits")]
    verdicts = [
        c["body"]
        for c in comments
        if _by(c, REVIEWER) and c["body"].startswith("Verdict: changes needed")
    ]
    corrections = [n["body"] for n in notes if _by(n, owner) and "@claude" in (n["body"] or "")]
    return (
        [Finding("Reviewer", text) for text in verdicts]
        + [Finding("Fixer, red CI", m) for m in messages if m.startswith("Fix CI:")]
        + [Finding("Owner", text) for text in corrections]
    )


def lesson_issues(run: Run) -> list[LessonIssue]:
    fields = ["--limit", "500", "--json", "number,state,title,body"]
    found = _gh(run, "issue", "list", "--label", LABEL, "--state", "all", *fields)
    return [
        LessonIssue(i["number"], i["state"].lower(), i["title"], m[1] if m else None)
        for i in found
        for m in [_KIND_MARK.search(i["body"] or "")]
    ]


def history(run: Run, owner: str, limit: int) -> str:
    lessons = lesson_issues(run)
    lines = ["# Lesson issues", ""]
    lines += [f"- #{i.number} {i.state}, kind `{i.kind}`: {i.title}" for i in lessons] or ["None."]
    fields = ["--limit", str(limit), "--json", "number,title"]
    prs = _gh(run, "pr", "list", "--state", "merged", "--base", "main", *fields)
    numbers = ", ".join(f"#{pr['number']}" for pr in prs)
    lines += ["", f"# Findings on the last {len(prs)} merged PRs ({numbers})"]
    for pr in prs:
        found = findings(run, owner, pr["number"])
        lines += ["", f"## #{pr['number']} {pr['title']}"] if found else []
        for finding in found:
            lines += ["", f"{finding.source}:", *(f"> {x}" for x in finding.text.splitlines())]
    return "\n".join(lines) + "\n"


def _quiet(text: str) -> str:
    """Put @mentions in code, so that an issue notifies nobody."""
    return _MENTION.sub(r"`@\1`", text.strip())


def _items(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _lesson(raw: object) -> Lesson:
    """Read one lesson the agent returned; ValueError says what is wrong with it."""
    item = cast(dict[str, object], raw) if isinstance(raw, dict) else {}
    texts = [item.get(t) for t in TEXTS]
    criteria, evidence = _items(item.get("criteria")), _items(item.get("evidence"))
    if not all(map(_is_text, texts)):
        raise ValueError(f"{', '.join(TEXTS)} must each be a text")
    if not criteria or not all(map(_is_text, criteria)):
        raise ValueError("criteria must be a list of texts")
    if not evidence or not all(type(pr) is int for pr in evidence):  # a bool is no PR number
        raise ValueError("evidence must be a list of PR numbers")
    kind, proposes, title, mistake, build = map(_quiet, cast(list[str], texts))
    if not _KIND.fullmatch(kind) or len(kind) > 60:
        raise ValueError("kind must be a few lowercase words joined by hyphens")
    if proposes not in ("rule", "check"):
        raise ValueError("proposes must be rule or check")
    if "\n" in title:
        raise ValueError("title must be one line")
    lines = [_quiet(c) for c in cast(list[str], criteria)]
    numbers = sorted(set(cast(list[int], evidence)))
    return Lesson(kind, proposes, title, mistake, build, lines, numbers)


def _unbacked(run: Run, owner: str, lesson: Lesson) -> str | None:
    """Why the evidence does not back the lesson, or None when it does."""
    need = RULE_EVIDENCE if lesson.proposes == "rule" else 1
    if len(lesson.evidence) < need:
        given = len(lesson.evidence)
        return f"a {lesson.proposes} needs {need} merged PRs with a finding, not {given}"
    for pr in lesson.evidence:
        command = ["gh", "pr", "view", str(pr), "--json", "state,baseRefName"]
        view = run(command, capture_output=True, text=True)
        found: dict[str, Any] = json.loads(view.stdout) if view.returncode == 0 else {}
        merged = found.get("state") == "MERGED" and found.get("baseRefName") == "main"
        if not merged or not findings(run, owner, pr):
            return f"#{pr} is not a PR merged into main with a finding"
    return None


def _body(lesson: Lesson) -> str:
    what = "a rule in CLAUDE.md" if lesson.proposes == "rule" else "a check"
    criteria = "\n".join(f"- [ ] {line}" for line in lesson.criteria)
    evidence = ", ".join(f"#{pr}" for pr in lesson.evidence)
    return (
        f"<!-- lesson-kind: {lesson.kind} -->\n"
        f"The learner agent proposes {what}. Label this issue `ready-for-agent` to accept it,"
        " or close it: either way, this kind is not proposed again.\n\n"
        f"## What to build\n\n{lesson.build}\n\n"
        f"## Acceptance criteria\n\n{criteria}\n\n"
        f"## Evidence\n\n{lesson.mistake} Seen on {evidence}.\n"
    )


def open_lessons(run: Run, owner: str, path: Path, dry_run: bool) -> list[str]:
    """Open an issue for each lesson code finds backed; return the report."""
    returned = json.loads(path.read_text())
    lessons = (
        cast(dict[str, object], returned).get("lessons") if isinstance(returned, dict) else None
    )
    if not isinstance(lessons, list):
        raise ValueError(f"{path} has no list of lessons")
    taken = {i.kind: f"lesson #{i.number}" for i in lesson_issues(run) if i.kind}
    report: list[str] = []
    for raw in cast(list[object], lessons):
        try:
            lesson = _lesson(raw)
        except ValueError as error:
            report.append(f"- Refused a lesson: {error}.")
            continue
        if lesson.kind in taken:
            reason = f"{taken[lesson.kind]} covers it"
        else:
            reason = _unbacked(run, owner, lesson)
        if reason:
            report.append(f"- Refused `{lesson.kind}`: {reason}.")
            continue
        body = _body(lesson)
        if dry_run:
            report += [f"## Would open: {lesson.title}", "", f"Label `{LABEL}`.", "", body]
        else:
            command = ["gh", "issue", "create", "--title", lesson.title, "--label", LABEL]
            done = run([*command, "--body-file", "-"], input=body, capture_output=True, text=True)
            done.check_returncode()
            report.append(f"- Opened {done.stdout.strip()}: {lesson.title}")
        taken[lesson.kind] = "a lesson of this run"
    return report or ["No lesson: nothing recurring or obvious that a rule or a check misses."]


def main(
    argv: Sequence[str] | None = None,
    run: Run = subprocess.run,
    env: Mapping[str, str] = os.environ,
) -> int:
    parser = argparse.ArgumentParser(description="The learner agent's history and lessons.")
    commands = parser.add_subparsers(dest="command", required=True)
    reader = commands.add_parser("history", help="what went wrong on recent merged PRs")
    reader.add_argument("--limit", type=int, default=LIMIT)
    opener = commands.add_parser("open", help="open the lessons that code finds backed")
    opener.add_argument("lessons", type=Path)
    opener.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    owner = env.get("GITHUB_REPOSITORY_OWNER", "")
    if not owner:
        parser.error("set GITHUB_REPOSITORY_OWNER: only the owner's @claude comments count")

    if args.command == "history":
        print(history(run, owner, args.limit), end="")
        return 0
    heading = "Learner, dry run: what it would open" if args.dry_run else "Learner: lessons"
    report = "\n".join([f"# {heading}", "", *open_lessons(run, owner, args.lessons, args.dry_run)])
    print(report)
    if summary := env.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as file:
            file.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
