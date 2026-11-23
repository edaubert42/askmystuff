"""An issue labeled `ready-for-agent` waits for the issues it is blocked by, then starts alone.

GitHub is faked through `gh` (the network); the workflows are read at the paths GitHub runs.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from blockers import blocked_by, main

WORKFLOWS = Path(__file__).parents[1] / ".github/workflows"
OWNER = "owner-example"
READY = "ready-for-agent"


def _body(blocked: str) -> str:
    return (
        f"## What to build\n\nAfter #5.\n\n## Blocked by\n\n{blocked}\n\n## Out of scope\n\n- #99\n"
    )


def _labeled(login: str, name: str = READY) -> dict[str, Any]:
    return {"event": "labeled", "actor": {"login": login}, "label": {"name": name}}


def _unlabeled(login: str) -> dict[str, Any]:
    return {"event": "unlabeled", "actor": {"login": login}, "label": {"name": READY}}


def _issue(
    state: str, blocked: str, events: list[dict[str, Any]], reason: str = "completed"
) -> dict[str, Any]:
    state_reason = reason if state == "closed" else None
    return {"state": state, "state_reason": state_reason, "body": _body(blocked), "events": events}


# A fictional backlog: 29 is the database, the others wait for it.
ISSUES: dict[int, dict[str, Any]] = {
    29: _issue("closed", "None.", [_labeled(OWNER)]),
    30: _issue("open", "#29", [_labeled(OWNER)]),
    31: _issue("open", "#29 (the tables)", [_labeled(OWNER, "hard"), _labeled(OWNER)]),
    32: _issue("open", "#29", [_labeled("passer-by")]),
    33: _issue("open", "#29", []),
    34: _issue("closed", "#29", [_labeled(OWNER)]),
    35: _issue("open", "#29, #36", [_labeled(OWNER)]),
    36: _issue("open", "None.", [_labeled(OWNER)]),
    37: _issue("open", "#29", [_labeled(OWNER), _unlabeled(OWNER)]),
    38: _issue("closed", "None.", [_labeled(OWNER)], reason="not_planned"),
    39: _issue("open", "#38", [_labeled(OWNER)]),
    40: _issue("open", "#29", [_labeled(OWNER)]),
}
BRANCHES = ["agent/issue-40-backup", "agent/issue-300-other"]  # the coder already built 40


class FakeGh:
    """Stands in for GitHub: serves the fictional backlog."""

    def __init__(self, issues: dict[int, dict[str, Any]]) -> None:
        self.issues = issues

    def __call__(self, args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        served = self._serve(args)
        # `--jq .[]` prints one item per line
        out = (
            "".join(f"{json.dumps(item)}\n" for item in served)
            if "--jq" in args
            else json.dumps(served)
        )
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    def _labels(self, issue: dict[str, Any]) -> list[str]:
        labels: set[str] = set()
        for event in issue["events"]:
            name = event["label"]["name"]
            (labels.add if event["event"] == "labeled" else labels.discard)(name)
        return sorted(labels)

    def _serve(self, args: list[str]) -> Any:
        match args:
            case ["gh", "api", "--paginate", "--jq", ".[]", path]:
                found = re.fullmatch(
                    r"repos/\{owner\}/\{repo\}/issues/(\d+)/events\?per_page=100", path
                )
                assert found, path
                return self.issues[int(found.group(1))]["events"]
            case ["gh", "api", path] if "/git/matching-refs/heads/" in path:
                prefix = path.split("/git/matching-refs/heads/")[1]
                return [{"ref": f"refs/heads/{b}"} for b in BRANCHES if b.startswith(prefix)]
            case ["gh", "api", path]:
                found = re.fullmatch(r"repos/\{owner\}/\{repo\}/issues/(\d+)", path)
                assert found, path
                issue = self.issues[int(found.group(1))]
                labels = [{"name": name} for name in self._labels(issue)]
                return {
                    **{k: issue[k] for k in ("state", "state_reason", "body")},
                    "labels": labels,
                }
            case ["gh", "issue", "list", "--label", label, "--state", "open", *_]:
                return [
                    {"number": n, "body": issue["body"]}
                    for n, issue in self.issues.items()
                    if issue["state"] == "open" and label in self._labels(issue)
                ]
            case _:
                raise AssertionError(args)


@pytest.fixture(autouse=True)
def owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", OWNER)


def _run(capsys: pytest.CaptureFixture[str], issues: dict[int, dict[str, Any]], *argv: str):
    code = main(list(argv), run=FakeGh(issues))
    return code, capsys.readouterr().out


def test_blocked_by_reads_only_the_blocked_by_section():
    assert blocked_by(_body("None.")) == []
    assert blocked_by(_body("#29 (the tables), #30")) == [29, 30]
    assert blocked_by("## What to build\n\nAfter #29.\n") == []


def test_an_issue_waits_while_one_of_its_blockers_is_open(capsys: pytest.CaptureFixture[str]):
    assert _run(capsys, ISSUES, "check", "35") == (0, "waiting for #36\n")


def test_an_issue_starts_once_its_blockers_are_closed(capsys: pytest.CaptureFixture[str]):
    assert _run(capsys, ISSUES, "check", "30") == (0, "ready\n")
    assert _run(capsys, ISSUES, "check", "36") == (0, "ready\n")


def test_a_blocker_closed_as_not_planned_keeps_blocking(capsys: pytest.CaptureFixture[str]):
    assert _run(capsys, ISSUES, "check", "39") == (0, "waiting for #38\n")
    assert _run(capsys, ISSUES, "unblocked", "38") == (0, "")


def test_an_issue_the_coder_already_started_does_not_start_again(
    capsys: pytest.CaptureFixture[str],
):
    assert _run(capsys, ISSUES, "check", "40") == (0, "already started\n")
    assert _run(capsys, ISSUES, "check", "30") == (0, "ready\n")  # agent/issue-300-* is not 30's


@pytest.mark.parametrize("number", ["32", "33", "37"])
def test_an_issue_not_labeled_by_the_owner_does_not_start(
    capsys: pytest.CaptureFixture[str], number: str
):
    assert _run(capsys, ISSUES, "check", number) == (0, f"not labeled {READY} by the owner\n")


def test_closing_a_blocker_lists_the_issues_it_was_the_last_blocker_of(
    capsys: pytest.CaptureFixture[str],
):
    # Not 32 (labeled by someone else), 33 (no label), 34 (closed), 35 (#36 open),
    # 37 (unlabeled), 40 (already started).
    assert _run(capsys, ISSUES, "unblocked", "29") == (0, "30\n31\n")


def test_closing_an_issue_still_open_elsewhere_starts_nothing(capsys: pytest.CaptureFixture[str]):
    reopened = {**ISSUES, 29: {**ISSUES[29], "state": "open"}}
    assert _run(capsys, reopened, "unblocked", "29") == (0, "")


def _workflow(name: str) -> dict[Any, Any]:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def test_the_coder_runs_the_agent_only_when_the_check_says_ready():
    workflow = _workflow("agent-coder.yml")
    assert "workflow_dispatch" in workflow[True]  # `on:` is read as True by YAML 1.1
    job = workflow["jobs"]["code"]
    # A dispatch runs only main's code, where the owner-label check cannot be edited away.
    assert (
        "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'" in job["if"]
    )
    steps = job["steps"]
    check = next(step for step in steps if step.get("id") == "ready")
    assert "tools/blockers.py check" in check["run"]
    agent = next(step for step in steps if "claude-code-action" in step.get("uses", ""))
    assert agent["if"] == "steps.ready.outputs.go == 'true'"


def test_closing_an_issue_starts_the_coder_on_each_unblocked_issue():
    workflow = _workflow("agent-unblock.yml")
    assert workflow[True] == {"issues": {"types": ["closed"]}}
    job = workflow["jobs"]["unblock"]
    assert "state_reason == 'completed'" in job["if"]
    run = "\n".join(step.get("run", "") for step in job["steps"])
    assert "tools/blockers.py unblocked" in run
    assert "gh workflow run agent-coder.yml" in run


def test_a_coder_started_by_a_workflow_is_not_refused_as_a_bot():
    # agent-unblock.yml and the answerer dispatch the coder with GITHUB_TOKEN.
    job = _workflow("agent-coder.yml")["jobs"]["code"]
    agents = [s for s in job["steps"] if "claude-code-action" in s.get("uses", "")]
    assert agents and all(s["with"]["allowed_bots"] == "github-actions[bot]" for s in agents)


def test_a_skipped_comment_run_cannot_cancel_a_queued_coder():
    workflow = _workflow("agent-coder.yml")
    assert "concurrency" not in workflow
    assert workflow["jobs"]["code"]["concurrency"]["cancel-in-progress"] is False
