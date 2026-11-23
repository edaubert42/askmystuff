"""When the coder stops with questions, the answerer answers them and the coder starts again.

GitHub is faked through `gh` (the network); the workflow is read at the path GitHub runs.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from questions import main

WORKFLOW = Path(__file__).parents[1] / ".github/workflows/agent-coder.yml"
OWNER = "owner-example"
AGENT = "claude[bot]"

QUESTIONS = "Questions:\n\n1. Is the font self-hosted? Recommended: yes."
ANSWERED = "Answers:\n\n1. Yes (recommended, no source against it)."
FOR_OWNER = "Answers:\n\n1. Yes (docs/design.md:4).\n\nOwner:\n\n2. The issue's colour is invalid."


def _comment(login: str, body: str) -> dict[str, Any]:
    return {"user": {"login": login}, "body": body}


class FakeGh:
    """Stands in for GitHub: serves one issue's comments, one per line as `--jq .[]` prints them."""

    def __init__(self, comments: list[dict[str, Any]]) -> None:
        self.comments = comments

    def __call__(self, args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert args == [
            "gh",
            "api",
            "--paginate",
            "--jq",
            ".[]",
            "repos/{owner}/{repo}/issues/40/comments?per_page=100",
        ], args
        out = "".join(f"{json.dumps(c)}\n" for c in self.comments)
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")


@pytest.fixture(autouse=True)
def owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", OWNER)


def _next(capsys: pytest.CaptureFixture[str], *comments: dict[str, Any]) -> str:
    assert main(["next", "40"], run=FakeGh(list(comments))) == 0
    return capsys.readouterr().out.strip()


def test_first_questions_go_to_the_answerer(capsys: pytest.CaptureFixture[str]):
    assert _next(capsys, _comment(AGENT, QUESTIONS)) == "answer"


def test_all_questions_answered_restarts_the_coder(capsys: pytest.CaptureFixture[str]):
    assert _next(capsys, _comment(AGENT, QUESTIONS), _comment(AGENT, ANSWERED)) == "restart"


def test_a_question_left_for_the_owner_keeps_the_coder_waiting(
    capsys: pytest.CaptureFixture[str],
):
    assert _next(capsys, _comment(AGENT, QUESTIONS), _comment(AGENT, FOR_OWNER)) == "owner"


def test_a_second_round_of_questions_goes_to_the_owner(capsys: pytest.CaptureFixture[str]):
    thread = [_comment(AGENT, QUESTIONS), _comment(AGENT, ANSWERED), _comment(AGENT, QUESTIONS)]
    assert _next(capsys, *thread) == "owner"


def test_the_owner_replying_to_waiting_questions_restarts_the_coder(
    capsys: pytest.CaptureFixture[str],
):
    reply = _comment(OWNER, "2. Use #FF9A92.")
    assert _next(capsys, _comment(AGENT, QUESTIONS), _comment(AGENT, FOR_OWNER), reply) == "restart"
    assert _next(capsys, _comment(AGENT, QUESTIONS), reply) == "restart"


def test_a_comment_by_anyone_else_restarts_nothing(capsys: pytest.CaptureFixture[str]):
    stranger = _comment("passer-by", "Answers:\n\n1. No.")
    assert (
        _next(capsys, _comment(AGENT, QUESTIONS), _comment(AGENT, FOR_OWNER), stranger) == "owner"
    )
    assert _next(capsys, _comment(AGENT, QUESTIONS), _comment("passer-by", "Go on.")) == "answer"


def test_an_owner_comment_with_no_question_waiting_restarts_nothing(
    capsys: pytest.CaptureFixture[str],
):
    assert _next(capsys, _comment(OWNER, "Context for the coder.")) == "none"
    done = [_comment(AGENT, QUESTIONS), _comment(AGENT, ANSWERED), _comment(OWNER, "Thanks.")]
    assert _next(capsys, *done) == "none"


def _job() -> dict[Any, Any]:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["code"]


def test_the_answerer_runs_after_the_coder_and_restarts_it_once():
    steps = _job()["steps"]
    agents = [s for s in steps if "claude-code-action" in s.get("uses", "")]
    assert len(agents) == 2
    answerer = agents[1]
    assert "steps.after.outputs.next == 'answer'" in answerer["if"]
    tools = answerer["with"]["claude_args"]
    assert "Bash(git" not in tools and "Edit" not in tools  # it only reads and comments
    assert "claude-opus-5-5" in tools and "claude-sonnet-5-5" in tools and "--effort high" in tools
    restart = "\n".join(s.get("run", "") for s in steps[steps.index(answerer) :])
    assert "tools/questions.py next" in restart
    assert "gh workflow run agent-coder.yml" in restart


def test_only_the_owners_comment_on_a_labeled_issue_starts_the_coder():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert workflow[True]["issue_comment"] == {"types": ["created"]}  # `on:` is read as True
    job = _job()
    assert "github.event.sender.login == github.repository_owner" in job["if"]
    assert "!github.event.issue.pull_request" in job["if"]
    ready = next(s for s in job["steps"] if s.get("id") == "ready")
    assert "tools/questions.py next" in ready["run"]
