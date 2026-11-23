import json
import subprocess
from pathlib import Path

import pytest
from edit_pr import main


class FakeGh:
    """Stands in for the network: records the gh call instead of sending it."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(
        self, args: list[str], *, input: str, **_: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, json.loads(input)))
        return subprocess.CompletedProcess(args, 0)


def test_title_only_patches_the_title_of_that_pr():
    gh = FakeGh()
    assert main(["8", "--title", "Text: a job title becomes a slug"], run=gh) == 0
    [(args, payload)] = gh.calls
    assert args[:4] == ["gh", "api", "-X", "PATCH"]
    assert args[4] == "repos/{owner}/{repo}/pulls/8"
    assert payload == {"title": "Text: a job title becomes a slug"}


def test_body_file_patches_the_body_with_the_file_content(tmp_path: Path):
    body = tmp_path / "body.md"
    body.write_text("Closes #7\n\n## What\n")
    gh = FakeGh()
    assert main(["8", "--body-file", str(body)], run=gh) == 0
    assert gh.calls[0][1] == {"body": "Closes #7\n\n## What\n"}


def test_no_title_and_no_body_changes_nothing():
    gh = FakeGh()
    with pytest.raises(SystemExit) as exit_:
        main(["8"], run=gh)
    assert exit_.value.code != 0
    assert gh.calls == []


def test_any_other_field_is_refused():
    gh = FakeGh()
    with pytest.raises(SystemExit) as exit_:
        main(["8", "--title", "x", "--state", "closed"], run=gh)
    assert exit_.value.code != 0
    assert gh.calls == []


def test_pr_must_be_a_number():
    gh = FakeGh()
    with pytest.raises(SystemExit):
        main(["8/merge"], run=gh)
    assert gh.calls == []
