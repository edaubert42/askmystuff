import subprocess
from pathlib import Path

import pytest
import yaml
from pr_size import LIMIT, main

CI = Path(__file__).parents[1] / ".github/workflows/ci.yml"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git repository with one commit on main and a branch `work` checked out."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "ada@example.org")
    _git(tmp_path, "config", "user.name", "Ada Martin")
    (tmp_path / "README.md").write_text("askmystuff\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "start")
    _git(tmp_path, "checkout", "-q", "-b", "work")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _commit(repo: Path, files: dict[str, int]) -> None:
    for name, lines in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"line {i}\n" for i in range(lines)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "work")


def test_400_changed_lines_pass_and_600_fail(repo: Path):
    _commit(repo, {"src/a.py": 400})
    assert main(["main"]) == 0
    _commit(repo, {"src/b.py": 200})
    assert main(["main"]) == 1


def test_lock_files_are_not_counted(repo: Path):
    _commit(repo, {"src/a.py": 400, "uv.lock": 900, "frontend/package-lock.json": 900})
    assert main(["main"]) == 0


def test_failure_prints_the_changed_lines_and_the_limit(
    repo: Path, capsys: pytest.CaptureFixture[str]
):
    _commit(repo, {"src/a.py": 600})
    assert main(["main"]) == 1
    out = capsys.readouterr().out
    assert "600 changed lines" in out
    assert f"limit {LIMIT}" in out


def test_ci_runs_the_check_on_pull_requests_unless_labeled_big_ok():
    workflow = yaml.safe_load(CI.read_text())
    job = workflow["jobs"]["size"]
    assert "pull_request" in job["if"]
    run = "\n".join(step.get("run", "") for step in job["steps"])
    assert "tools/pr_size.py" in run
    assert "big-ok" in run
