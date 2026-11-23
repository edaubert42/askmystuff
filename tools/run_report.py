"""Say what an agent was denied and how it ended, from claude-code-action's execution file.

The action hides the transcript: `show_full_output` would put every tool output in this public
repository's logs. Its execution file holds every message but dies with the runner. This prints
only the denied tool calls and the agent's last message, cut short, in code blocks so nothing the
agent wrote renders or mentions anyone. It never fails: a missing file is one line.

Usage: python3 tools/run_report.py EXECUTION_FILE
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

INPUT_CHARS = 200
MESSAGE_CHARS = 1500
FENCE = "~~~"


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _plain(text: str) -> str:
    return text.replace(FENCE, "~ ~ ~")  # a fence inside would close the block


def _denial(denial: Any) -> str:
    entry = cast(dict[str, Any], denial) if isinstance(denial, dict) else {}
    raw = entry.get("tool_input")
    tool_input = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
    target = json.dumps(tool_input, ensure_ascii=False)
    for key in ("command", "file_path", "path", "url", "pattern"):
        value = tool_input.get(key)
        if isinstance(value, str):
            target = value
            break
    return _plain(f"{entry.get('tool_name')}: {_cut(' '.join(target.split()), INPUT_CHARS)}")


def _result(path: str) -> dict[str, Any] | None:
    try:
        messages = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(messages, list):
        return None
    results = [
        cast(dict[str, Any], m)
        for m in cast(list[Any], messages)
        if isinstance(m, dict) and cast(dict[str, Any], m).get("type") == "result"
    ]
    return results[-1] if results else None


def report(path: str) -> str:
    result = _result(path) if path else None
    if result is None:
        return "No execution file: nothing to say about denials or the last message."
    raw = result.get("permission_denials")
    denials = [_denial(d) for d in cast(list[Any], raw)] if isinstance(raw, list) else []
    lines = [f"Denied ({len(denials)}):", FENCE, *denials, FENCE] if denials else ["Denied: none."]
    message = str(result.get("result") or "").strip()
    lines += ["", "Last message:", FENCE, _plain(_cut(message, MESSAGE_CHARS)) or "(none)", FENCE]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Say what an agent was denied, and its last words."
    )
    parser.add_argument("execution_file")
    print(report(parser.parse_args(argv).execution_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
