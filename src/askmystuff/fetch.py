"""Download the manuals listed in `data/manuals.yaml` into `data/pdf/`.

Usage: uv run python -m askmystuff.fetch [--force]

Writes `data/pdf/<slug>.pdf` and `data/pdf/manifest.json` (sha256, size and
page count per slug). A manual whose file still matches the manifest's sha256
is not downloaded again, unless `--force`. One failing manual does not stop
the others; the exit code is then 1 (2 when `manuals.yaml` itself is invalid).
"""

import argparse
import json
import re
import sys
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pymupdf
import yaml

ROOT = Path(__file__).parents[2]
DEFAULT_MANUALS = ROOT / "data" / "manuals.yaml"
DEFAULT_OUT = ROOT / "data" / "pdf"

# Some hosts refuse requests that do not look like a browser.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)
TIMEOUT_SECONDS = 120
FIELDS = ("slug", "kind", "brand", "model", "label", "manual_url")
SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


class ManualsError(ValueError):
    """`manuals.yaml` is invalid; the message names the slug of the bad entry."""


class FetchError(Exception):
    """One manual could not be downloaded."""


@dataclass(frozen=True)
class Manual:
    slug: str
    kind: str
    brand: str
    model: str
    label: str
    manual_url: str


def load_manuals(path: Path) -> list[Manual]:
    """Read and validate the manuals file."""
    data: Any = yaml.safe_load(path.read_text())
    if not isinstance(data, list):
        raise ManualsError(f"{path}: expected a list of manuals")
    manuals: list[Manual] = []
    for position, raw in enumerate(cast(list[Any], data), start=1):
        manual = _parse(raw, position)
        if any(m.slug == manual.slug for m in manuals):
            raise ManualsError(f"{manual.slug}: slug used twice")
        manuals.append(manual)
    return manuals


def _parse(raw: Any, position: int) -> Manual:
    if not isinstance(raw, dict):
        raise ManualsError(f"entry {position}: expected a mapping")
    entry = cast(dict[str, Any], raw)
    slug = entry.get("slug")
    if not isinstance(slug, str) or not SLUG.fullmatch(slug):
        raise ManualsError(f"entry {position}: slug must be lowercase words joined by '-'")
    for field in FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ManualsError(f"{slug}: {field} must be a non-empty string")
    unknown = sorted(set(entry) - set(FIELDS))
    if unknown:
        raise ManualsError(f"{slug}: unknown field {', '.join(unknown)}")
    if entry["kind"] != "appliance":
        raise ManualsError(f"{slug}: kind must be 'appliance', got {entry['kind']!r}")
    if not entry["manual_url"].startswith(("http://", "https://")):
        raise ManualsError(f"{slug}: manual_url must be an http(s) link")
    return Manual(**{field: entry[field] for field in FIELDS})


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "unknown")
            body: bytes = response.read()
    except OSError as error:  # URLError, HTTPError and timeouts
        raise FetchError(f"download failed: {error}") from error
    if not body.startswith(b"%PDF"):
        raise FetchError(f"not a PDF (content type {content_type}, no %PDF header)")
    return body


def _page_count(body: bytes) -> int:
    try:
        with pymupdf.open(stream=body, filetype="pdf") as doc:  # pyright: ignore[reportUnknownMemberType]
            return int(doc.page_count)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as error:
        raise FetchError(f"unreadable PDF: {error}") from error


def _describe(body: bytes) -> dict[str, Any]:
    return {"sha256": sha256(body).hexdigest(), "size": len(body), "pages": _page_count(body)}


def _is_current(target: Path, known: dict[str, Any] | None) -> bool:
    """The file is on disk and still has the sha256 the manifest recorded."""
    if known is None or not target.is_file():
        return False
    return sha256(target.read_bytes()).hexdigest() == known.get("sha256")


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data: Any = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return cast(dict[str, dict[str, Any]], data) if isinstance(data, dict) else {}


def fetch(manuals: Sequence[Manual], out: Path, force: bool = False) -> dict[str, str]:
    """Download the manuals; return the error message of each slug that failed."""
    out.mkdir(parents=True, exist_ok=True)
    old = _read_manifest(out / "manifest.json")
    manifest: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for manual in manuals:
        target = out / f"{manual.slug}.pdf"
        known = old.get(manual.slug)
        if not force and _is_current(target, known) and known is not None:
            print(f"{manual.slug}: already downloaded")
            manifest[manual.slug] = known
            continue
        try:
            body = _download(manual.manual_url)
            manifest[manual.slug] = _describe(body)
        except FetchError as error:
            errors[manual.slug] = str(error)
            print(f"{manual.slug}: {error} ({manual.manual_url})", file=sys.stderr)
            continue
        partial = target.with_suffix(".pdf.part")
        partial.write_bytes(body)
        partial.replace(target)
        print(f"{manual.slug}: downloaded, {manifest[manual.slug]['pages']} pages")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the manuals of data/manuals.yaml.")
    parser.add_argument("--manuals", type=Path, default=DEFAULT_MANUALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="download again every manual")
    args = parser.parse_args(argv)
    try:
        manuals = load_manuals(args.manuals)
    except ManualsError as error:
        print(f"{args.manuals}: {error}", file=sys.stderr)
        return 2
    errors = fetch(manuals, args.out, force=args.force)
    if errors:
        print(
            f"{len(errors)} of {len(manuals)} manuals failed: {', '.join(errors)}", file=sys.stderr
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
