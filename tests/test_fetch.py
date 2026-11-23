import json
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pymupdf
import pytest
import yaml

from askmystuff.fetch import DEFAULT_MANUALS, ManualsError, load_manuals, main

ROOT = Path(__file__).parents[1]

Run = Callable[..., tuple[int, Path]]


def make_pdf(pages: int, text: str = "Notice") -> bytes:
    doc = pymupdf.open()  # pyright: ignore[reportUnknownMemberType]
    for number in range(1, pages + 1):
        page = doc.new_page()  # pyright: ignore[reportUnknownMemberType]
        page.insert_text((72, 72), f"{text} page {number}")  # pyright: ignore[reportUnknownMemberType]
    return doc.tobytes()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


class Site:
    """A local HTTP server: `routes` maps a path to (content type, body)."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[str, bytes]] = {}
        self.hits: list[tuple[str, str]] = []  # (path, User-Agent)
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                site.hits.append((self.path, self.headers.get("User-Agent", "")))
                route = site.routes.get(self.path)
                if route is None:
                    self.send_error(404)
                    return
                content_type, body = route
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_port}{path}"

    def hit_count(self, path: str) -> int:
        return sum(1 for hit, _ in self.hits if hit == path)


@pytest.fixture
def site() -> Iterator[Site]:
    site = Site()
    yield site
    site.server.shutdown()
    site.server.server_close()


def entry(slug: str, url: str) -> dict[str, str]:
    return {
        "slug": slug,
        "kind": "appliance",
        "brand": "Acme",
        "model": slug.upper(),
        "label": "Phone",
        "manual_url": url,
    }


def write_manuals(path: Path, entries: list[dict[str, str]]) -> Path:
    path.write_text(yaml.safe_dump(entries))  # pyright: ignore[reportUnknownMemberType]
    return path


@pytest.fixture
def run(tmp_path: Path) -> Run:
    """Run the script on a manuals file; return the exit code and the output folder."""
    out = tmp_path / "pdf"

    def _run(entries: list[dict[str, str]], *flags: str) -> tuple[int, Path]:
        manuals = write_manuals(tmp_path / "manuals.yaml", entries)
        return main(["--manuals", str(manuals), "--out", str(out), *flags]), out

    return _run


# Criterion 1: the schema


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"kind": "book"}, "kind"),
        ({"manual_url": "ftp://x/a.pdf"}, "manual_url"),
        ({"brand": ""}, "brand"),
        ({"model": None}, "model"),
    ],
)
def test_a_bad_entry_names_its_slug_and_field(tmp_path: Path, change: dict[str, str], field: str):
    bad = {**entry("broken-one", "http://x/a.pdf"), **change}
    manuals = write_manuals(tmp_path / "m.yaml", [entry("fine-one", "http://x/f.pdf"), bad])
    with pytest.raises(ManualsError) as error:
        load_manuals(manuals)
    assert "broken-one" in str(error.value)
    assert field in str(error.value)


def test_a_missing_field_names_its_slug(tmp_path: Path):
    bad = entry("no-label", "http://x/a.pdf")
    del bad["label"]
    with pytest.raises(ManualsError, match=r"no-label.*label"):
        load_manuals(write_manuals(tmp_path / "m.yaml", [bad]))


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([entry("twin", "http://x/a"), entry("twin", "http://x/b")], "twin"),
        ([entry("../escape", "http://x/a")], "slug"),
    ],
)
def test_a_duplicate_or_unsafe_slug_is_refused(
    tmp_path: Path, entries: list[dict[str, str]], message: str
):
    with pytest.raises(ManualsError, match=message):
        load_manuals(write_manuals(tmp_path / "m.yaml", entries))


# Criterion 2: download and manifest


def test_each_manual_is_saved_with_a_manifest(site: Site, run: Run):
    first, second = make_pdf(3), make_pdf(5, "Autre")
    site.routes = {"/a.pdf": ("application/pdf", first), "/b.pdf": ("application/pdf", second)}
    code, out = run([entry("a-one", site.url("/a.pdf")), entry("b-two", site.url("/b.pdf"))])

    assert code == 0
    assert (out / "a-one.pdf").read_bytes() == first
    assert (out / "b-two.pdf").read_bytes() == second
    manifest = json.loads((out / "manifest.json").read_text())
    assert sorted(manifest) == ["a-one", "b-two"]
    assert manifest["a-one"]["sha256"] == sha256(first).hexdigest()
    assert manifest["a-one"]["size"] == len(first)
    assert manifest["a-one"]["pages"] == 3
    assert manifest["b-two"]["pages"] == 5
    assert "Mozilla/5.0" in site.hits[0][1]  # hosts refuse requests without a browser-like agent


def test_the_module_runs_with_python_dash_m(site: Site, tmp_path: Path):
    site.routes = {"/a.pdf": ("application/pdf", make_pdf(2))}
    manuals = write_manuals(tmp_path / "m.yaml", [entry("a-one", site.url("/a.pdf"))])
    out = tmp_path / "o"
    command = [sys.executable, "-m", "askmystuff.fetch", "--manuals", str(manuals)]
    done = subprocess.run([*command, "--out", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert (out / "a-one.pdf").exists()


# Criterion 3: not downloaded again unless --force


def test_a_known_manual_is_downloaded_again_only_with_force(site: Site, run: Run):
    site.routes = {"/a.pdf": ("application/pdf", make_pdf(2))}
    entries = [entry("a-one", site.url("/a.pdf"))]
    run(entries)
    code, out = run(entries)
    assert code == 0
    assert site.hit_count("/a.pdf") == 1
    assert json.loads((out / "manifest.json").read_text())["a-one"]["pages"] == 2

    site.routes["/a.pdf"] = ("application/pdf", make_pdf(4))
    assert run(entries, "--force")[0] == 0
    assert site.hit_count("/a.pdf") == 2
    assert json.loads((out / "manifest.json").read_text())["a-one"]["pages"] == 4


# Criterion 4: a bad URL fails alone


def test_a_url_that_is_not_a_pdf_fails_alone(
    site: Site, run: Run, capsys: pytest.CaptureFixture[str]
):
    good = make_pdf(2)
    site.routes = {
        "/page.html": ("text/html", b"<html>Accept cookies</html>"),
        "/good.pdf": ("application/pdf", good),
    }
    entries = [  # a page that is not a PDF, a missing page, a good manual
        entry("is-html", site.url("/page.html")),
        entry("is-gone", site.url("/nope.pdf")),
        entry("is-good", site.url("/good.pdf")),
    ]
    code, out = run(entries)
    assert code == 1
    assert sorted(path.name for path in out.iterdir()) == ["is-good.pdf", "manifest.json"]
    assert (out / "is-good.pdf").read_bytes() == good
    assert sorted(json.loads((out / "manifest.json").read_text())) == ["is-good"]
    error = capsys.readouterr().err
    assert "is-html" in error
    assert "is-gone" in error


def test_a_pdf_served_with_the_wrong_content_type_is_accepted(site: Site, run: Run):
    site.routes = {"/a.pdf": ("application/octet-stream", make_pdf(1))}
    code, out = run([entry("a-one", site.url("/a.pdf"))])
    assert code == 0
    assert (out / "a-one.pdf").exists()


def test_a_bad_manuals_file_stops_before_any_download(site: Site, tmp_path: Path):
    site.routes = {"/a.pdf": ("application/pdf", make_pdf(1))}
    bad = {**entry("broken-one", site.url("/a.pdf")), "kind": "book"}
    manuals = write_manuals(tmp_path / "m.yaml", [entry("fine-one", site.url("/a.pdf")), bad])
    assert main(["--manuals", str(manuals), "--out", str(tmp_path / "pdf")]) == 2
    assert site.hits == []


# Criterion 6: the committed file


def test_the_committed_manuals_file_has_the_eight_devices():
    manuals = load_manuals(DEFAULT_MANUALS)
    assert [m.slug for m in manuals] == [
        "samsung-galaxy-s25",
        "thinkpad-p1-gen7",
        "philips-tv-8501",
        "philips-soundbar-tab5201",
        "bose-soundlink-micro",
        "bose-qc-ultra-headphones",
        "epson-ecotank-et-2830",
        "dacia-duster",
    ]
    assert all(m.manual_url.startswith("https://") for m in manuals)


# Criterion 7: the design names the devices


def test_the_design_devices_section_names_the_devices():
    design = (ROOT / "docs" / "DESIGN.md").read_text()
    section = design.split("## Devices", 1)[1].split("\n## ", 1)[0]
    for manual in load_manuals(DEFAULT_MANUALS):
        assert manual.brand in section
        assert manual.label.lower() in section.lower()
    assert "soundbar" in section.lower()
    assert "Chromecast" not in section
