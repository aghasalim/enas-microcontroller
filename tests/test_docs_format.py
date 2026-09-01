"""Rendering checks for the markdown, not content checks.

Every failure mode here is one that has actually shipped in this portfolio: a
table whose cells lost the space after the pipe and collapsed into prose, two
code spans that fused because the backticks ran together, and non-ASCII
punctuation that arrived from an editor rather than from the keyboard.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".venv", ".pytest_cache", "build", "node_modules"}
# rglob otherwise picks up README.md files that tools drop in their own caches,
# which are not ours to format and inflate the local count over CI's.
DOCS = sorted(p for p in ROOT.rglob("*.md") if not SKIP & set(p.parts))


def prose_lines(text: str):
    """Yield (number, line) outside fenced blocks, indented code and front matter."""
    lines = text.splitlines()
    fenced = False
    start = 0
    if lines and lines[0].strip() == "---":                 # YAML front matter
        end = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), 0)
        start = end + 1
    for i, line in enumerate(lines[start:], start + 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or line.startswith("    "):
            continue
        yield i, line


def outside_code(line: str):
    """Yield characters that are not inside a backtick span."""
    inside = False
    for ch in line:
        if ch == "`":
            inside = not inside
            continue
        if not inside:
            yield ch


def cells(line: str) -> list[str]:
    """Interior cells of a table row, before any stripping."""
    return line.strip().strip("|").split("|") if line.strip().startswith("|") else []


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_table_cells_keep_their_padding(doc: Path):
    """A cell written `|x |` instead of `| x |` collapses the table into prose.

    Checked on every cell rather than only on cells that begin with a backtick,
    because the pipe does not care what character follows it.
    """
    bad = []
    for i, line in prose_lines(doc.read_text()):
        for c in cells(line):
            if c and not c.startswith(" ") and set(c.strip()) - set("-: "):
                bad.append((i, c[:24]))
    assert not bad, f"{doc.name}: cell lost the space after its pipe at {bad}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_code_spans_are_not_fused(doc: Path):
    bad = [i for i, line in prose_lines(doc.read_text()) if "``" in line
           and not line.lstrip().startswith("```")]
    assert not bad, f"{doc.name}: fused code spans on lines {bad}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_punctuation_is_plain_ascii(doc: Path):
    banned = "—–‘’“”"
    bad = [(i, ch) for i, line in prose_lines(doc.read_text())
           for ch in outside_code(line) if ch in banned or ord(ch) > 0x2500]
    assert not bad, f"{doc.name}: non-ASCII punctuation {bad}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_tables_have_a_separator_row(doc: Path):
    """A header with no `| --- |` under it renders as one run-on line."""
    lines = list(prose_lines(doc.read_text()))
    for n, (i, line) in enumerate(lines):
        if line.lstrip().startswith("|") and n + 1 < len(lines):
            prev = lines[n - 1][1] if n else ""
            nxt = lines[n + 1][1]
            if not prev.lstrip().startswith("|"):           # this is a header row
                assert set(nxt.strip()) <= set("|-: "), \
                    f"{doc.name}: table at line {i} has no separator row"
