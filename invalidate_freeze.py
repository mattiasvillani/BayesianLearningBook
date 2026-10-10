#!/usr/bin/env python3
"""Drop stale entries from _freeze/ when an included file has changed.

With `freeze: auto` Quarto re-executes a document only when the .qmd itself
changes. Files pulled in with {{< include >}} (the book's exercise .tex files
and the per-exercise .qmd files) are not checked, so edits to them never reach
the site. This script hashes every document's includes, recursively, and
removes the document's _freeze entry when the hash differs from the last run.
Run by the pre-render hook in _quarto.yml, after `make resolve`.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "_freeze"
HASHES = FREEZE / "include-hashes.json"
SKIP = {"docs", "_site", "_freeze", ".quarto", ".git"}
INCLUDE_RE = re.compile(r"\{\{<\s*include\s+(\S+)\s*>\}\}")


def includes(path, seen):
    """All files included by path, recursively, in document order."""
    for target in INCLUDE_RE.findall(path.read_text(errors="replace")):
        inc = ROOT / target.lstrip("/") if target.startswith("/") else path.parent / target
        inc = inc.resolve()
        if inc in seen or not inc.is_file():
            continue
        seen.append(inc)
        includes(inc, seen)
    return seen


def main():
    if not FREEZE.is_dir():
        return
    old = json.loads(HASHES.read_text()) if HASHES.exists() else {}
    new = {}
    for qmd in sorted(ROOT.rglob("*.qmd")):
        rel = qmd.relative_to(ROOT)
        if rel.parts[0] in SKIP:
            continue
        incs = includes(qmd, [])
        if not incs:
            continue
        digest = hashlib.sha256()
        for inc in incs:
            digest.update(str(inc.relative_to(ROOT)).encode())
            digest.update(inc.read_bytes())
        key = str(rel)
        new[key] = digest.hexdigest()
        frozen = FREEZE / rel.with_suffix("")
        # A document seen for the first time is only recorded, not invalidated
        if key in old and old[key] != new[key] and frozen.is_dir():
            print(f"invalidate_freeze: includes of {key} changed, re-executing")
            shutil.rmtree(frozen)
    HASHES.write_text(json.dumps(new, indent=2) + "\n")


if __name__ == "__main__":
    main()
