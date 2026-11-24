"""Paste the generated LaTeX tables into report/report_final.tex and report/presentation.tex.

Every block between the marker comments

    % >>> TABLE <name>
    % <<< TABLE <name>

is replaced by the contents of ``report_build/tables/<name>.tex`` (written by
``make_tables.py``). Everything outside the markers is left untouched, so the documents
can be edited freely, e.g. in Overleaf, as long as the marker lines are kept.

    python report_build/sync_tables.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "report" / "report_final.tex", ROOT / "report" / "presentation.tex"]
TAB = ROOT / "report_build" / "tables"
BLOCK = re.compile(r"(^% >>> TABLE (\w+)[ \t]*\n)(.*?)(^% <<< TABLE \2[ \t]*$)", re.S | re.M)

for document in DOCUMENTS:
    if not document.exists():
        continue
    used = []

    def fill(match: re.Match) -> str:
        name = match.group(2)
        used.append(name)
        body = (TAB / f"{name}.tex").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
        return match.group(1) + body + "\n" + match.group(4)

    new_text = BLOCK.sub(fill, document.read_text(encoding="utf-8"))
    with open(document, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_text)
    print(f"filled {len(used)} table blocks in {document.relative_to(ROOT)}: {', '.join(used)}")
