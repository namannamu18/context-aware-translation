"""Compiles every code cell of the notebooks in kaggle/ (lines starting with ! or % are IPython commands and are skipped), so that a
quoting mistake is found here and not on Kaggle.   python scripts/check_notebooks_syntax.py"""
import json
import sys
from pathlib import Path

bad = 0
for f in sorted((Path(__file__).resolve().parent.parent / "kaggle").glob("*.ipynb")):
    for i, c in enumerate(json.load(open(f))["cells"]):
        if c["cell_type"] != "code":
            continue
        lines, cont = [], False
        for l in "".join(c["source"]).splitlines():
            if cont or l.lstrip().startswith(("!", "%")):     # IPython command, possibly continued with a trailing backslash
                cont = l.rstrip().endswith("\\")
                lines.append("pass")
            else:
                lines.append(l)
        src = "\n".join(lines)
        try:
            compile(src, f"{f.name}[cell {i}]", "exec")
        except SyntaxError as e:
            bad += 1
            print(f"SYNTAX ERROR {f.name} cell {i}: {e}")
print("all notebook cells compile" if not bad else f"{bad} cell(s) with errors")
sys.exit(1 if bad else 0)
