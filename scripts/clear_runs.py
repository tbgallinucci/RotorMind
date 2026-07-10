"""Clear all simulation runs from the wiki: pages, plots, HTML reports, and
their index.md rows. Theory pages are untouched. Run via clear-runs.bat or:
    python scripts/clear_runs.py
"""

import re
import shutil
from datetime import date
from pathlib import Path

WIKI = Path(__file__).resolve().parent.parent / "assistant" / "wiki"
RUNS = WIKI / "runs"
INDEX = WIKI / "index.md"
LOG = WIKI / "log.md"


def main() -> None:
    n_pages = len(list(RUNS.glob("*.md"))) if RUNS.exists() else 0
    if RUNS.exists():
        shutil.rmtree(RUNS)          # pages + plot folders + report.html
        print(f"removed wiki/runs/ ({n_pages} run page(s) and their plots)")
    else:
        print("no wiki/runs/ folder - nothing to remove")

    if INDEX.exists():
        src = INDEX.read_text(encoding="utf-8")
        # drop every '## Runs / <date>' section (up to the next '## ' or EOF)
        cleaned = re.sub(r"\n*## Runs / .*?(?=\n## |\Z)", "", src, flags=re.S)
        if cleaned != src:
            INDEX.write_text(cleaned.rstrip() + "\n", encoding="utf-8")
            print("index.md: run sections removed")
        else:
            print("index.md: no run sections found")

    if LOG.exists():
        with LOG.open("a", encoding="utf-8") as f:
            f.write(f"* **{date.today().isoformat()}** Cleared all simulation "
                    f"runs from the wiki ({n_pages} page(s)).\n")

    print("done - refresh the app (the sidebar reloads on next chat reply or F5)")


if __name__ == "__main__":
    main()
