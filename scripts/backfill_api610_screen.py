"""One-shot backfill: add the precomputed 'API 610 Classically-Stiff Screen'
section to run pages written before the engine started printing it.

New runs get the section from build_wiki_page; this brings the already-saved
corpus up to the same contract, so the chat model can QUOTE a verdict for any
run instead of doing the comparison arithmetic itself (the incident this fixes:
the model put a run's first critical on the wrong side of the 1.2x MCS floor
and reported "no lateral analysis required" for rotors far below it).

Usage:  python scripts/backfill_api610_screen.py [--dry-run]
Idempotent: pages that already have the section are skipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.rotordynamics.report import api610_screen  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parents[1] / "assistant" / "wiki" / "runs"
SECTION_HEADING = "## API 610 Classically-Stiff Screen"

# First data row of the Critical Speeds table: | 1st | 120.5 | 19.2 | 1151 |
_CRIT_ROW = re.compile(r"^\|\s*1(?:st)?\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|")
_NO_CRIT_ROW = re.compile(r"^\|\s*-\s*\|\s*no critical speeds found")
_MCS_ROW = re.compile(r"^\|\s*MCS \(max\. continuous speed\)\s*\|\s*([\d.]+)\s*Hz")


def backfill_page(path: Path, dry_run: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if SECTION_HEADING in text:
        return "skip (already has section)"
    if "## Bearing Reactions" not in text:
        return "skip (unrecognized layout)"

    first_crit_hz: float | None = None
    mcs_hz: float | None = None
    saw_no_crit = False
    for line in text.splitlines():
        if (m := _CRIT_ROW.match(line)) and first_crit_hz is None:
            first_crit_hz = float(m.group(1))
        elif _NO_CRIT_ROW.match(line):
            saw_no_crit = True
        elif (m := _MCS_ROW.match(line)) and mcs_hz is None:
            mcs_hz = float(m.group(1))

    if mcs_hz is None:
        return "skip (no MCS row - page predates the MCS field)"
    if first_crit_hz is None and not saw_no_crit:
        return "skip (could not parse the Critical Speeds table)"

    screen = api610_screen(first_crit_hz, mcs_hz)
    section = "\n".join(screen["markdown_lines"])
    new_text = text.replace("## Bearing Reactions",
                            section + "\n\n## Bearing Reactions", 1)
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return f"added ({screen['verdict']})"


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    if not RUNS_DIR.exists():
        print(f"no runs directory at {RUNS_DIR}")
        return
    for page in sorted(RUNS_DIR.glob("*.md")):
        print(f"{page.name}: {backfill_page(page, dry_run)}")


if __name__ == "__main__":
    main()
