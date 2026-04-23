"""
Pull the latest CSVs from Greco1899/scrape_ufc_stats.
Clones on first run, then does `git pull` on subsequent runs.
Run this daily (or via GitHub Actions) before greco_loader.py.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/Greco1899/scrape_ufc_stats.git"
DEFAULT_TARGET = Path("data/raw/greco1899")


def pull(target: Path | None = None) -> Path:
    dest = Path(target or DEFAULT_TARGET)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if (dest / ".git").exists():
        log.info("Pulling latest CSVs from Greco1899 repo…")
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True)
    else:
        log.info("Cloning Greco1899 repo → %s", dest)
        subprocess.run(["git", "clone", "--depth=1", REPO_URL, str(dest)], check=True)

    csv_files = list(dest.glob("*.csv"))
    log.info("CSVs available: %s", [f.name for f in csv_files])
    return dest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pull()
