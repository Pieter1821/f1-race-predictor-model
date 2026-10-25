"""Fetch the OpenF1 snapshot (practice laps/stints, race laps/stints/pits, drivers) for 2023-2026 up to now.

    python scripts/fetch_openf1.py
"""
from datetime import datetime, timezone

import pandas as pd

from f1pred.config import DATA_DIR
from f1pred.data import read_snapshot
from f1pred.openf1 import fetch_snapshot

schedule = read_snapshot(sorted((DATA_DIR / "snapshots").glob("jolpica_2022-2026_*T*"))[-1]).schedule
path = fetch_snapshot(schedule, until=pd.Timestamp(datetime.now(timezone.utc)), log=lambda m: print(m, flush=True))
print(f"saved {path}")
