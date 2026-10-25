"""OpenF1 (https://openf1.org): free historical lap, stint, pit and weather data from 2023 onwards.

Used for what Jolpica lacks: practice-session pace and tyre wear (known before the race), and race
stints/pit stops (history for strategy). Fetched into immutable snapshots like the Jolpica data.
"""
import time

import pandas as pd
import requests

from f1pred.config import DATA_DIR
from f1pred.data import snapshot_stamp

BASE = "https://api.openf1.org/v1"
PAUSE_S = 0.5            # no published rate limit; stay polite
PRACTICE = ("Practice 1", "Practice 2", "Practice 3")


def get(endpoint: str, retries: int = 5, **params) -> pd.DataFrame:
    """One OpenF1 request as a DataFrame. Retries with backoff on rate limits and server errors;
    any other error raises (never silently returns nothing)."""
    for attempt in range(retries):
        time.sleep(PAUSE_S)
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt * 2)
            continue
        if r.status_code == 404:  # OpenF1 answers 404 when a filter matches nothing
            return pd.DataFrame()
        r.raise_for_status()
        return pd.DataFrame(r.json())
    raise RuntimeError(f"OpenF1 {endpoint} {params}: still failing after {retries} attempts")


def match_sessions(sessions: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach our (season, round) to OpenF1 sessions: a meeting's Race starts at our calendar's start time.
    Meetings without a race on our calendar (e.g. testing) are dropped."""
    races = sessions[sessions["session_name"] == "Race"].copy()
    races["start_utc"] = pd.to_datetime(races["date_start"], utc=True)
    cal = schedule[["season", "round", "start_utc"]].copy()
    cal["start_utc"] = pd.to_datetime(cal["start_utc"], utc=True)
    m = pd.merge_asof(races.sort_values("start_utc"), cal.sort_values("start_utc"), on="start_utc",
                      direction="nearest", tolerance=pd.Timedelta("30h"))
    # 30h, not minutes: Jolpica dates Las Vegas 2024 a day early (local date + UTC time); weekends are 7 days apart
    m = m.dropna(subset=["season"])[["meeting_key", "season", "round"]]
    m[["season", "round"]] = m[["season", "round"]].astype(int)
    assert m["meeting_key"].is_unique and not m.duplicated(["season", "round"]).any(), "ambiguous meeting match"
    return sessions.merge(m, on="meeting_key", how="inner", validate="many_to_one")


def fetch_snapshot(schedule: pd.DataFrame, seasons=(2023, 2024, 2025, 2026), until: pd.Timestamp | None = None,
                   log=print):
    """Everything we use, for every matched meeting up to `until`. Returns the snapshot folder."""
    sessions = match_sessions(pd.concat([get("sessions", year=y) for y in seasons], ignore_index=True), schedule)
    if until is not None:
        sessions = sessions[pd.to_datetime(sessions["date_start"], utc=True) < until]
    tables = {k: [] for k in ("drivers", "practice_laps", "practice_stints", "race_laps", "race_stints", "race_pits")}
    meetings = sessions.groupby(["season", "round", "meeting_key"])
    for i, ((season, rnd, mk), ss) in enumerate(meetings, 1):
        keyed = lambda df, s: df.assign(session_key=s["session_key"], session_name=s["session_name"], season=season, round=rnd)
        first = ss.sort_values("date_start").iloc[0]
        tables["drivers"].append(keyed(get("drivers", session_key=first["session_key"]), first))
        for _, s in ss.iterrows():
            if s["session_name"] in PRACTICE:
                tables["practice_laps"].append(keyed(get("laps", session_key=s["session_key"]), s))
                tables["practice_stints"].append(keyed(get("stints", session_key=s["session_key"]), s))
            elif s["session_name"] == "Race":
                for name, ep in (("race_laps", "laps"), ("race_stints", "stints"), ("race_pits", "pit")):
                    tables[name].append(keyed(get(ep, session_key=s["session_key"]), s))
        log(f"  [{i}/{len(meetings)}] {season} R{rnd}: {len(ss)} sessions")
    path = DATA_DIR / "snapshots" / f"openf1_{min(seasons)}-{max(seasons)}_{snapshot_stamp()}"
    path.mkdir(parents=True, exist_ok=False)
    sessions.to_csv(path / "sessions.csv", index=False)
    for name, parts in tables.items():
        df = pd.concat([p.dropna(axis=1, how="all") for p in parts if not p.empty], ignore_index=True)
        df.to_parquet(path / f"{name}.parquet", index=False)
    return path
