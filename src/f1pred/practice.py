"""Clean OpenF1 practice laps and turn them into pre-race pace features.

Cleaning rules (each one counted in the report):
  - laps without a time, pit-out laps, and in-laps (the lap before a pit-out lap) are dropped
  - each lap gets its tyre compound and tyre age from the stint it belongs to; laps outside any stint are dropped
  - laps more than SLOW_LAP_PCT slower than their stint's median (traffic, cool-down) are dropped
  - a long run = a stint with at least MIN_LONGRUN_LAPS clean laps
  - long runs further than MAX_LONGRUN_GAP_PCT from the field (red flags, mislabelled tyres) are dropped
  - relative tyre wear is clipped to +-MAX_DEG_REL s/lap
Features compare drivers only within the same session and compound (soft vs hard isn't comparable).
"""
import numpy as np
import pandas as pd

MIN_LONGRUN_LAPS = 5
SLOW_LAP_PCT = 4.0
# Physical plausibility (set from what race simulations look like, not tuned on results): a genuine long
# run is within 5% of the field's median pace on that compound, and tyre wear differs by < 0.5 s/lap
MAX_LONGRUN_GAP_PCT = 5.0
MAX_DEG_REL = 0.5
PRACTICE_FEATURES = ["fp_has_longrun", "fp_longrun_gap_pct", "fp_deg"]


def map_drivers(drivers: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    """OpenF1 (season, round, driver_number) -> our driver_id via the 3-letter code of that race's entrants.

    Driver lists come from FP1, where rookies often replace regulars, so each weekend's number->code table
    is completed from the rest of that season (a driver keeps his number all year; numbers used by two
    drivers in a season are skipped). Drivers who aren't in the race (FP1 reserves) stay unmapped."""
    own = drivers[["season", "round", "driver_number", "name_acronym"]].drop_duplicates(["season", "round", "driver_number"])
    season_codes = drivers.drop_duplicates(["season", "driver_number", "name_acronym"])
    # only unambiguous pairs: one code per number and one number per code (Bearman raced as #38 and #50 in 2024)
    ambiguous = season_codes.duplicated(["season", "driver_number"], keep=False) | \
        season_codes.duplicated(["season", "name_acronym"], keep=False)
    season_codes = season_codes[~ambiguous][["season", "driver_number", "name_acronym"]]
    filled = own[["season", "round"]].drop_duplicates().merge(season_codes, on="season")
    # the weekend's own list wins: fill only numbers and codes that weekend doesn't already have
    filled = filled.merge(own[["season", "round", "driver_number"]], how="left", indicator=True)
    filled = filled[filled.pop("_merge") == "left_only"]
    filled = filled.merge(own[["season", "round", "name_acronym"]], how="left", indicator=True)
    filled = filled[filled.pop("_merge") == "left_only"]
    d = pd.concat([own, filled], ignore_index=True)
    e = entries[["season", "round", "driver_id", "code"]].drop_duplicates()
    return d.merge(e, left_on=["season", "round", "name_acronym"], right_on=["season", "round", "code"],
                   how="inner", validate="one_to_one")[["season", "round", "driver_number", "driver_id"]]


def clean_laps(laps: pd.DataFrame, stints: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"raw laps": len(laps)}
    x = laps[["season", "round", "session_key", "session_name", "driver_number", "lap_number", "lap_duration",
              "is_pit_out_lap"]].copy()
    x = x[x["lap_duration"].notna()]
    report["no lap time"] = report["raw laps"] - len(x)
    pit_out = x[x["is_pit_out_lap"] == True][["session_key", "driver_number", "lap_number"]]  # noqa: E712
    in_laps = pit_out.assign(lap_number=pit_out["lap_number"] - 1, _in=True)
    x = x.merge(in_laps, on=["session_key", "driver_number", "lap_number"], how="left")
    n = len(x)
    x = x[(x["is_pit_out_lap"] != True) & x["_in"].isna()].drop(columns=["_in", "is_pit_out_lap"])  # noqa: E712
    report["pit-out and in-laps"] = n - len(x)

    st = stints[["session_key", "driver_number", "stint_number", "lap_start", "lap_end", "compound",
                 "tyre_age_at_start"]].dropna(subset=["lap_start", "lap_end"])
    x = x.merge(st, on=["session_key", "driver_number"], how="inner")
    x = x[(x["lap_number"] >= x["lap_start"]) & (x["lap_number"] <= x["lap_end"])]
    report["laps without a stint/compound"] = n - report["pit-out and in-laps"] - len(x)
    x["tyre_age"] = x["tyre_age_at_start"].fillna(0) + x["lap_number"] - x["lap_start"]

    med = x.groupby(["session_key", "driver_number", "stint_number"])["lap_duration"].transform("median")
    keep = x["lap_duration"] <= med * (1 + SLOW_LAP_PCT / 100)
    report[f"slow laps (> {SLOW_LAP_PCT}% over stint median)"] = int((~keep).sum())
    x = x[keep]
    report["clean laps"] = len(x)
    return x.reset_index(drop=True), report


def longrun_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Per (season, round, driver_number): long-run pace gap and tyre degradation vs the field,
    same session and compound only, averaged over a driver's long runs weighted by laps."""
    stint = ["session_key", "driver_number", "stint_number"]
    size = clean.groupby(stint)["lap_duration"].transform("size")
    lr = clean[size >= MIN_LONGRUN_LAPS]

    def slope(g):
        return np.polyfit(g["tyre_age"], g["lap_duration"], 1)[0] if g["tyre_age"].nunique() > 1 else np.nan

    runs = lr.groupby(["season", "round", "session_key", "compound", *stint[1:]]).agg(
        pace=("lap_duration", "median"), laps=("lap_duration", "size")).reset_index()
    runs["deg"] = lr.groupby(["season", "round", "session_key", "compound", *stint[1:]])[["tyre_age", "lap_duration"]] \
        .apply(slope).to_numpy()
    field = runs.groupby(["session_key", "compound"])
    runs["gap_pct"] = (runs["pace"] / field["pace"].transform("median") - 1) * 100
    runs["deg_rel"] = runs["deg"] - field["deg"].transform("median")
    # a single car on a compound has no field to compare with; implausible runs aren't race simulations
    runs = runs[(field["pace"].transform("size") >= 3) & (runs["gap_pct"].abs() <= MAX_LONGRUN_GAP_PCT)].copy()
    runs["deg_rel"] = runs["deg_rel"].clip(-MAX_DEG_REL, MAX_DEG_REL)
    w = runs["laps"]
    out = runs.assign(gw=runs["gap_pct"] * w, dw=runs["deg_rel"].fillna(0) * w).groupby(
        ["season", "round", "driver_number"]).agg(gw=("gw", "sum"), dw=("dw", "sum"), w=("laps", "sum"))
    return pd.DataFrame({"fp_longrun_gap_pct": out["gw"] / out["w"], "fp_deg": out["dw"] / out["w"],
                         "fp_longrun_laps": out["w"]}).reset_index()


def add_practice_features(table: pd.DataFrame, feats: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Join practice features onto the model table. Missing (no OpenF1 data, no long run) -> flag 0, and
    the values set to the field-typical 0 (no gap, typical wear) so linear models aren't fed holes."""
    f = feats.merge(mapping, on=["season", "round", "driver_number"], how="inner", validate="one_to_one")
    out = table.merge(f[["season", "round", "driver_id", "fp_longrun_gap_pct", "fp_deg"]],
                      on=["season", "round", "driver_id"], how="left", validate="one_to_one")
    out.index = table.index
    out["fp_has_longrun"] = out["fp_longrun_gap_pct"].notna().astype(int)
    out[["fp_longrun_gap_pct", "fp_deg"]] = out[["fp_longrun_gap_pct", "fp_deg"]].fillna(0.0)
    return out
