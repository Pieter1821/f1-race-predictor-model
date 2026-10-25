"""End-to-end prediction for one race: inputs -> features -> locked models -> simulation -> table.

Used by scripts/predict_race.py. Nothing here is specific to one race: the race is identified by
(season, round) and everything is cut off at that race's scheduled start (as_of).
"""
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from f1pred.config import MANUAL_DIR, PROJECT_ROOT
from f1pred.data import NON_STARTERS, TEAM_LINEAGE, RawData
from f1pred.features import append_target_race, build_features
from f1pred.models import FINAL_FEATURES, FINISHER_TARGET, add_finisher_target, final_model, final_sim_model
from f1pred.race_inputs import load_race_inputs
from f1pred.simulate import SimSettings, add_dnf_probability, race_probabilities, temper_win
from f1pred.validate import predicted_rank

SIM_SETTINGS_PATH = PROJECT_ROOT / "reports" / "sim_settings.json"
INPUT_COLUMNS = ["season", "round", "grid", "driver_id", "code", "constructor_id", "quali_position",
                 "q1", "q2", "q3", "team", "grid_penalty"]


def race_inputs(season: int, rnd: int, data: RawData, history: pd.DataFrame, as_of: pd.Timestamp,
                backtest: bool) -> tuple[pd.DataFrame, str]:
    """Qualifying and final grid for the race being predicted, plus a label saying where they came from.

    Live: the hand-entered CSVs (strictly validated) when present. Jolpica publishes no pre-race
    grid, so without a manual grid the qualifying order is used as the grid, with a warning.
    Backtest: the starting grid recorded in the race result (known before lights out).
    """
    schedule = data.schedule.set_index(["season", "round"])
    quali_start = schedule.loc[(season, rnd), "qualifying_start_utc"]
    quali = data.qualifying[(data.qualifying["season"] == season) & (data.qualifying["round"] == rnd)]

    if backtest:
        res = data.results[(data.results["season"] == season) & (data.results["round"] == rnd)]
        res = res[~res["status"].isin(NON_STARTERS)]
        df = res[["driver_id", "driver_code", "constructor_id", "grid"]].rename(columns={"driver_code": "code"})
        df["grid"] = df["grid"].where(df["grid"] > 0, len(df))  # pit-lane start = back of the grid
        df = df.merge(quali[["driver_id", "quali_position", "q1", "q2", "q3"]], on="driver_id", how="left")
        source = "backtest: grid from the race result, qualifying from Jolpica"
    elif (MANUAL_DIR / f"{season}_R{rnd}_grid.csv").exists():
        df = load_race_inputs(season, rnd, history, quali_start, as_of)
        source = f"manual CSVs data/manual/{season}_R{rnd}_*.csv (validated)"
        if len(quali):
            j = quali.set_index("driver_id")["quali_position"]
            m = df.set_index("driver_id")["quali_position"]
            diff = m[m != j.reindex(m.index)]
            source += f"; Jolpica qualifying also available, {len(diff)} position differences"
    elif len(quali):
        df = quali.rename(columns={"position": "quali_position"}).copy()
        df["grid"] = df["quali_position"].astype(int)
        df["code"] = df["driver_id"].str[:3].str.upper()
        source = "Jolpica qualifying, grid ASSUMED = qualifying order (no penalties known)"
    else:
        raise SystemExit(f"No inputs for {season} R{rnd}: add data/manual/{season}_R{rnd}_grid.csv and "
                         f"_qualifying.csv, or wait for Jolpica to publish qualifying.")

    df = df.assign(season=season, round=rnd)
    df["team"] = df["constructor_id"].replace(TEAM_LINEAGE)
    df["grid_penalty"] = df["grid"] - df["quali_position"]
    return df[INPUT_COLUMNS].reset_index(drop=True), source


def git_commit() -> str:
    """Commit of the code that made the prediction; '+uncommitted' if the code differs from it."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        dirty = git("status", "--porcelain", "--", "src", "scripts", "reports")
        return git("rev-parse", "--short", "HEAD") + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def predict(history: pd.DataFrame, inputs: pd.DataFrame, circuit_id: str, as_of: pd.Timestamp,
            n_sims: int = 20_000) -> pd.DataFrame:
    """Train the locked models on every earlier race, predict the target race, simulate it."""
    table = build_features(append_target_race(history, inputs, circuit_id, as_of.normalize().tz_localize(None)))
    table = add_dnf_probability(add_finisher_target(table))
    train = table[~table["is_target"] & table["position"].notna()].sort_values(["season", "round", "driver_id"])
    target = table[table["is_target"]].sort_values("driver_id").copy()
    assert not target[FINAL_FEATURES].isna().any().any(), "target race has missing features"

    # 1. Order model (Stage 5, locked): predicted finishing order
    order = final_model().fit(train, FINAL_FEATURES)
    target["order_score"] = order.predict(target, FINAL_FEATURES)
    target["predicted_position"] = predicted_rank(target, "order_score")

    # 2. Simulation (v3.1): finishers-only pace model + DNF dice + win calibration
    s = json.loads(SIM_SETTINGS_PATH.read_text())
    pace = final_sim_model(target=s["target"]).fit(train[train[FINISHER_TARGET].notna()], FINAL_FEATURES)
    target["pred"] = pace.predict(target, FINAL_FEATURES)
    sim = race_probabilities(target, SimSettings(grid_blend=s["grid_blend"], sigma=s["sigma"],
                                                 spacing=s["spacing"], n_sims=n_sims))
    sim = temper_win(sim, s["win_temperature"])
    sim.attrs["sim_version"] = s["version"]
    sim.attrs["train_races"] = train.groupby(["season", "round"]).ngroups
    return sim


def prediction_table(sim: pd.DataFrame, race_name: str, as_of: pd.Timestamp, inputs_source: str,
                     data_source: str) -> pd.DataFrame:
    cols = ["season", "round", "driver_id", "code", "team", "grid", "quali_position", "grid_penalty",
            "predicted_position", "expected_position", "p_win", "p_podium", "p_points", "p_dnf"]
    out = sim[cols].sort_values("predicted_position").reset_index(drop=True)
    for c in ("expected_position", "p_win", "p_podium", "p_points", "p_dnf"):
        out[c] = out[c].round(4)
    return out.assign(
        race_name=race_name,
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        as_of_utc=as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
        inputs_source=inputs_source,
        data_source=data_source,
        model="order: ridge + tuned lightgbm (rank ensemble); sim: v" + str(sim.attrs["sim_version"]),
        train_races=sim.attrs["train_races"],
        git_commit=git_commit(),
    )
