"""Export the pipeline's real outputs as JSON for the web frontend (web/public/data/).

    python scripts/export_web_data.py --season 2026 --round 18
Run again after the race: if Jolpica has published the result, race.json gains the actual order and
the scorecards are included. The frontend only displays these numbers; it never recomputes them.
"""
import argparse
import json
import math
from datetime import datetime, timezone

import pandas as pd

from f1pred import config, experiments
from f1pred.backtest import walk_forward
from f1pred.data import NON_STARTERS, fetch_season_results
from f1pred.evaluate import SEASONS
from f1pred.models import final_model
from f1pred.pipeline import history_table
from f1pred.validate import predicted_rank, score_many

parser = argparse.ArgumentParser()
parser.add_argument("--season", type=int, required=True)
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--no-fetch", action="store_true", help="don't ask Jolpica for the race result")
args = parser.parse_args()

OUT = config.PROJECT_ROOT / "web" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)
P = config.PREDICTIONS_DIR
OPENF1 = sorted((config.DATA_DIR / "snapshots").glob("openf1_*"))[-1]


def clean(v):
    """JSON-safe: NaN -> None, numpy scalars -> Python."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        return clean(v.item())
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def latest(pattern: str):
    files = sorted(f for f in P.glob(pattern))
    return files[-1] if files else None


data, as_of, history, table = history_table(season=args.season, rnd=args.round)
race_row = data.schedule.set_index(["season", "round"]).loc[(args.season, args.round)]
key = f"{args.season}_R{args.round}"

# ---------------------------------------------------------------- the race: predictions, variant, strategy
official = pd.read_csv(latest(f"{key}_2*.csv"))
variant_f, strategy_f = latest(f"{key}_variant_*.csv"), latest(f"{key}_strategy_*.csv")
drivers = pd.read_parquet(OPENF1 / "drivers.parquet")
names = drivers[(drivers["season"] == args.season) & (drivers["round"] == args.round)] \
    .drop_duplicates("name_acronym").set_index("name_acronym")[["full_name", "team_name"]]
# full names for drivers who skipped FP1: their latest entry this season
season_names = drivers[drivers["season"] == args.season].drop_duplicates("name_acronym", keep="last") \
    .set_index("name_acronym")[["full_name", "team_name"]]
names = names.combine_first(season_names)

race = official.rename(columns={"predicted_position": "predicted"}).copy()
race["name"] = race["code"].map(names["full_name"]).str.title()
race["team_name"] = race["code"].map(names["team_name"])
if variant_f is not None:
    v = pd.read_csv(variant_f).set_index("driver_id")
    race["variant_predicted"] = race["driver_id"].map(v["predicted_position"])
    race["variant_p_win"] = race["driver_id"].map(v["p_win"])
if strategy_f is not None:
    s = pd.read_csv(strategy_f).set_index("driver")
    race["stops"] = race["code"].map(s["stops"])
    race["pit_laps"] = race["code"].map(s["pit window (best plan)"]).map(
        lambda x: [int(p) for p in str(x).split(" & ")] if pd.notna(x) else [])
    race["two_stop_cost_s"] = race["code"].map(s["2-stop is slower by (s)"])
    strategy_meta = {k: clean(s[k].iloc[0]) for k in ("pit_loss_s", "wear_source", "validation", "generated_at_utc",
                                                      "git_commit", "status")}
else:
    strategy_meta = None

actual = None
if not args.no_fetch:
    try:
        res = fetch_season_results(args.season)
        res = res[(res["round"] == args.round) & ~res["status"].isin(NON_STARTERS)]
        if len(res):
            actual = records(res.assign(position=res["position"].rank(method="first").astype(int))
                             [["driver_id", "driver_code", "position", "status", "laps", "grid"]])
    except Exception as e:  # network down or not published: say so in the file, don't guess
        print(f"result not fetched: {e}")

cota = history[history["circuit_id"] == race_row["circuit_id"]]
winners = cota[cota["position"] == 1][["season", "driver_code", "grid", "team"]]
card = P / "scorecard.csv"
scores = records(pd.read_csv(card)[lambda d: (d["season"] == args.season) & (d["round"] == args.round)]) \
    if card.exists() else []
scard = P / "strategy_scorecard.csv"
strategy_scores = records(pd.read_csv(scard)[lambda d: (d["season"] == args.season) & (d["round"] == args.round)]) \
    if scard.exists() else []

race_json = {
    "race": {"season": args.season, "round": args.round, "name": race_row["race_name"],
             "circuit_id": race_row["circuit_id"], "laps": int(history[history["circuit_id"] == race_row["circuit_id"]]["laps"].max()),
             "start_utc": as_of.isoformat(), "qualifying_start_utc": race_row["qualifying_start_utc"].isoformat()},
    "official": {"file": latest(f"{key}_2*.csv").name, "generated_at_utc": official["generated_at_utc"].iloc[0],
                 "git_commit": official["git_commit"].iloc[0], "model": official["model"].iloc[0],
                 "data_source": official["data_source"].iloc[0], "inputs_source": official["inputs_source"].iloc[0]},
    "variant_file": variant_f.name if variant_f is not None else None,
    "strategy": strategy_meta,
    "drivers": records(race.drop(columns=[c for c in race.columns if c in (
        "generated_at_utc", "as_of_utc", "inputs_source", "data_source", "model", "git_commit", "race_name", "season", "round")])),
    "track_history": records(winners.rename(columns={"driver_code": "code"})),
    "actual": actual,
    "scores": scores,
    "strategy_scores": strategy_scores,
}

# ---------------------------------------------------------------- backtests: the locked v1 model, walk-forward 2023-26
features = config.MODEL_FEATURES["v1"]
preds = walk_forward(table, final_model, features, SEASONS)
preds["pred_rank"] = pd.concat([predicted_rank(r, "pred") for _, r in preds.groupby(["season", "round"])])
model_s = score_many(preds, "pred").set_index(["season", "round"])
grid_s = score_many(preds.assign(pred=preds["grid"]), "pred").set_index(["season", "round"])
names_by_race = data.schedule.set_index(["season", "round"])
races = []
for (season, rnd), g in preds.groupby(["season", "round"]):
    m, b = model_s.loc[(season, rnd)], grid_s.loc[(season, rnd)]
    rows = g.sort_values("position")[["driver_code", "team", "grid", "grid_penalty", "pred_rank", "position", "status_bucket"]]
    races.append({"season": int(season), "round": int(rnd), "name": names_by_race.loc[(season, rnd), "race_name"],
                  "circuit_id": names_by_race.loc[(season, rnd), "circuit_id"],
                  "model": {k: clean(m[k]) for k in ("mae", "spearman", "winner_hit", "podium_hit")},
                  "grid": {k: clean(b[k]) for k in ("mae", "spearman", "winner_hit", "podium_hit")},
                  "dnfs": int((~g["finished"].astype(bool)).sum()),
                  "rows": records(rows.rename(columns={"driver_code": "code", "pred_rank": "predicted",
                                                       "status_bucket": "status"}))})
d = model_s["mae"] - grid_s["mae"]
lo, hi = experiments.bootstrap_ci(d)
# The honest headline: only the test seasons, never used for any choice. 2023-24 were used to choose
# features, tuning and the simulation, so their gain is flattered (like the best of 40 dice rolls).
test = d[d.index.get_level_values(0).isin(config.TEST_SEASONS)]
tlo, thi = experiments.bootstrap_ci(test)
test_summary = {"seasons": config.TEST_SEASONS, "races": int(len(test)), "gain": -test.mean(), "ci": [-thi, -tlo],
         "mae_model": model_s["mae"][test.index].mean(), "mae_grid": grid_s["mae"][test.index].mean(),
         "better": int((test < 0).sum()), "worse": int((test > 0).sum())}
by_season = pd.DataFrame({"model": model_s.groupby(level=0)["mae"].mean(), "grid": grid_s.groupby(level=0)["mae"].mean(),
                          "model_spearman": model_s.groupby(level=0)["spearman"].mean(),
                          "grid_spearman": grid_s.groupby(level=0)["spearman"].mean(),
                          "model_winner": model_s.groupby(level=0)["winner_hit"].mean(),
                          "grid_winner": grid_s.groupby(level=0)["winner_hit"].mean(),
                          "races": model_s.groupby(level=0).size()}).reset_index(names="season")
backtests = {"model": "v1 (locked; the Austin prediction's model)", "seasons": SEASONS, "clean": test_summary,
             "dev_seasons": sorted(set(config.DEV_SEASONS) | set(config.SIM_DEV_SEASONS)),
             "summary": {"races": len(races), "mae_model": d.add(grid_s["mae"]).mean(), "mae_grid": grid_s["mae"].mean(),
                         "gain": -d.mean(), "ci": [-hi, -lo], "better": int((d < 0).sum()), "worse": int((d > 0).sum())},
             "by_season": records(by_season), "races": races}

# ---------------------------------------------------------------- the model: calibration, slices, experiments, versions
calib = []
sim_file = config.PROCESSED_DIR / "stage07_sim_v3_2025_26.parquet"
if sim_file.exists():
    sim = pd.read_parquet(sim_file)
    for col, places, label in (("p_win", 1, "Win"), ("p_podium", 3, "Podium"), ("p_points", 10, "Points")):
        b = pd.cut(sim[col], [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0], include_lowest=True)
        g = sim.assign(hit=sim["position"] <= places).groupby(b, observed=True).agg(
            predicted=(col, "mean"), actual=("hit", "mean"), n=("hit", "size")).reset_index(drop=True)
        calib.append({"kind": label, "bins": records(g)})
p = preds.assign(err=(preds["pred_rank"] - preds["position"]).abs())
p["grid_rank"] = pd.concat([predicted_rank(r.assign(g=r["grid"]), "g") for _, r in p.groupby(["season", "round"])])
p["grid_err"] = (p["grid_rank"] - p["position"]).abs()
slices = {"Front row to P3": p["grid"] <= 3, "P4 to P12": p["grid"].between(4, 12), "P13 and back": p["grid"] >= 13,
          "Penalised 5+ places": p["grid_penalty"] >= 5, "Rookies": p["rookie"].astype(bool),
          "Retired (DNF)": ~p["finished"].astype(bool)}
slice_rows = [{"slice": k, "rows": int(m.sum()), "model": p.loc[m, "err"].mean(), "grid": p.loc[m, "grid_err"].mean()}
              for k, m in slices.items()]
exp = pd.read_csv(config.PROJECT_ROOT / "reports" / "experiments.csv")
exp_cols = [c for c in ("date_utc", "change", "seasons", "mae_change", "ci_low", "ci_high", "better_seasons", "decision",
                        "status", "latest")
            if c in exp.columns]
model_json = {"calibration": calib, "slices": records(pd.DataFrame(slice_rows)),
              "experiments": records(exp[exp_cols]),
              "versions": json.loads((config.PROJECT_ROOT / "reports" / "model_versions.json").read_text()),
              "simulation": json.loads((config.PROJECT_ROOT / "reports" / "sim_settings.json").read_text())}

stamp = {"exported_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
for name, obj in (("race", race_json), ("backtests", backtests), ("model", model_json)):
    (OUT / f"{name}.json").write_text(json.dumps({**stamp, **obj}, default=clean, allow_nan=False))
print(f"wrote race.json ({len(race_json['drivers'])} drivers, result {'in' if actual else 'not yet'}), "
      f"backtests.json ({len(races)} races), model.json -> {OUT}")
